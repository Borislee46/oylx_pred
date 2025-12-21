import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.pages.prediction.result_modifier.config import (
    FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR,
    MIN_SIMILARITY_THRESHOLD,
    PROFESSIONAL_MAJORS_LOWER,
    PROFESSIONAL_REDUCTION_FACTOR,
    PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR,
)
from src.pages.prediction.result_modifier.faculty_filters import apply_out_of_scope_faculty_penalty
from src.pages.prediction.result_modifier.probability_adjuster import ProbabilityAdjuster
from src.pages.prediction.result_modifier.text_boost_provider import TextBoostProvider
from src.pages.prediction.result_modifier.utils import (
    apply_cross_major_penalty_if_needed,
    clip_probability,
    get_probability,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


@dataclass
class AdjustmentContext:
    gpa: float | None = None
    language_score: float | None = None
    background_university: str | None = None
    background_major: str | None = None
    background_faculty: str | None = None
    internship_count: int = 0
    user_specified_majors: list[str] = field(default_factory=list)
    experience_details: dict[str, str] = field(default_factory=dict)
    cases_df: pd.DataFrame | None = None
    admitted_combinations: set[tuple[str, str]] = field(default_factory=set)
    is_new_major_cache: dict[tuple[str, str], bool] = field(default_factory=dict)


class ProbabilityAdjustmentPipeline:
    def __init__(
        self,
        probability_adjuster: ProbabilityAdjuster | None = None,
        text_boost_provider: TextBoostProvider | None = None,
        enable_cross_major_penalty: bool = True,
    ):
        self.probability_adjuster = probability_adjuster
        self.text_boost_provider = text_boost_provider
        self.enable_cross_major_penalty = enable_cross_major_penalty

    def adjust_single(
        self,
        result: dict[str, Any],
        ctx: AdjustmentContext,
    ) -> dict[str, Any]:
        result_copy = result.copy()
        current_prob = get_probability(result_copy)

        if self.probability_adjuster and ctx.gpa is not None and ctx.language_score is not None:
            current_prob = self._apply_gpa_language_adjustment(
                current_prob, ctx.gpa, ctx.language_score, ctx.background_university
            )

        if self.enable_cross_major_penalty and ctx.background_major:
            current_prob = self._apply_cross_major_penalty(
                result_copy, current_prob, ctx.admitted_combinations
            )

        if ctx.background_faculty:
            temp = result_copy.copy()
            temp["probability"] = clip_probability(current_prob)
            adjusted = apply_out_of_scope_faculty_penalty(
                [temp], ctx.background_faculty, FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR
            )
            if adjusted:
                current_prob = get_probability(adjusted[0], current_prob)

        if ctx.internship_count <= 0:
            major = str(result_copy.get("major", "")).lower()
            if any(p in major for p in PROFESSIONAL_MAJORS_LOWER):
                is_spec = any(s.lower() in major for s in ctx.user_specified_majors)
                factor = (
                    PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR
                    if is_spec
                    else PROFESSIONAL_REDUCTION_FACTOR
                )
                current_prob *= factor

        univ, major = result_copy.get("university"), result_copy.get("major")
        if univ and major:
            result_copy["is_new_major"] = ctx.is_new_major_cache.get((univ, major), False)

        result_copy["probability"] = clip_probability(current_prob)
        return result_copy

    def adjust_batch(
        self,
        results: list[dict[str, Any]],
        ctx: AdjustmentContext,
        progress_reporter: Any | None = None,
    ) -> list[dict[str, Any]]:
        if not results:
            return results

        def _process():
            res = [self.adjust_single(r, ctx) for r in results]
            if self.text_boost_provider and ctx.experience_details:
                res = self._apply_text_boost(res, ctx.experience_details)
            return res

        if progress_reporter and self.text_boost_provider and ctx.experience_details:
            from src.pages.prediction.config.ui_messages import (
                EXPERIENCE_BOOST_TEMPLATE,
                EXPERIENCE_DEFAULT_MSG,
                EXPERIENCE_ITEM_NAMES,
            )
            from src.pages.prediction.result_modifier.ui_handler import LoadingMessageAnimator

            items = [
                name for k, name in EXPERIENCE_ITEM_NAMES.items() if ctx.experience_details.get(k)
            ]
            msg = (
                EXPERIENCE_BOOST_TEMPLATE.format(items="、".join(items))
                if items
                else EXPERIENCE_DEFAULT_MSG
            )

            animator = LoadingMessageAnimator(progress_reporter=progress_reporter)
            animator.show(msg, force=True)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_process)
                while not future.done():
                    animator.tick()
                    time.sleep(0.3)
                final_res = future.result()
            animator.clear()
            return final_res

        return _process()

    def _apply_gpa_language_adjustment(
        self,
        probability: float,
        gpa: float,
        language_score: float,
        background_university: str | None,
    ) -> float:
        if self.probability_adjuster is None:
            return probability
        try:
            return self.probability_adjuster.adjust_probability(
                probability, gpa, language_score, background_university_name=background_university
            )
        except (TypeError, ValueError, OverflowError) as e:
            logger.warning(f"GPA/语言成绩调整失败: {e}")
            return probability
        except (AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"GPA/语言成绩调整发生未知错误: {e}", exc_info=True)
            return probability

    def _apply_cross_major_penalty(
        self,
        result: dict,
        probability: float,
        admitted_combinations: set[tuple[str, str]],
    ) -> float:
        if result.get("similarity", 1.0) >= MIN_SIMILARITY_THRESHOLD:
            return probability
        return apply_cross_major_penalty_if_needed(
            result=result,
            probability=probability,
            admitted_combinations=admitted_combinations,
            check_admitted_field=True,
        )

    def _apply_text_boost(
        self,
        results: list[dict[str, Any]],
        experience_details: dict[str, str],
    ) -> list[dict[str, Any]]:
        if not self.text_boost_provider:
            return results

        probabilities = [get_probability(r) for r in results]

        try:
            apply_result = self.text_boost_provider.apply(probabilities, experience_details)
            if apply_result:
                boosted_probs, boost_info = apply_result
                if boost_info:
                    logger.debug(f"文本增强应用成功: {boost_info}")
                for i, prob in enumerate(boosted_probs):
                    if i < len(results):
                        results[i]["probability"] = clip_probability(prob)
        except (TypeError, ValueError) as e:
            logger.warning(f"文本增强失败: {e}")
        except (AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"文本增强发生未知错误: {e}", exc_info=True)

        return results
