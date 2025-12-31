import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.pages.prediction.result_modifier.arbitrator import (
    AdjustmentArbitrator,
    NormalizationLayer,
)
from src.pages.prediction.result_modifier.config import (
    FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR,
    MIN_SIMILARITY_THRESHOLD,
    PROFESSIONAL_MAJORS_LOWER,
    PROFESSIONAL_REDUCTION_FACTOR,
    PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR,
)
from src.pages.prediction.result_modifier.faculty_filters import is_faculty_out_of_scope
from src.pages.prediction.result_modifier.probability_adjuster import ProbabilityAdjuster
from src.pages.prediction.result_modifier.text_boost_provider import TextBoostProvider
from src.pages.prediction.result_modifier.types import AdjustmentFactor, AdjustmentFactorType
from src.pages.prediction.result_modifier.utils import (
    clip_probability,
    cross_major_penalty_factor,
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

        arbitrator = AdjustmentArbitrator()

        if self.probability_adjuster and ctx.gpa is not None and ctx.language_score is not None:
            penalties = self.probability_adjuster.get_penalties(ctx.gpa, ctx.language_score)
            if penalties.get("gpa", 0) > 0:
                arbitrator.add_factor(
                    AdjustmentFactor(
                        name="GPA Penalty",
                        value=penalties["gpa"],
                        factor_type=AdjustmentFactorType.PENALTY,
                        description="GPA成绩影响",
                    )
                )
            if penalties.get("language", 0) > 0:
                arbitrator.add_factor(
                    AdjustmentFactor(
                        name="Language Penalty",
                        value=penalties["language"],
                        factor_type=AdjustmentFactorType.PENALTY,
                        description="语言成绩影响",
                    )
                )

        if self.enable_cross_major_penalty and ctx.background_major:
            similarity = float(result_copy.get("similarity", 1.0))
            if similarity < MIN_SIMILARITY_THRESHOLD:
                key = (result_copy.get("university"), result_copy.get("major"))
                has_admitted = key in ctx.admitted_combinations or result_copy.get("admitted") == 1
                if not has_admitted:
                    p_factor = 1.0 - cross_major_penalty_factor(similarity)
                    if p_factor > 0:
                        arbitrator.add_factor(
                            AdjustmentFactor(
                                name="Cross Major Penalty",
                                value=p_factor,
                                factor_type=AdjustmentFactorType.PENALTY,
                                description=f"背景相似度低 ({similarity:.2f})",
                            )
                        )

        if ctx.background_faculty:
            target_faculty = result_copy.get("faculty")
            if is_faculty_out_of_scope(ctx.background_faculty, target_faculty):
                arbitrator.add_factor(
                    AdjustmentFactor(
                        name="Faculty Out of Scope Penalty",
                        value=1.0 - FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR,
                        factor_type=AdjustmentFactorType.PENALTY,
                        description="申请学部跨度过大",
                    )
                )

        if ctx.internship_count <= 0:
            major = str(result_copy.get("major", "")).lower()
            if any(p in major for p in PROFESSIONAL_MAJORS_LOWER):
                is_spec = any(s.lower() in major for s in ctx.user_specified_majors)
                reduction_ratio = 1.0 - (
                    PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR
                    if is_spec
                    else PROFESSIONAL_REDUCTION_FACTOR
                )
                if reduction_ratio > 0:
                    arbitrator.add_factor(
                        AdjustmentFactor(
                            name="Professional Major Penalty",
                            value=reduction_ratio,
                            factor_type=AdjustmentFactorType.PENALTY,
                            description="专业项目缺乏实习背景",
                        )
                    )

        adjusted_prob = arbitrator.arbitrate(current_prob)

        if arbitrator.trace:
            result_copy["_adjustment_trace"] = arbitrator.trace

        result_copy["probability"] = NormalizationLayer.apply(adjusted_prob)

        univ, major = result_copy.get("university"), result_copy.get("major")
        if univ and major:
            result_copy["is_new_major"] = ctx.is_new_major_cache.get((univ, major), False)

        return result_copy

    def adjust_batch(
        self,
        results: list[dict[str, Any]],
        ctx: AdjustmentContext,
        progress_reporter: Any | None = None,
        batch_tag: str = "",
    ) -> list[dict[str, Any]]:
        if not results:
            return results

        def _process():
            res = [self.adjust_single(r, ctx) for r in results]
            if self.text_boost_provider and ctx.experience_details:
                res = self._apply_text_boost(res, ctx.experience_details)

            self._log_adjustment_deltas(res, ctx, batch_tag)
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

    def _apply_text_boost(
        self,
        results: list[dict[str, Any]],
        experience_details: dict[str, str],
    ) -> list[dict[str, Any]]:
        if not self.text_boost_provider:
            return results

        probabilities = [get_probability(r) for r in results]

        try:
            boosted_probs = self.text_boost_provider.apply(probabilities, experience_details)
            if boosted_probs:
                for i, prob in enumerate(boosted_probs):
                    if i < len(results):
                        old_prob = probabilities[i]
                        new_prob = clip_probability(prob)
                        results[i]["probability"] = new_prob

                        trace = results[i].get("_adjustment_trace", {})
                        trace["boost_NLP_Text"] = new_prob - old_prob
                        results[i]["_adjustment_trace"] = trace

        except (TypeError, ValueError) as e:
            logger.warning(f"文本增强失败: {e}")
        except (AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"文本增强发生未知错误: {e}", exc_info=True)

        return results

    def _log_adjustment_deltas(
        self,
        results: list[dict[str, Any]],
        ctx: AdjustmentContext,
        batch_tag: str = "",
        sample_size: int = 5,
    ):
        if not results:
            return

        sorted_res = sorted(results, key=lambda x: x.get("probability", 0), reverse=True)
        samples = sorted_res[:sample_size]

        for i, r in enumerate(samples):
            trace = r.get("_adjustment_trace", {})
            if not trace:
                continue

            trace_str = ", ".join(
                [f"{k}: {v:+.4f}" for k, v in trace.items() if k != "base" and k != "final"]
            )
            logger.info(
                f"监控样本 {i + 1} | {r.get('university')} - {r.get('major')} | "
                f"原始概率: {trace.get('base', 0):.4f} | "
                f"修正Delta: [{trace_str}] | "
                f"修正后概率: {r.get('probability', 0):.4f}"
            )

        for r in results:
            r.pop("_adjustment_trace", None)
