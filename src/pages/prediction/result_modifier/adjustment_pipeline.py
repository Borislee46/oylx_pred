from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.pages.prediction.result_modifier.config import (
    FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR,
    MIN_SIMILARITY_THRESHOLD,
)
from src.pages.prediction.result_modifier.faculty_filters import apply_out_of_scope_faculty_penalty
from src.pages.prediction.result_modifier.probability_adjuster import ProbabilityAdjuster
from src.pages.prediction.result_modifier.text_boost_provider import TextBoostProvider
from src.pages.prediction.result_modifier.utils import clip_probability, cross_major_penalty_factor
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
        original_prob = self._get_probability(result_copy)
        current_prob = original_prob

        if self.probability_adjuster and ctx.gpa is not None and ctx.language_score is not None:
            current_prob = self._apply_gpa_language_adjustment(
                current_prob, ctx.gpa, ctx.language_score, ctx.background_university
            )

        if self.enable_cross_major_penalty and ctx.background_major:
            current_prob = self._apply_cross_major_penalty(
                result_copy, current_prob, ctx.background_major, ctx.admitted_combinations
            )

        if ctx.background_faculty:
            temp = result_copy.copy()
            temp["probability"] = clip_probability(current_prob)
            adjusted = apply_out_of_scope_faculty_penalty(
                [temp], ctx.background_faculty, FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR
            )
            if adjusted:
                current_prob = float(adjusted[0].get("probability", current_prob) or current_prob)

        result_copy["probability"] = clip_probability(current_prob)
        return result_copy

    def adjust_batch(
        self,
        results: list[dict[str, Any]],
        ctx: AdjustmentContext,
    ) -> list[dict[str, Any]]:
        if not results:
            return results

        adjusted_results = [self.adjust_single(r, ctx) for r in results]

        if self.text_boost_provider and ctx.experience_details:
            adjusted_results = self._apply_text_boost(adjusted_results, ctx.experience_details)

        return adjusted_results

    def _get_probability(self, result: dict) -> float:
        prob = result.get("probability", 0.0)
        try:
            val = float(prob)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0

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
        except Exception as e:
            logger.warning(f"GPA/语言成绩调整失败: {e}")
            return probability

    def _apply_cross_major_penalty(
        self,
        result: dict,
        probability: float,
        background_major: str,
        admitted_combinations: set[tuple[str, str]],
    ) -> float:
        similarity = result.get("similarity", 1.0)
        is_cross_major = similarity < MIN_SIMILARITY_THRESHOLD

        if not is_cross_major:
            return probability

        key = (result.get("university"), result.get("major"))
        has_admitted_case = bool(result.get("admitted") == 1 or key in admitted_combinations)

        if has_admitted_case:
            return probability

        return probability * cross_major_penalty_factor(similarity)

    def _apply_text_boost(
        self,
        results: list[dict[str, Any]],
        experience_details: dict[str, str],
    ) -> list[dict[str, Any]]:
        if not self.text_boost_provider:
            return results

        probabilities = [self._get_probability(r) for r in results]

        try:
            apply_result = self.text_boost_provider.apply(probabilities, experience_details)
            if apply_result:
                boosted_probs, boost_info = apply_result
                if boost_info:
                    logger.debug(f"文本增强应用成功: {boost_info}")
                for i, prob in enumerate(boosted_probs):
                    if i < len(results):
                        results[i]["probability"] = clip_probability(prob)
        except Exception as e:
            logger.warning(f"文本增强失败: {e}")

        return results
