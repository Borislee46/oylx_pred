from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.pages.prediction.config.ui_messages import (
    EXPERIENCE_BOOST_TEMPLATE,
    EXPERIENCE_ITEM_NAMES,
)
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
from src.pages.prediction.result_modifier.ui_handler import LoadingMessageAnimator
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
        arbitrator: AdjustmentArbitrator | None = None,
    ) -> dict[str, Any]:
        current_prob = get_probability(result)

        if arbitrator is None:
            arbitrator = AdjustmentArbitrator()
        else:
            arbitrator.reset(keep_static=True)

        if self.enable_cross_major_penalty and ctx.background_major:
            similarity = float(result.get("similarity", 1.0))
            if similarity < MIN_SIMILARITY_THRESHOLD:
                key = (result.get("university"), result.get("major"))
                has_admitted = key in ctx.admitted_combinations or result.get("admitted") == 1
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
            target_faculty = result.get("faculty")
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
            major = str(result.get("major", "")).lower()
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

        res = result.copy()
        if arbitrator.trace:
            res["_adjustment_trace"] = arbitrator.trace

        res["probability"] = NormalizationLayer.apply(adjusted_prob)

        univ, major_name = res.get("university"), res.get("major")
        if univ and major_name:
            res["is_new_major"] = ctx.is_new_major_cache.get((univ, major_name), False)

        return res

    def adjust_batch(
        self,
        results: list[dict[str, Any]],
        ctx: AdjustmentContext,
        progress_reporter: Any | None = None,
        batch_tag: str = "",
    ) -> list[dict[str, Any]]:
        if not results:
            return results

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
                    ),
                    is_static=True,
                )
            if penalties.get("language", 0) > 0:
                arbitrator.add_factor(
                    AdjustmentFactor(
                        name="Language Penalty",
                        value=penalties["language"],
                        factor_type=AdjustmentFactorType.PENALTY,
                        description="语言成绩影响",
                    ),
                    is_static=True,
                )

        adjusted_results = [self.adjust_single(r, ctx, arbitrator) for r in results]

        if self.text_boost_provider and ctx.experience_details:
            items = [
                name for k, name in EXPERIENCE_ITEM_NAMES.items() if ctx.experience_details.get(k)
            ]

            if items:
                msg = EXPERIENCE_BOOST_TEMPLATE.format(items="、".join(items))
                animator = LoadingMessageAnimator(progress_reporter=progress_reporter)
                animator.show(msg, force=True)

                adjusted_results = self._apply_text_boost(adjusted_results, ctx.experience_details)

                animator.clear()

        return adjusted_results

    def _apply_text_boost(
        self,
        results: list[dict[str, Any]],
        experience_details: dict[str, str],
    ) -> list[dict[str, Any]]:
        if not self.text_boost_provider:
            return results

        probabilities = [r.get("probability", 0.0) for r in results]

        boosted_probs = self.text_boost_provider.apply(probabilities, experience_details)

        if boosted_probs:
            for i, prob in enumerate(boosted_probs):
                if i < len(results):
                    new_prob = clip_probability(prob)
                    if abs(new_prob - probabilities[i]) < 1e-6:
                        continue

                    res = results[i].copy()
                    old_prob = probabilities[i]
                    res["probability"] = new_prob
                    trace = dict(res.get("_adjustment_trace", {}))
                    trace["boost_NLP_Text"] = new_prob - old_prob
                    res["_adjustment_trace"] = trace
                    results[i] = res

        return results
