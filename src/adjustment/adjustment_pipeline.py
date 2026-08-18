from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.adjustment.admission_cache import (
    get_cross_major_admission_stats,
)
from src.adjustment.arbitrator import (
    AdjustmentArbitrator,
    NormalizationLayer,
)
from src.adjustment.config import (
    BAYESIAN_SHRINKAGE_GLOBAL_PRIOR,
    BAYESIAN_SHRINKAGE_PRIOR_STRENGTH,
    CROSS_MAJOR_EVIDENCE_MIN_CASES,
    CROSS_MAJOR_EVIDENCE_PRIOR_STRENGTH,
    CROSS_MAJOR_EVIDENCE_RATIO_THRESHOLD,
    CROSS_MAJOR_SIGMOID_MIDPOINT,
    CROSS_MAJOR_SIGMOID_STEEPNESS,
    PROFESSIONAL_MAJORS_LOWER,
    PROFESSIONAL_REDUCTION_FACTOR,
    PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR,
)
from src.adjustment.counterfactual import attach_trace_extras
from src.adjustment.engine import AdjustmentFactor, AdjustmentFactorType
from src.adjustment.faculty_filters import (
    get_cross_faculty_penalty_factor,
    is_faculty_out_of_scope,
)
from src.adjustment.probability_adjuster import ProbabilityAdjuster
from src.adjustment.text_boost_provider import TextBoostProvider
from src.adjustment.utils import (
    compute_school_stats,
    cross_major_penalty_factor_sigmoid,
    get_probability,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability, clip_probability_coerce, clip_scalar

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
    cross_major_stats: dict[tuple[str, str], dict] | None = None
    school_stats: dict[str, dict[str, float]] | None = None
    baseline_admit_lookup: dict[tuple[str, str], tuple[float, int]] | None = None


class ProbabilityAdjustmentPipeline:
    def __init__(
        self,
        probability_adjuster: ProbabilityAdjuster | None = None,
        text_boost_provider: TextBoostProvider | None = None,
        enable_gpa_penalty: bool = True,
        enable_language_penalty: bool = True,
        enable_cross_major_penalty: bool = True,
        enable_cross_faculty_penalty: bool = True,
        enable_professional_penalty: bool = True,
        enable_text_boost: bool = True,
        ablation_tag: str = "",
        quality_verifier: Any | None = None,
    ):
        self.probability_adjuster = probability_adjuster
        self.text_boost_provider = text_boost_provider
        self.quality_verifier = quality_verifier
        self.enable_gpa_penalty = enable_gpa_penalty
        self.enable_language_penalty = enable_language_penalty
        self.enable_cross_major_penalty = enable_cross_major_penalty
        self.enable_cross_faculty_penalty = enable_cross_faculty_penalty
        self.enable_professional_penalty = enable_professional_penalty
        self.enable_text_boost = enable_text_boost
        self.ablation_tag = ablation_tag

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
            raw_similarity = result.get("similarity", 1.0)
            similarity = float(raw_similarity if raw_similarity is not None else 1.0)
            multiplier = cross_major_penalty_factor_sigmoid(
                similarity, k=CROSS_MAJOR_SIGMOID_STEEPNESS, midpoint=CROSS_MAJOR_SIGMOID_MIDPOINT
            )
            if multiplier < 1.0 - 1e-9:
                base_penalty = 1.0 - multiplier
                p_factor = self._adjust_cross_major_by_evidence(result, base_penalty, ctx)
                target_uni = str(result.get("university", ""))
                if target_uni and ctx.school_stats and p_factor > 0:
                    school = ctx.school_stats.get(target_uni)
                    if school:
                        difficulty = school.get("difficulty", 0.5)
                        # 目标校难度缩放：difficulty 是跨学校 min-max 归一化值（越高越难）。
                        # 语义：目标校越容易，跨专业惩罚比例越低（惩罚×难度）。
                        # 注意：该交互无独立开关，且生产默认 enable_cross_major_penalty=false，
                        # 仅在消融/实验启用；如需调整请先评审此处设计。
                        p_factor *= difficulty
                if p_factor > 0:
                    arbitrator.add_factor(
                        AdjustmentFactor(
                            name="Cross Major Penalty",
                            value=p_factor,
                            factor_type=AdjustmentFactorType.PENALTY,
                            description=f"背景相似度 ({similarity:.2f})",
                        )
                    )

        if self.enable_cross_faculty_penalty and ctx.background_faculty:
            target_faculty = result.get("faculty")
            if is_faculty_out_of_scope(ctx.background_faculty, target_faculty):
                penalty_factor = get_cross_faculty_penalty_factor(
                    ctx.background_faculty, target_faculty
                )
                severity_label = {
                    0.70: "轻度",
                    0.50: "中度",
                    0.30: "重度",
                }.get(penalty_factor, "重度")
                arbitrator.add_factor(
                    AdjustmentFactor(
                        name="Faculty Out of Scope Penalty",
                        value=1.0 - penalty_factor,
                        factor_type=AdjustmentFactorType.PENALTY,
                        description=f"学部跨度{severity_label}（×{penalty_factor:.2f}）",
                    )
                )

        if self.enable_professional_penalty and ctx.internship_count <= 0:
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
            res["_adjustment_trace"] = dict(arbitrator.trace)
        if arbitrator.steps:
            res["_adjustment_steps"] = list(arbitrator.steps)

        # 语言要求惩罚发生在调整链之前（result_processor 直接乘概率）。
        # 在此补记一条 step/trace，保证审计链连续且可消融开关可观察。
        req_step = self._language_requirement_step(result, current_prob)
        if req_step is not None:
            steps = res.get("_adjustment_steps")
            if steps is None:
                steps = []
                res["_adjustment_steps"] = steps
            steps.insert(0, req_step)
            trace = res.get("_adjustment_trace")
            if trace is None:
                trace = {"base": current_prob}
                res["_adjustment_trace"] = trace
            trace["penalty_Language Requirement"] = req_step["delta"]

        normalized_prob = NormalizationLayer.apply(adjusted_prob)
        res["probability"] = normalized_prob
        trace = res.get("_adjustment_trace")
        if trace is not None:
            trace["final"] = normalized_prob
        return res

    @staticmethod
    def _language_requirement_step(
        result: dict[str, Any], current_prob: float
    ) -> dict[str, Any] | None:
        """链外语言要求惩罚的 step 描述（result 需携带 result_processor 的标记）。"""
        multiplier = result.get("_language_requirement_multiplier")
        if multiplier is None or "_probability_before_lang_req" not in result:
            return None
        before = clip_probability_coerce(result.get("_probability_before_lang_req"))
        after = current_prob
        delta = after - before
        if abs(delta) < 1e-12:
            return None
        return {
            "name": "Language Requirement Penalty",
            "before": round(before, 6),
            "after": round(after, 6),
            "delta": round(delta, 6),
            "type": "penalty",
            "description": "单校语言要求未达标（链外乘数）",
        }

    def adjust_batch(
        self,
        results: list[dict[str, Any]],
        ctx: AdjustmentContext,
        progress_reporter: Any | None = None,
        batch_tag: str = "",
        precomputed_quality: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not results:
            return results

        if ctx.school_stats is None and ctx.cases_df is not None:
            ctx.school_stats = compute_school_stats(ctx.cases_df)

        if ctx.baseline_admit_lookup is None and ctx.cases_df is not None:
            from src.adjustment.admission_cache import get_baseline_admit_lookup

            ctx.baseline_admit_lookup = get_baseline_admit_lookup(ctx.cases_df)

        original_probs = [clip_probability_coerce(r.get("probability")) for r in results]
        arbitrator = AdjustmentArbitrator()

        if self.probability_adjuster and ctx.gpa is not None and ctx.language_score is not None:
            penalties = self.probability_adjuster.get_penalties(ctx.gpa, ctx.language_score)
            logger.info(
                "调整管道: batch=%s count=%d | gpa=%.2f lang=%.2f | gpa_penalty=%.2f lang_penalty=%.2f",
                batch_tag,
                len(results),
                ctx.gpa,
                ctx.language_score,
                penalties.get("gpa", 0),
                penalties.get("language", 0),
            )
            if self.enable_gpa_penalty and penalties.get("gpa", 0) > 0:
                adj = self.probability_adjuster
                z = (adj.gpa_mean - ctx.gpa) / max(adj.gpa_std, 1e-6)
                desc = (
                    f"GPA {ctx.gpa:.2f} vs 全样本均值 {adj.gpa_mean:.2f}±{adj.gpa_std:.2f}"
                    f" | z={z:.2f}"
                )
                arbitrator.add_factor(
                    AdjustmentFactor(
                        name="GPA Penalty",
                        value=penalties["gpa"],
                        factor_type=AdjustmentFactorType.PENALTY,
                        description=desc,
                    ),
                    is_static=True,
                )
            if self.enable_language_penalty and penalties.get("language", 0) > 0:
                adj = self.probability_adjuster
                z = (adj.language_pass_line - ctx.language_score) / max(adj.language_std, 1e-6)
                desc = (
                    f"语言 {ctx.language_score:.2f} vs pass-line {adj.language_pass_line:.2f}"
                    f" | z={z:.2f}"
                )
                arbitrator.add_factor(
                    AdjustmentFactor(
                        name="Language Penalty",
                        value=penalties["language"],
                        factor_type=AdjustmentFactorType.PENALTY,
                        description=desc,
                    ),
                    is_static=True,
                )

        adjusted_results = [self.adjust_single(r, ctx, arbitrator) for r in results]

        attach_trace_extras(self, adjusted_results, ctx, original_probs)

        n_shrunk = 0
        if ctx.school_stats:
            global_admit_rate = BAYESIAN_SHRINKAGE_GLOBAL_PRIOR
            k = BAYESIAN_SHRINKAGE_PRIOR_STRENGTH
            for r in adjusted_results:
                n = int(r.get("_baseline_sample_count", 0) or 0)
                if n >= k:
                    continue  # sufficient data, trust the model
                univ = str(r.get("university", ""))
                school = ctx.school_stats.get(univ)
                prior = school["admit_rate"] if school else global_admit_rate
                model_prob = clip_probability_coerce(r.get("probability"))
                weight_model = n / (n + k)
                shrunk = weight_model * model_prob + (1.0 - weight_model) * prior
                r["probability"] = clip_probability(shrunk)
                n_shrunk += 1
                trace = dict(r.get("_adjustment_trace", {}))
                trace["bayesian_shrinkage"] = shrunk - model_prob
                r["_adjustment_trace"] = trace
            if n_shrunk > 0:
                logger.debug(
                    "Bayesian 收缩 | batch=%s n_shrunk=%d/%d k=%d",
                    batch_tag,
                    n_shrunk,
                    len(adjusted_results),
                    k,
                )

        if self.enable_text_boost and self.text_boost_provider and ctx.experience_details:
            if precomputed_quality is not None:
                quality_tags = precomputed_quality.get("quality_tags", {})
                llm_verified = precomputed_quality.get("llm_verified", {})
            else:
                quality_tags = self.text_boost_provider.get_quality_tags(ctx.experience_details)
                llm_verified: dict[str, dict[str, object]] = {}
                if quality_tags:
                    if self.quality_verifier is not None:
                        field_cn = {
                            "research_details": "科研",
                            "award_details": "奖项",
                            "internship_details": "实习",
                            "paper_details": "论文",
                        }
                        try:
                            verify_input: dict[str, dict[str, object]] = {}
                            for fk, fcn in field_cn.items():
                                text = str(ctx.experience_details.get(fk, "") or "").strip()
                                if text:
                                    verify_input[fk] = {
                                        "label": fcn,
                                        "text": text,
                                        "signal_hits": list(quality_tags.get(fk, [])),
                                    }
                            if verify_input:
                                llm_verified = self.quality_verifier(verify_input)
                                if llm_verified:
                                    logger.info(
                                        "LLM 含金量校验完成 | batch=%s fields=%d",
                                        batch_tag,
                                        len(llm_verified),
                                    )
                        except Exception:
                            logger.error("LLM 含金量校验失败 | batch=%s", batch_tag, exc_info=True)
                            llm_verified = {}

                    tag_parts = [
                        f"{field_cn.get(k, k)}: {', '.join(v)}"
                        for k, v in quality_tags.items()
                        if v
                    ]
                    if tag_parts:
                        logger.info(
                            "含金量标签已提取 | batch=%s count=%d | %s",
                            batch_tag,
                            len(adjusted_results),
                            "; ".join(tag_parts),
                        )

            if quality_tags:
                for r in adjusted_results:
                    trace = dict(r.get("_adjustment_trace", {}))
                    trace["quality_signals"] = {
                        "raw_tags": quality_tags,
                        "llm_verified": llm_verified,
                    }
                    r["_adjustment_trace"] = trace

        return adjusted_results

    def _adjust_cross_major_by_evidence(
        self,
        result: dict[str, Any],
        base_penalty: float,
        ctx: AdjustmentContext,
    ) -> float:
        if ctx.cases_df is None:
            return base_penalty

        if ctx.cross_major_stats is None:
            ctx.cross_major_stats = get_cross_major_admission_stats(
                ctx.cases_df, ctx.background_major or ""
            )

        key = (result.get("university"), result.get("major"))
        stats = ctx.cross_major_stats.get(key)
        if not stats or stats["n_total"] == 0:
            return base_penalty

        prior = CROSS_MAJOR_EVIDENCE_PRIOR_STRENGTH
        baseline_shrunk = (stats["admitted_total"] + 1) / (stats["n_total"] + 2)

        n_cross = stats["n_cross"]
        admitted_cross = stats["admitted_cross"]

        if n_cross == 0:
            return base_penalty

        cross_shrunk = (admitted_cross + prior * baseline_shrunk) / (n_cross + prior)

        if cross_shrunk >= baseline_shrunk * CROSS_MAJOR_EVIDENCE_RATIO_THRESHOLD:  # 0.85
            evidence_mult = 0.2
        else:
            ratio = cross_shrunk / max(baseline_shrunk, 0.01)
            evidence_mult = 1.0 - 0.8 * (ratio / CROSS_MAJOR_EVIDENCE_RATIO_THRESHOLD)
            evidence_mult = clip_scalar(evidence_mult, 0.2, 1.0)

        confidence = min(1.0, n_cross / CROSS_MAJOR_EVIDENCE_MIN_CASES)
        final_mult = confidence * evidence_mult + (1.0 - confidence) * 1.0

        adjusted = base_penalty * final_mult
        logger.debug(
            "跨专业证据调整 | target=%s@%s n_cross=%d baseline_rate=%.3f cross_rate=%.3f "
            "base_penalty=%.4f → adjusted=%.4f (evidence=%.2f confidence=%.2f)",
            result.get("major"),
            result.get("university"),
            n_cross,
            baseline_shrunk,
            cross_shrunk,
            base_penalty,
            adjusted,
            evidence_mult,
            confidence,
        )
        return adjusted
