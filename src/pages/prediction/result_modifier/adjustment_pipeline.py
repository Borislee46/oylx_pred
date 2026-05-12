# =============================================================================
# 概率调整管道 (Probability Adjustment Pipeline)
# ─────────────────────────────────────────────────────────────────────────────
# 核心设计：多层后处理链，每层解决一个模型盲区。
# XGBoost + 校准输出的是 "历史数据中类似案例的录取率"，
# 但这个概率在以下场景会给出不合理的估计：
#
# [Layer 1] GPA/语言偏差惩罚 (probability_adjuster.py)
#   模型用 case 整体模式预测，个体极端低分可能被数据中的
#   高竞争 case 拉高 → 必须在后处理中修正。
#
# [Layer 2] 跨专业惩罚 (Cross Major Penalty) — 本文件
#   模型不知道"同校不同专业"的录取差异有多大。
#   跨专业相似度低 → 录取难度完全不同 → 惩罚系数最高 ×0.5。
#
# [Layer 3] 跨学部惩罚 (Cross Faculty Penalty) — faculty_filters.py
#   从理学院跨到文学院 → 几乎不可能 → 惩罚系数 ×0.3。
#
# [Layer 4] 职业学位惩罚 (Professional Degree Penalty) — 本文件
#   MBA/Business Admin 等重实习经验的专业，无实习 → 降权。
#
# [Layer 5] 文本背景提升 (TF-IDF Text Boost) — 本文件
#   背提文本质量高 → 轻微 boost（上限+15%）。
#
# 管道设计原则：
# 1. 每层有清晰的触发条件（不满足时无开销）
# 2. 多层叠加有衰减（Arbitrator 中实现，多重惩罚递减以防止过度惩罚）
# 3. 惩罚 → 提升 → 归一化 → clip，顺序固定
# =============================================================================

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.pages.prediction.config.ui_messages import (
    EXPERIENCE_BOOST_TEMPLATE,
    EXPERIENCE_ITEM_NAMES,
)
from src.pages.prediction.result_modifier.admission_cache import (
    get_cross_major_admission_stats,
)
from src.pages.prediction.result_modifier.arbitrator import (
    AdjustmentArbitrator,
    NormalizationLayer,
)
from src.pages.prediction.result_modifier.config import (
    CROSS_MAJOR_EVIDENCE_MIN_CASES,
    CROSS_MAJOR_EVIDENCE_PRIOR_STRENGTH,
    CROSS_MAJOR_EVIDENCE_RATIO_THRESHOLD,
    FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR,
    MIN_SIMILARITY_THRESHOLD,
    PROFESSIONAL_MAJORS_LOWER,
    PROFESSIONAL_REDUCTION_FACTOR,
    PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR,
)
from src.pages.prediction.result_modifier.counterfactual import attach_trace_extras
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
    cross_major_stats: dict[tuple[str, str], dict] | None = None
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

        # ─────────────────────────────────────────────────────────────────────
        # Layer 2: 跨专业惩罚 (Cross Major Penalty)
        # ─────────────────────────────────────────────────────────────────────
        # 触发条件：相似度 < 0.89 (MIN_SIMILARITY_THRESHOLD)
        # 基础惩罚：线性插值，similarity=0.8 → ×0.5, similarity=0.89 → ×1.0
        #
        # 证据调整：用 shrinkage (empirical Bayes) 估计该背景专业→目标专业的
        #   录取率相对于整体录取率的比值。同专业的基线录取率作为 cross-major
        #   估计的 prior center，prior_strength=5 控制收缩强度。
        #   - n_cross=0 时无证据，保持完整基础惩罚（保守）
        #   - cross_shrunk ≥ baseline × 0.85 时证据表明跨专业无障碍 → 惩罚降至 20%
        #   - cross_shrunk < baseline × 0.85 时证据确认障碍 → 惩罚按比值线性缩放
        # ─────────────────────────────────────────────────────────────────────
        if self.enable_cross_major_penalty and ctx.background_major:
            similarity = float(result.get("similarity", 1.0))
            if similarity < MIN_SIMILARITY_THRESHOLD:
                base_penalty = 1.0 - cross_major_penalty_factor(similarity)
                if base_penalty > 0:
                    p_factor = self._adjust_cross_major_by_evidence(result, base_penalty, ctx)
                    if p_factor > 0:
                        arbitrator.add_factor(
                            AdjustmentFactor(
                                name="Cross Major Penalty",
                                value=p_factor,
                                factor_type=AdjustmentFactorType.PENALTY,
                                description=f"背景相似度低 ({similarity:.2f})",
                            )
                        )

        # ─────────────────────────────────────────────────────────────────────
        # Layer 3: 跨学部惩罚 (Cross Faculty Penalty)
        # ─────────────────────────────────────────────────────────────────────
        # 学部跨度判断依据 CROSS_FACULTY_RULES (faculty_filters.py)：
        #   理学院 → 工程学院 ✓（允许，不触发惩罚）
        #   理学院 → 法学院 ✗（超范围，触发惩罚 ×0.3）
        #
        # 惩罚 = ×0.3 (FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR)
        # 比跨专业惩罚更重（0.3 vs 0.5），因为学部跨度是更根本的障碍：
        #   跨专业 = 同一知识体系下的分支切换
        #   跨学部 = 整个知识体系不同，需要补修大量先修课
        #
        # 为什么硬编码学部规则而不是用数据驱动？
        #   跨学部录取案例极少（<1%），数据稀疏无法可靠学习。
        #   这种情况下 domain knowledge 优于 data-driven。
        # ─────────────────────────────────────────────────────────────────────
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

        # ─────────────────────────────────────────────────────────────────────
        # Layer 4: 职业学位惩罚 (Professional Degree Penalty)
        # ─────────────────────────────────────────────────────────────────────
        # 触发条件：目标专业是职业导向学位（MBA、Business Administration 等）
        #          且学生没有实习经历 (internship_count <= 0)
        #
        # 惩罚力度：
        #   - 非用户指定: ×0.70 (PROFESSIONAL_REDUCTION_FACTOR)
        #   - 用户指定: ×0.50 (PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR)
        #
        # 为什么用户指定反而惩罚更重？
        #   用户主动选了 MBA 说明有意愿，但没有实习 → 申请竞争力更弱。
        #   系统推荐的非用户指定 MBA 可能只是相似度匹配结果，
        #   用户未必真想申，惩罚轻一些。
        #
        # 为什么单挑实习，而不是工作经验？
        #   留学申请中实习经历是 MBA/商科最直接的竞争力信号。
        #   科研+论文对理工科有价值但对 MBA 几乎无关。
        #   用 internship_count 而非所有经历计数 — 针对性惩罚。
        # ─────────────────────────────────────────────────────────────────────
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
            res["_adjustment_trace"] = dict(arbitrator.trace)
        if arbitrator.steps:
            res["_adjustment_steps"] = list(arbitrator.steps)

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

        original_probs = [float(r.get("probability", 0.0) or 0.0) for r in results]
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
            if penalties.get("gpa", 0) > 0:
                adj = self.probability_adjuster
                z = (adj.gpa_mean - ctx.gpa) / max(adj.gpa_std, 1e-6)
                desc = (
                    f"GPA {ctx.gpa:.2f} vs 录取者均值 {adj.gpa_mean:.2f}±{adj.gpa_std:.2f}"
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
            if penalties.get("language", 0) > 0:
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
                logger.info(
                    "文本提升已应用 | batch=%s count=%d items=%s",
                    batch_tag,
                    len(adjusted_results),
                    items,
                )

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
                    steps = list(res.get("_adjustment_steps", []))
                    steps.append(
                        {
                            "name": "NLP Text Boost",
                            "before": round(old_prob, 6),
                            "after": round(new_prob, 6),
                            "delta": round(new_prob - old_prob, 6),
                            "type": "boost",
                            "description": "文本背景提升",
                        }
                    )
                    res["_adjustment_steps"] = steps
                    results[i] = res

        return results

    def _adjust_cross_major_by_evidence(
        self,
        result: dict[str, Any],
        base_penalty: float,
        ctx: AdjustmentContext,
    ) -> float:
        """Scale cross-major penalty by empirical admission rate evidence.

        Uses shrinkage (empirical Bayes) to estimate cross-major vs baseline
        admission rates per target, accounting for small sample sizes.

        Returns adjusted penalty factor in [0.2 * base_penalty, base_penalty].
        """
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

        if cross_shrunk >= baseline_shrunk * CROSS_MAJOR_EVIDENCE_RATIO_THRESHOLD:
            evidence_mult = 0.2
        else:
            ratio = cross_shrunk / max(baseline_shrunk, 0.01)
            evidence_mult = 1.0 - 0.8 * (ratio / CROSS_MAJOR_EVIDENCE_RATIO_THRESHOLD)
            evidence_mult = max(0.2, min(1.0, evidence_mult))

        confidence = min(1.0, n_cross / CROSS_MAJOR_EVIDENCE_MIN_CASES)
        final_mult = confidence * evidence_mult + (1.0 - confidence) * 1.0

        return base_penalty * final_mult
