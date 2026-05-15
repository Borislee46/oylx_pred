"""
概率调整管道 — XGBoost 后处理的 5 层调整链。

模型输出的校准概率基于"历史类似案例的录取率"，但存在信息不对称：
  - 模型不知道个体 GPA 是否显著低于历史均值
  - 模型不知道跨专业的录取难度差异
  - 模型不知道学部边界的硬约束
  - 模型不知道职业学位对实习的强依赖

每层解决一个模型盲区，层间通过 AdjustmentArbitrator 管理叠加效应。

调整链顺序（不可变，每层依赖前一层的输出）：
  1. GPA/语言惩罚 (ProbabilityAdjuster, 每 batch 的 static 层)
  2. 跨专业惩罚 (Cross Major, per-result, similarity < 0.89)
  3. 跨学部惩罚 (Faculty, per-result, 硬编码学部规则)
  4. 职业学位惩罚 (Professional, per-result, 无实习 + MBA 类)
  5. 文本提升 (TF-IDF, batch 统一, 经历文本质量信号)

仲裁器衰减 (AdjustmentArbitrator)：
  多层惩罚叠加时，每层权重按 0.85 衰减（DEC-007）。
  e.g. 三层惩罚：×0.7 × ×0.5 × ×0.3
  → 仲裁后：×0.7 × (×0.5)^0.85 × (×0.3)^(0.85^2) ≈ ×0.7 × ×0.55 × ×0.36
  防止多惩罚叠加后概率被压到接近 0。

已知问题（来自 CLAUDE.md）：
  - 5 层惩罚联合效应从未被系统设计过，是一层一层加上去的
  - ECE=0.1155 > 0.10，严重失校准
  - C9 学生被低估 18pp，双非只被低估 6pp——惩罚对强者的伤害更大
"""

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
    MIN_SIMILARITY_THRESHOLD,
    PROFESSIONAL_MAJORS_LOWER,
    PROFESSIONAL_REDUCTION_FACTOR,
    PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR,
)
from src.pages.prediction.result_modifier.counterfactual import attach_trace_extras
from src.pages.prediction.result_modifier.faculty_filters import (
    get_cross_faculty_penalty_factor,
    is_faculty_out_of_scope,
)
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


# ── 调整上下文 ────────────────────────────────────────────
# 打包调整链所需的全部信息（20+ 字段）。
# 设计为可变 dataclass：pipeline 可以在处理过程中更新 cross_major_stats（延迟计算 + 缓存）。
@dataclass
class AdjustmentContext:
    gpa: float | None = None                              # 归一化 GPA
    language_score: float | None = None                   # 归一化语言成绩
    background_university: str | None = None              # 本科院校名
    background_major: str | None = None                   # 本科专业名（映射后）
    background_faculty: str | None = None                 # 本科专业所属学部
    internship_count: int = 0                             # 实习经历数量
    user_specified_majors: list[str] = field(default_factory=list)  # 用户指定的目标专业
    experience_details: dict[str, str] = field(default_factory=dict)  # 四段经历详细文本
    cases_df: pd.DataFrame | None = None                  # 历史案例（用于统计量计算）
    admitted_combinations: set[tuple[str, str]] = field(default_factory=set)  # 已知录取组合
    cross_major_stats: dict[tuple[str, str], dict] | None = None  # 跨专业录取统计（延迟计算）
    is_new_major_cache: dict[tuple[str, str], bool] = field(default_factory=dict)  # 新专业缓存


# ── 概率调整管道 ──────────────────────────────────────────
class ProbabilityAdjustmentPipeline:
    """5 层后处理链的编排器。

    使用方式：
        pipeline = ProbabilityAdjustmentPipeline(adjuster, text_boost)
        results = pipeline.adjust_batch(raw_results, ctx)

    adjust_batch 内部流程：
    1. 计算 batch 级 static 调整（GPA/语言惩罚，对该用户所有结果相同）
    2. 对每条结果调用 adjust_single（per-result 调整：跨专业/跨学部/职业学位）
    3. 应用文本提升（batch 级，TF-IDF 模型对所有结果统一加分）
    """

    def __init__(
        self,
        probability_adjuster: ProbabilityAdjuster | None = None,
        text_boost_provider: TextBoostProvider | None = None,
        enable_cross_major_penalty: bool = True,
    ):
        self.probability_adjuster = probability_adjuster  # Layer 1: GPA/语言惩罚计算器
        self.text_boost_provider = text_boost_provider    # Layer 5: TF-IDF 文本提升
        self.enable_cross_major_penalty = enable_cross_major_penalty  # 是否启用 Layer 2

    def adjust_single(
        self,
        result: dict[str, Any],
        ctx: AdjustmentContext,
        arbitrator: AdjustmentArbitrator | None = None,
    ) -> dict[str, Any]:
        """对单条预测结果应用 per-result 调整层（Layer 2-4）。

        调整顺序（按代码序，仲裁器管理叠加衰减）：
        1. Layer 2: 跨专业惩罚 — 相似度 < 0.89 触发，用 evidence 调整力度
        2. Layer 3: 跨学部惩罚 — 学部不一致触发，三级 severity (0.70/0.50/0.30)
        3. Layer 4: 职业学位惩罚 — MBA 类 + 无实习触发 (0.70/0.50)

        Arbitrator 复用：
          同一个 batch 内所有结果共享同一个 arbitrator 实例。
          static 层（GPA/语言惩罚）在 batch 开始时设置一次，
          每条结果调用 reset(keep_static=True) 只清除 per-result 层。
        """
        current_prob = get_probability(result)

        if arbitrator is None:
            arbitrator = AdjustmentArbitrator()
        else:
            arbitrator.reset(keep_static=True)  # 保留 GPA/语言 static 层

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
        #   理学院 → 法学院 ✗（超范围，触发惩罚）
        #
        # 三级 severity (B scheme)：
        #   轻度 ×0.70 — 知识体系有桥接（如 计算机→设计 HCI，理→社科 量化社科）
        #   中度 ×0.50 — 部分交集需补修（如 理→法 IP law，文→计算机 DH）
        #   重度 ×0.30 — 根本领域切换（如 理→文，法→医，默认值）
        #
        # 为什么硬编码学部规则而不是用数据驱动？
        #   跨学部录取案例极少（<1%），数据稀疏无法可靠学习。
        #   这种情况下 domain knowledge 优于 data-driven。
        # ─────────────────────────────────────────────────────────────────────
        if ctx.background_faculty:
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
        """对一批预测结果应用完整的 5 层调整链。

        这是 pipeline.py 调用的入口。流程：
        1. 计算 static 层（GPA/语言惩罚）— 对该用户所有结果相同
        2. 逐条调用 adjust_single（per-result 层：跨专业/跨学部/职业学位）
        3. 附加 trace extras（_adjustment_trace / _adjustment_steps）
        4. 如果文本提升 provider 可用 → 应用 TF-IDF 文本 boost
        """
        if not results:
            return results

        original_probs = [float(r.get("probability", 0.0) or 0.0) for r in results]
        arbitrator = AdjustmentArbitrator()

        # Static 层（每 batch 一次，所有结果共享）：
        # GPA 惩罚 + 语言惩罚通过 arbitrator.add_factor(is_static=True) 注册
        if self.probability_adjuster and ctx.gpa is not None and ctx.language_score is not None:
            penalties = self.probability_adjuster.get_penalties(ctx.gpa, ctx.language_score)
            logger.info(
                "调整管道: batch=%s count=%d | gpa=%.2f lang=%.2f | gpa_penalty=%.2f lang_penalty=%.2f",
                batch_tag, len(results), ctx.gpa, ctx.language_score,
                penalties.get("gpa", 0), penalties.get("language", 0),
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
                        name="GPA Penalty", value=penalties["gpa"],
                        factor_type=AdjustmentFactorType.PENALTY, description=desc,
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
                        name="Language Penalty", value=penalties["language"],
                        factor_type=AdjustmentFactorType.PENALTY, description=desc,
                    ),
                    is_static=True,
                )

        # Per-result 调整（每条结果独立计算 cross-major / faculty / professional）
        adjusted_results = [self.adjust_single(r, ctx, arbitrator) for r in results]

        # 附加 trace（counterfactual.py 中的反事实分析元数据）
        attach_trace_extras(self, adjusted_results, ctx, original_probs)

        # 文本提升（batch 级，TF-IDF 对所有结果统一加分）
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
                    batch_tag, len(adjusted_results), items,
                )

        return adjusted_results

    def _apply_text_boost(
        self,
        results: list[dict[str, Any]],
        experience_details: dict[str, str],
    ) -> list[dict[str, Any]]:
        """应用 TF-IDF 文本提升（Layer 5）。

        text_boost_provider.apply 对概率列表统一计算 boost 值。
        提升量 = 经历文本的 TF-IDF 信号强度（0~15%）。
        只有 delta > 1e-6 时才写入结果（避免无意义更新）。

        _adjustment_trace 和 _adjustment_steps 记录 boost 前后的概率变化，
        用于 UI 的"调整链可见性"（用户可看到每条结果被哪些因素调整了多少）。
        """
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
        """用跨专业录取率的实证数据缩放跨专业惩罚力度。

        使用 shrinkage (empirical Bayes) 估计每个目标 (院校, 专业) 的
        跨专业录取率相对于整体录取率的比值。

        核心思想：
        - 如果历史数据显示该跨专业组合的实际录取率不低 → 降低惩罚
        - 如果历史数据显示录取率确实低 → 保持完整惩罚
        - 样本量小时向 baseline 收缩（prior_strength=5），保守估计

        Returns:
            adjusted penalty in [0.2 * base_penalty, base_penalty]

        详细逻辑：
        1. baseline_shrunk = (总录取数 + 1) / (总申请数 + 2)  ← Laplace 平滑
        2. cross_shrunk = (跨专业录取数 + prior × baseline) / (跨专业数 + prior)
        3. 如果 cross_shrunk ≥ baseline × 0.85 → evidence_mult = 0.2（大幅减罚）
        4. 否则 → 按比值线性缩放 evidence_mult
        5. confidence = min(1.0, n_cross / 5)  ← 样本量信心
        6. final_mult = confidence × evidence + (1-confidence) × 1.0
           （样本少时向 1.0 收缩，保持保守）
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

        prior = CROSS_MAJOR_EVIDENCE_PRIOR_STRENGTH  # 5
        baseline_shrunk = (stats["admitted_total"] + 1) / (stats["n_total"] + 2)

        n_cross = stats["n_cross"]
        admitted_cross = stats["admitted_cross"]

        if n_cross == 0:
            return base_penalty

        cross_shrunk = (admitted_cross + prior * baseline_shrunk) / (n_cross + prior)

        if cross_shrunk >= baseline_shrunk * CROSS_MAJOR_EVIDENCE_RATIO_THRESHOLD:  # 0.85
            evidence_mult = 0.2  # 证据表明无障碍，惩罚降至 20%
        else:
            ratio = cross_shrunk / max(baseline_shrunk, 0.01)
            evidence_mult = 1.0 - 0.8 * (ratio / CROSS_MAJOR_EVIDENCE_RATIO_THRESHOLD)
            evidence_mult = max(0.2, min(1.0, evidence_mult))

        confidence = min(1.0, n_cross / CROSS_MAJOR_EVIDENCE_MIN_CASES)  # n_cross / 5
        final_mult = confidence * evidence_mult + (1.0 - confidence) * 1.0

        return base_penalty * final_mult
