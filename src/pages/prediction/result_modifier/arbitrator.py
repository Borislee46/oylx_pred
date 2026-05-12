# =============================================================================
# 调整仲裁器 (Adjustment Arbitrator)
# ─────────────────────────────────────────────────────────────────────────────
# 当一个预测结果触发多个调整因子时，不能简单叠加。
# 需要仲裁器来处理因子间的交互：排序、衰减、上限。
#
# 核心问题：多个惩罚不能累加过度
#   假设 GpaPenalty ×0.85 + CrossMajorPenalty ×0.7 + FacultyPenalty ×0.3
#   如果直接连乘: prob × 0.85 × 0.7 × 0.3 = prob × 0.1785 (过重！)
#   仲裁器用衰减机制让后续因子影响递减，保证总惩罚不超过 MAX_TOTAL_PENALTY_RATIO (70%)
#
# 设计决策：惩罚-先于-提升 的顺序
#   1. 先降分（惩罚）再提分（提升）→ 防止"先提再降"把提升也打折了
#   2. 实际效果：先应用 GPA/语言/跨专业/学部/职业学位惩罚 →
#      得到一个"保守估计"的概率 → 再应用文本提升 → 最终概率
#   3. 这个顺序保守且可解释：基础实力决定底线，文本质量提供增量
# =============================================================================

from src.pages.prediction.result_modifier.config import (
    ARBITRATION_MIN_PROBABILITY,
    BOOST_DECAY_FACTOR,
    MAX_TOTAL_BOOST_RATIO,
    MAX_TOTAL_PENALTY_RATIO,
    PENALTY_DECAY_FACTOR,
)
from src.pages.prediction.result_modifier.types import AdjustmentFactor, AdjustmentFactorType
from src.pages.prediction.result_modifier.utils import clip_probability


class AdjustmentArbitrator:
    def __init__(self, include_trace: bool = True):
        self.factors: list[AdjustmentFactor] = []
        self.static_factors: list[AdjustmentFactor] = []
        self.trace: dict[str, float] = {}
        self.steps: list[dict] = []
        self.include_trace = include_trace

    def reset(self, keep_static: bool = False):
        self.factors.clear()
        if not keep_static:
            self.static_factors.clear()
        self.trace.clear()
        self.steps.clear()

    def add_factor(self, factor: AdjustmentFactor, is_static: bool = False):
        if is_static:
            self.static_factors.append(factor)
        else:
            self.factors.append(factor)

    # ─────────────────────────────────────────────────────────────────────────
    # 仲裁核心算法：衰减叠加 + 上限保护
    # ─────────────────────────────────────────────────────────────────────────
    # 流程：
    #   1. 惩罚因子按 severity 降序排列（最严重的先影响）
    #   2. 每个后续惩罚 × 衰减因子 (PENALTY_DECAY_FACTOR = 0.85)
    #      第1个: ×1.0, 第2个: ×0.85, 第3个: ×0.72, ...
    #   3. 总惩罚上限 70% (MAX_TOTAL_PENALTY_RATIO) — 无论如何保留 30% 底线
    #   4. 提升因子同理，衰减 + 上限 30%
    #
    # 为什么是 0.85 衰减（不是 0.5 或 0.9）？
    #   0.5: 太快衰减，第2个惩罚只剩一半效果 → 多层惩罚失去意义
    #   0.9: 太慢衰减，3个惩罚接近全额叠加 → 回到过重问题
    #   0.85: 2个惩罚 ≈ 0.85 效果，3个 ≈ 0.72 效果
    #         够显著但不会灾难性叠加
    #
    # 为什么衰减在各因子间应用而不是总惩罚上？
    #   如果直接给总惩罚打折 → 单个惩罚也被打折了，不合理
    #   衰减用于降低"第N个惩罚的边际影响" → 第1个全额，后续递减
    #   这符合直觉：主要问题（如 GPA 低）应该完整反映，
    #   次要问题（跨专业）在主要问题已反映后应该弱化
    # ─────────────────────────────────────────────────────────────────────────
    def arbitrate(self, base_probability: float) -> float:
        all_factors = self.factors + self.static_factors
        if not all_factors:
            return base_probability

        penalties = []
        boosts = []
        for f in all_factors:
            if f.factor_type == AdjustmentFactorType.PENALTY:
                penalties.append(f)
            elif f.factor_type == AdjustmentFactorType.BOOST:
                boosts.append(f)

        if len(penalties) > 1:
            penalties.sort(key=lambda x: x.value, reverse=True)
        if len(boosts) > 1:
            boosts.sort(key=lambda x: x.value, reverse=True)

        if self.include_trace:
            self.trace = {"base": base_probability}
            self.steps = []

        total_penalty_ratio = 0.0
        p_decay = 1.0
        for p in penalties:
            contribution = p.value * p_decay * p.weight
            before = base_probability * (1 - min(total_penalty_ratio, MAX_TOTAL_PENALTY_RATIO))
            total_penalty_ratio += contribution
            effective_cumulative = min(total_penalty_ratio, MAX_TOTAL_PENALTY_RATIO)
            after = base_probability * (1 - effective_cumulative)
            if self.include_trace:
                self.trace[f"penalty_{p.name}"] = -base_probability * contribution
                self.steps.append(
                    {
                        "name": p.name,
                        "before": round(before, 6),
                        "after": round(after, 6),
                        "delta": round(after - before, 6),
                        "type": "penalty",
                        "description": p.description,
                    }
                )
            p_decay *= PENALTY_DECAY_FACTOR

        total_penalty_ratio = min(total_penalty_ratio, MAX_TOTAL_PENALTY_RATIO)
        prob_after_penalty = base_probability * (1 - total_penalty_ratio)

        total_boost_ratio = 0.0
        b_decay = 1.0
        for b in boosts:
            contribution = b.value * b_decay * b.weight
            before = prob_after_penalty * (1 + min(total_boost_ratio, MAX_TOTAL_BOOST_RATIO))
            total_boost_ratio += contribution
            effective_cumulative = min(total_boost_ratio, MAX_TOTAL_BOOST_RATIO)
            after = prob_after_penalty * (1 + effective_cumulative)
            if self.include_trace:
                self.trace[f"boost_{b.name}"] = prob_after_penalty * contribution
                self.steps.append(
                    {
                        "name": b.name,
                        "before": round(before, 6),
                        "after": round(after, 6),
                        "delta": round(after - before, 6),
                        "type": "boost",
                        "description": b.description,
                    }
                )
            b_decay *= BOOST_DECAY_FACTOR

        total_boost_ratio = min(total_boost_ratio, MAX_TOTAL_BOOST_RATIO)
        final_prob = prob_after_penalty * (1 + total_boost_ratio)

        if self.include_trace:
            self.trace["final"] = final_prob
        return final_prob


# ─────────────────────────────────────────────────────────────────────────────
# 归一化层 (Normalization Layer)
# ─────────────────────────────────────────────────────────────────────────────
# 最终概率的约束：
#   - 裁剪到 [0, 1] 区间（经过多次惩罚/提升计算，可能浮点溢出）
#   - 非零概率有 floor = 0.005 (ARBITRATION_MIN_PROBABILITY)
#
# 为什么设 0.005 而非 0？
#   0 表示"绝对不可能"，但录取预测中几乎没有真正不可能的情况
#   （特例学生、特殊渠道、政策变化都可能改变结果）。
#   0.005 = "极不可能但不是0"，保留了微小的可能性但不给用户虚假希望。
#   这既是统计上的谨慎，也是产品体验 — 0% 会让用户觉得系统武断。
# ─────────────────────────────────────────────────────────────────────────────
class NormalizationLayer:
    @staticmethod
    def apply(probability: float) -> float:
        prob = clip_probability(probability)
        if prob > 0:
            prob = max(prob, ARBITRATION_MIN_PROBABILITY)
        return prob
