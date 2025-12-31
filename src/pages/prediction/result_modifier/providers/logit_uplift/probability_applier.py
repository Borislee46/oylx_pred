from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.pages.prediction.result_modifier.config import (
    PROBABILITY_BOOST_MAX,
    PROBABILITY_BOOST_MIN,
    PROBABILITY_SCALE_CENTER,
    PROBABILITY_SCALE_FACTOR,
    QUALITY_SCORE_MAX_WEIGHT,
    QUALITY_SCORE_MEAN_WEIGHT,
)

if TYPE_CHECKING:
    from src.pages.prediction.result_modifier.providers.logit_uplift.text_processor import (
        TextProcessor,
    )


class ProbabilityApplier:
    """
    概率加成计算器v2.6 - 将抽象加成量转换为概率提升

    核心功能：
    1. Logit空间平滑转换：通过sigmoid函数确保概率值始终在(0,1)范围内
    2. 自适应封顶机制：基于文本质量和基础概率动态限制最大加成幅度

    转换流程：
    基础概率 P → Logit空间 L = log(P/(1-P)) → 加成 L' = L + Δ·smoothing → 新概率 P' = sigmoid(L')

    封顶逻辑：
    最大提升倍数 = 1 + MaxBoost × QualityFactor × ScaleFactor
    - QualityFactor: 文本质量评分（0-1），质量越高加成上限越高
    - ScaleFactor: 概率缩放因子，中段概率(≈0.5)加成空间最大，两端趋近于0

    设计约束：
    - 绝对安全：输出概率永不越界(0,1)
    - 平滑响应：sigmoid函数确保变化连续
    - 合理边界：防止低质量文本获得过高加成
    """

    def __init__(
        self,
        text_processor: TextProcessor,
        max_total_boost: float,
        smoothing: float,
        cap_min_factor: float,
        cap_quality_gamma: float,
    ) -> None:
        """
        Args:
            text_processor: 文本处理器，用于获取维度信息。
            max_total_boost: 总概率允许的最大提升比例（如 0.05）。
            smoothing: 平滑因子，控制 Logit 增量的生效强度。
            cap_min_factor: 即使质量很差，也保留的最小封顶比例（防止完全不加成）。
            cap_quality_gamma: 指数因子，调节质量对封顶上限的贡献斜率。
        """
        self._text_processor = text_processor
        self._max_total_boost = max_total_boost
        self._smoothing = smoothing
        self._cap_min_factor = cap_min_factor
        self._cap_quality_gamma = cap_quality_gamma

    def apply_probability_boost(
        self,
        probabilities: list[float],
        delta_logit: float,
        sims: dict[str, float],
    ) -> list[float]:
        """
        应用概率加成。
        """
        probs = np.array(probabilities, dtype=np.float64)

        # 计算质量因子 Q
        # Q = MaxWeight * MaxSimilarity + MeanWeight * MeanSimilarity
        s_values = np.array([sims.get(k, 0.0) for k in self._text_processor.text_keys])
        if s_values.size > 0:
            q_raw = QUALITY_SCORE_MAX_WEIGHT * np.max(
                s_values
            ) + QUALITY_SCORE_MEAN_WEIGHT * np.mean(s_values)
            # 应用伽马修正，增强高分端的区分度
            q_adj = q_raw ** max(1.0, self._cap_quality_gamma)
            # 最终封顶系数限制在 [min_factor, 1.0]
            cap_factor = min(1.0, max(self._cap_min_factor, q_adj))
        else:
            cap_factor = self._cap_min_factor

        # 应用平滑后的 Logit 增量
        effective_delta = delta_logit * self._smoothing

        # 过滤掉极端概率，只对有效区间内的概率进行处理
        mask = (probs >= PROBABILITY_BOOST_MIN) & (probs <= PROBABILITY_BOOST_MAX)
        if not np.any(mask):
            return probabilities

        updated = probs.copy()
        p_masked = probs[mask]

        # 概率 -> Logit -> +Delta -> 概率 (Sigmoid Inverse)
        logit_p = np.log(p_masked / (1.0 - p_masked))
        new_p = 1.0 / (1.0 + np.exp(-(logit_p + effective_delta)))

        # 计算自适应上限
        # Scale 逻辑：abs(p - 0.5) 越大（越接近 0 或 1），scale 越小，加成上限越紧
        scale = 1.0 - PROBABILITY_SCALE_FACTOR * np.abs(p_masked - PROBABILITY_SCALE_CENTER)
        cap = p_masked * (1.0 + self._max_total_boost * cap_factor * scale)

        # 取三者最小值：计算出的概率、上限、以及绝对上限 1.0
        updated[mask] = np.minimum(np.minimum(new_p, cap), 1.0)

        return updated.tolist()
