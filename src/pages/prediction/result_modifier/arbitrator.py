from typing import List
from src.pages.prediction.result_modifier.types import AdjustmentFactor, AdjustmentFactorType
from src.pages.prediction.result_modifier.config import (
    MAX_TOTAL_PENALTY_RATIO,
    MAX_TOTAL_BOOST_RATIO,
    PENALTY_DECAY_FACTOR,
    BOOST_DECAY_FACTOR,
    ARBITRATION_MIN_PROBABILITY,
)
from src.pages.prediction.result_modifier.utils import clip_probability


class AdjustmentArbitrator:
    def __init__(self):
        self.factors: List[AdjustmentFactor] = []

    def add_factor(self, factor: AdjustmentFactor):
        self.factors.append(factor)

    def arbitrate(self, base_probability: float) -> float:
        if not self.factors:
            return base_probability

        penalties = sorted(
            [f for f in self.factors if f.factor_type == AdjustmentFactorType.PENALTY],
            key=lambda x: x.value,
            reverse=True,
        )
        boosts = sorted(
            [f for f in self.factors if f.factor_type == AdjustmentFactorType.BOOST],
            key=lambda x: x.value,
            reverse=True,
        )

        total_penalty_ratio = 0.0
        p_decay = 1.0
        for p in penalties:
            total_penalty_ratio += p.value * p_decay * p.weight
            p_decay *= PENALTY_DECAY_FACTOR

        total_penalty_ratio = min(total_penalty_ratio, MAX_TOTAL_PENALTY_RATIO)

        total_boost_ratio = 0.0
        b_decay = 1.0
        for b in boosts:
            total_boost_ratio += b.value * b_decay * b.weight
            b_decay *= BOOST_DECAY_FACTOR

        total_boost_ratio = min(total_boost_ratio, MAX_TOTAL_BOOST_RATIO)

        final_prob = base_probability * (1 - total_penalty_ratio) * (1 + total_boost_ratio)

        return final_prob


class NormalizationLayer:
    @staticmethod
    def apply(probability: float) -> float:
        prob = clip_probability(probability)
        if prob > 0:
            prob = max(prob, ARBITRATION_MIN_PROBABILITY)
        return prob

