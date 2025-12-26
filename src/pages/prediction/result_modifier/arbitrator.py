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
    def __init__(self):
        self.factors: list[AdjustmentFactor] = []
        self.trace: dict[str, float] = {}

    def add_factor(self, factor: AdjustmentFactor):
        self.factors.append(factor)

    def arbitrate(self, base_probability: float) -> float:
        if not self.factors:
            self.trace = {}
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

        self.trace = {"base": base_probability}

        total_penalty_ratio = 0.0
        p_decay = 1.0
        for p in penalties:
            factor_delta_ratio = p.value * p_decay * p.weight
            total_penalty_ratio += factor_delta_ratio
            p_decay *= PENALTY_DECAY_FACTOR

            self.trace[f"penalty_{p.name}"] = -base_probability * factor_delta_ratio

        total_penalty_ratio = min(total_penalty_ratio, MAX_TOTAL_PENALTY_RATIO)
        prob_after_penalty = base_probability * (1 - total_penalty_ratio)

        total_boost_ratio = 0.0
        b_decay = 1.0
        for b in boosts:
            factor_delta_ratio = b.value * b_decay * b.weight
            total_boost_ratio += factor_delta_ratio
            b_decay *= BOOST_DECAY_FACTOR

            self.trace[f"boost_{b.name}"] = prob_after_penalty * factor_delta_ratio

        total_boost_ratio = min(total_boost_ratio, MAX_TOTAL_BOOST_RATIO)
        final_prob = prob_after_penalty * (1 + total_boost_ratio)

        self.trace["final"] = final_prob
        return final_prob


class NormalizationLayer:
    @staticmethod
    def apply(probability: float) -> float:
        prob = clip_probability(probability)
        if prob > 0:
            prob = max(prob, ARBITRATION_MIN_PROBABILITY)
        return prob
