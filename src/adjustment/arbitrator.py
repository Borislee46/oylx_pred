from src.adjustment.config import (
    ARBITRATION_MIN_PROBABILITY,
    BOOST_DECAY_FACTOR,
    MAX_TOTAL_BOOST_RATIO,
    MAX_TOTAL_PENALTY_RATIO,
    PENALTY_DECAY_FACTOR,
)
from src.adjustment.engine import AdjustmentFactor, AdjustmentFactorType
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability

logger = setup_logger("page3", "prediction")


class AdjustmentArbitrator:
    def __init__(self, include_trace: bool = True):
        self.factors: list[AdjustmentFactor] = []
        self.static_factors: list[AdjustmentFactor] = []
        self.trace: dict[str, float] = {}
        self.steps: list[dict] = []
        self.include_trace = include_trace

    def reset(self, keep_static: bool = False):
        n_factors = len(self.factors)
        if n_factors > 0:
            logger.debug(
                "仲裁器 reset | cleared=%d per_result_factors | keep_static=%s",
                n_factors,
                keep_static,
            )
        self.factors.clear()
        if not keep_static:
            n_static = len(self.static_factors)
            if n_static > 0:
                logger.debug("仲裁器 reset | cleared=%d static_factors", n_static)
            self.static_factors.clear()
        self.trace.clear()
        self.steps.clear()

    def add_factor(self, factor: AdjustmentFactor, is_static: bool = False):
        if is_static:
            self.static_factors.append(factor)
        else:
            self.factors.append(factor)
        logger.debug(
            "仲裁器 add_factor | name=%s value=%.4f type=%s static=%s desc=%s",
            factor.name,
            factor.value,
            factor.factor_type.value,
            is_static,
            factor.description,
        )

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

        logger.debug(
            "仲裁开始 | base=%.4f n_penalties=%d n_boosts=%d | penalties=%s boosts=%s",
            base_probability,
            len(penalties),
            len(boosts),
            [(p.name, round(p.value, 4)) for p in penalties],
            [(b.name, round(b.value, 4)) for b in boosts],
        )

        penalty_ceiling = MAX_TOTAL_PENALTY_RATIO

        if self.include_trace:
            self.trace = {"base": base_probability}
            self.steps = []

        total_penalty_ratio = 0.0
        p_decay = 1.0
        for p in penalties:
            contribution = p.value * p_decay * p.weight
            before = base_probability * (1 - min(total_penalty_ratio, penalty_ceiling))
            total_penalty_ratio += contribution
            effective_cumulative = min(total_penalty_ratio, penalty_ceiling)
            after = base_probability * (1 - effective_cumulative)
            if self.include_trace:
                # trace 必须与实际边际 delta 一致（累计达到 cap 后 delta=0，
                # 不能再用 -base*contribution，否则 trace 与 steps/final 对不上）。
                self.trace[f"penalty_{p.name}"] = after - before
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

        total_penalty_ratio = min(total_penalty_ratio, penalty_ceiling)
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
                self.trace[f"boost_{b.name}"] = after - before
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

        delta = final_prob - base_probability
        logger.debug(
            "仲裁完成 | base=%.4f → final=%.4f | delta=%+.4f | "
            "total_penalty=%.2f%% total_boost=%.2f%% | "
            "after_penalty=%.4f",
            base_probability,
            final_prob,
            delta,
            total_penalty_ratio * 100,
            total_boost_ratio * 100,
            prob_after_penalty,
        )

        if self.include_trace:
            self.trace["final"] = final_prob
        return final_prob


class NormalizationLayer:
    @staticmethod
    def apply(probability: float) -> float:
        prob = clip_probability(probability)
        if prob > 0 and prob < ARBITRATION_MIN_PROBABILITY:
            logger.debug(
                "归一化 floor | raw=%.6f → floored=%.6f",
                prob,
                ARBITRATION_MIN_PROBABILITY,
            )
            prob = ARBITRATION_MIN_PROBABILITY
        return prob
