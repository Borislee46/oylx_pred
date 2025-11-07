from __future__ import annotations

import numpy as np

from src.pages.prediction.result_modifier.config import (
    PROBABILITY_BOOST_MAX,
    PROBABILITY_BOOST_MIN,
    PROBABILITY_SCALE_CENTER,
    PROBABILITY_SCALE_FACTOR,
    QUALITY_SCORE_MAX_WEIGHT,
    QUALITY_SCORE_MEAN_WEIGHT,
    QUALITY_SCORE_THRESHOLD,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.text_processor import (
    TextProcessor,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.utils import (
    logit,
    safe_float,
    sigmoid,
)


class ProbabilityApplier:

    def __init__(
        self,
        text_processor: TextProcessor,
        max_total_boost: float,
        smoothing: float,
        cap_min_factor: float,
        cap_quality_gamma: float,
    ) -> None:
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
    ) -> tuple[list[float], list[float]]:
        s_values = [sims.get(k, 0.0) for k in self._text_processor.text_keys]
        q_raw = QUALITY_SCORE_MAX_WEIGHT * max(s_values) + QUALITY_SCORE_MEAN_WEIGHT * (
            sum(s_values) / len(s_values)
        )
        q_adj = q_raw ** max(1.0, self._cap_quality_gamma)
        cap_factor = min(1.0, max(self._cap_min_factor, q_adj))

        effective_delta = delta_logit * self._smoothing

        updated: list[float] = []
        boosts: list[float] = []

        for p in probabilities:
            p0 = safe_float(p, 0.0)
            if PROBABILITY_BOOST_MIN <= p0 <= PROBABILITY_BOOST_MAX:
                new_p = sigmoid(logit(p0) + effective_delta)
                scale = 1.0 - PROBABILITY_SCALE_FACTOR * abs(p0 - PROBABILITY_SCALE_CENTER)
                cap_boost = self._max_total_boost * cap_factor * scale
                cap = p0 * (1.0 + cap_boost)
                new_p = min(new_p, cap, 1.0)
                updated.append(new_p)
                boosts.append((new_p / p0) - 1.0)
            else:
                updated.append(p0)

        return updated, boosts

    def generate_summary(self, boosts: list[float], sims: dict[str, float]) -> str:
        if not boosts:
            return ""

        parts: list[str] = []
        name_map = {
            "research_details": "科研项目",
            "award_details": "获奖情况",
            "internship_details": "实习经历",
            "paper_details": "论文发表",
        }
        for k in self._text_processor.text_keys:
            s = sims.get(k, 0.0)
            if s > QUALITY_SCORE_THRESHOLD:
                parts.append(f"{name_map[k]}: {s:.2f}")

        avg_boost = float(np.mean(boosts))
        summary = f"+{avg_boost:.1%} ({', '.join(parts)})" if avg_boost > 0 else ""

        return summary

