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
    ) -> list[float]:
        probs = np.array(probabilities, dtype=np.float64)

        s_values = np.array([sims.get(k, 0.0) for k in self._text_processor.text_keys])
        if s_values.size > 0:
            q_raw = QUALITY_SCORE_MAX_WEIGHT * np.max(
                s_values
            ) + QUALITY_SCORE_MEAN_WEIGHT * np.mean(s_values)
            q_adj = q_raw ** max(1.0, self._cap_quality_gamma)
            cap_factor = min(1.0, max(self._cap_min_factor, q_adj))
        else:
            cap_factor = self._cap_min_factor

        effective_delta = delta_logit * self._smoothing

        mask = (probs >= PROBABILITY_BOOST_MIN) & (probs <= PROBABILITY_BOOST_MAX)
        if not np.any(mask):
            return probabilities

        updated = probs.copy()
        p_masked = probs[mask]

        logit_p = np.log(p_masked / (1.0 - p_masked))
        new_p = 1.0 / (1.0 + np.exp(-(logit_p + effective_delta)))

        scale = 1.0 - PROBABILITY_SCALE_FACTOR * np.abs(p_masked - PROBABILITY_SCALE_CENTER)
        cap = p_masked * (1.0 + self._max_total_boost * cap_factor * scale)

        updated[mask] = np.minimum(np.minimum(new_p, cap), 1.0)

        return updated.tolist()
