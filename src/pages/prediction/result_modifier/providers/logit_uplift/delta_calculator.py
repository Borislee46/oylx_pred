from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from src.pages.prediction.result_modifier.providers.logit_uplift.model_loader import (
    ModelLoader,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.similarity_computer import (
    SimilarityComputer,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.text_processor import (
    TextProcessor,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.utils import safe_float


class DeltaCalculator:
    def __init__(
        self,
        model_loader: ModelLoader,
        similarity_computer: SimilarityComputer,
        text_processor: TextProcessor,
        sim_gate_sum_min: float,
        sim_gate_max_min: float,
    ) -> None:
        self._model_loader = model_loader
        self._similarity_computer = similarity_computer
        self._text_processor = text_processor
        self._sim_gate_sum_min = sim_gate_sum_min
        self._sim_gate_max_min = sim_gate_max_min

    def _compute_delta_logit(self, sig: str) -> tuple[float, dict[str, float]]:
        weights_array = self._model_loader.weights_array
        text_keys = self._text_processor.text_keys
        count_keys = self._text_processor.count_keys

        try:
            details = json.loads(sig)
        except Exception:
            details = {}

        sims = self._similarity_computer.compute_similarities(details)

        s_values = [sims.get(k, 0.0) for k in text_keys]
        ssum = sum(s_values)
        smax = max(s_values) if s_values else 0.0

        if ssum < self._sim_gate_sum_min or smax < self._sim_gate_max_min:
            return 0.0, sims

        counts = np.array([safe_float(details.get(k, 0)) for k in count_keys], dtype=np.float64)
        log_counts = np.log1p(counts)

        s_arr = np.array(s_values, dtype=np.float64)

        delta = weights_array[0]
        delta += np.dot(weights_array[1:5], s_arr)
        delta += np.dot(weights_array[5:9], s_arr * log_counts)

        delta = max(0.0, float(delta))

        return delta, sims

    @lru_cache(maxsize=512)
    def cached_delta_logit(self, sig: str) -> tuple[float, dict[str, float]]:
        return self._compute_delta_logit(sig)
