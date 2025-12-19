from __future__ import annotations

import json
import math

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
        self._delta_cache: dict[
            str, tuple[float, tuple[tuple[str, float], ...], tuple[str, ...]]
        ] = {}
        self._cache_maxsize = 512

    def _compute_delta_logit(self, sig: str) -> tuple[float, dict[str, float], tuple[str, ...]]:
        weights_array = self._model_loader.weights_array
        text_keys = self._text_processor.text_keys
        count_keys = self._text_processor.count_keys

        num_text_features = len(text_keys)
        num_count_features = len(count_keys)

        try:
            details = json.loads(sig)
        except (json.JSONDecodeError, TypeError, ValueError):
            details = {}

        sims, reasons = self._similarity_computer.compute_similarities(details)

        s_values = [sims.get(k, 0.0) for k in text_keys]
        ssum = sum(s_values)
        smax = max(s_values) if s_values else 0.0

        if ssum < self._sim_gate_sum_min or smax < self._sim_gate_max_min:
            return 0.0, sims, reasons

        # 对于极小规模特征（4个），原生 math.log1p 和循环比 numpy 快得多
        log_counts = [math.log1p(safe_float(details.get(k, 0))) for k in count_keys]

        bias_idx = 0
        text_weights_start = bias_idx + 1
        text_weights_end = text_weights_start + num_text_features

        interact_weights_start = text_weights_end
        interact_weights_end = interact_weights_start + num_text_features

        if len(weights_array) < interact_weights_end:
            return 0.0, sims, reasons

        delta = weights_array[bias_idx]
        for i in range(num_text_features):
            delta += weights_array[text_weights_start + i] * s_values[i]

        if num_count_features == num_text_features:
            for i in range(num_text_features):
                delta += weights_array[interact_weights_start + i] * s_values[i] * log_counts[i]

        delta = max(0.0, float(delta))

        return delta, sims, reasons

    def _cached_delta_logit_internal(
        self, sig: str
    ) -> tuple[float, tuple[tuple[str, float], ...], tuple[str, ...]]:
        if sig in self._delta_cache:
            return self._delta_cache[sig]

        delta, sims, reasons = self._compute_delta_logit(sig)
        result = delta, tuple(sims.items()), reasons

        if len(self._delta_cache) >= self._cache_maxsize:
            self._delta_cache.clear()
        self._delta_cache[sig] = result
        return result

    def cached_delta_logit(self, sig: str) -> tuple[float, dict[str, float], tuple[str, ...]]:
        delta, sims_tuple, reasons = self._cached_delta_logit_internal(sig)
        return delta, dict(sims_tuple), reasons
