from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from src.pages.prediction.result_modifier.utils import has_any_experience
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class TextBoostProvider(ABC):
    @abstractmethod
    def apply(
        self, probabilities: list[float], experience_details: dict[str, str]
    ) -> list[float]: ...


class NullTextBoostProvider(TextBoostProvider):
    def apply(self, probabilities: list[float], experience_details: dict[str, str]) -> list[float]:
        return probabilities


class GatedTextBoostProvider(TextBoostProvider):
    def __init__(self, inner: TextBoostProvider):
        self._inner = inner

    def apply(self, probabilities: list[float], experience_details: dict[str, str]) -> list[float]:
        if not has_any_experience(experience_details):
            return probabilities
        return self._inner.apply(probabilities, experience_details)


def get_text_boost_provider(config: dict[str, Any] | None) -> TextBoostProvider:
    key = json.dumps(config or {}, ensure_ascii=False, sort_keys=True)
    return _get_text_boost_provider_cached(key)


@lru_cache(maxsize=16)
def _get_text_boost_provider_cached(config_key: str) -> TextBoostProvider:
    config = json.loads(config_key)
    from src.pages.prediction.result_modifier.providers.logit_uplift_provider import (
        LogitUpliftProvider,
    )

    model_paths = (config or {}).get("model_paths", {})
    vec_path = model_paths.get("tfidf_vectorizer")
    cen_path = model_paths.get("tfidf_centroids")
    w_path = model_paths.get("text_uplift_weights")

    max_total_boost = config.get("max_total_boost", 0.05)
    sim_gate_sum_min = config.get("sim_gate_sum_min")
    sim_gate_max_min = config.get("sim_gate_max_min")
    smoothing = config.get("smoothing")
    cap_min_factor = config.get("cap_min_factor")
    cap_quality_gamma = config.get("cap_quality_gamma")
    high_signal = config.get("high_signal")

    provider = LogitUpliftProvider(
        vectorizer_path=vec_path,
        centroids_path=cen_path,
        weights_path=w_path,
        max_total_boost=max_total_boost,
        sim_gate_sum_min=sim_gate_sum_min,
        sim_gate_max_min=sim_gate_max_min,
        smoothing=smoothing,
        cap_min_factor=cap_min_factor,
        cap_quality_gamma=cap_quality_gamma,
        high_signal=high_signal if isinstance(high_signal, dict) else None,
    )
    return GatedTextBoostProvider(provider)
