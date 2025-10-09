from __future__ import annotations

import json
from functools import lru_cache

from src.pages.prediction.result_modifier.utils import has_valid_experience_details


class TextBoostProvider:
    def apply(
        self, probabilities: list[float], experience_details: dict[str, str]
    ) -> tuple[list[float], str]:
        raise NotImplementedError


class NullTextBoostProvider(TextBoostProvider):
    def apply(
        self, probabilities: list[float], experience_details: dict[str, str]
    ) -> tuple[list[float], str]:
        return probabilities, ""


class GatedTextBoostProvider(TextBoostProvider):
    def __init__(self, inner: TextBoostProvider):
        self._inner = inner

    def apply(
        self, probabilities: list[float], experience_details: dict[str, str]
    ) -> tuple[list[float], str]:
        if not has_valid_experience_details(experience_details):
            return probabilities, ""
        return self._inner.apply(probabilities, experience_details)


class CachingTextBoostProvider(TextBoostProvider):
    def __init__(self, inner: TextBoostProvider, anchor_probability: float = 0.5):
        self._inner = inner
        self._anchor = float(anchor_probability)

    def _signature_from_details(self, experience_details: dict[str, str]) -> str:
        if not isinstance(experience_details, dict):
            return "{}"
        keys = ("research_details", "award_details", "internship_details", "paper_details")
        normalized = {k: (str(experience_details.get(k, "")).strip()) for k in keys}
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)

    @lru_cache(maxsize=512)
    def _cached_delta_for_signature(self, sig: str) -> tuple[float, str]:
        try:
            boosted, summary = self._inner.apply([self._anchor], {"_key": sig})
            if not boosted:
                return 0.0, summary or ""
            delta = float(boosted[0]) - self._anchor
            return max(-1.0, min(1.0, delta)), summary or ""
        except Exception:
            return 0.0, ""

    def apply(
        self, probabilities: list[float], experience_details: dict[str, str]
    ) -> tuple[list[float], str]:
        if not probabilities:
            return probabilities, ""
        sig = self._signature_from_details(experience_details)
        delta, summary = self._cached_delta_for_signature(sig)
        updated = []
        for p in probabilities:
            try:
                np = max(0.0, min(1.0, float(p) + delta))
            except Exception:
                np = p
            updated.append(np)
        return updated, summary


def get_text_boost_provider(config: dict | None) -> TextBoostProvider:
    if not config or not config.get("enabled"):
        return NullTextBoostProvider()

    try:
        key = json.dumps(config or {}, ensure_ascii=False, sort_keys=True)
        return _get_text_boost_provider_cached(key)
    except Exception:
        return NullTextBoostProvider()


@lru_cache(maxsize=16)
def _get_text_boost_provider_cached(config_key: str) -> TextBoostProvider:
    try:
        config = json.loads(config_key)
        from src.pages.prediction.result_modifier.providers.keyword_provider import (
            KeywordBoosterProvider,
        )
        from src.pages.prediction.result_modifier.providers.tfidf_provider import (
            TfidfBoostProvider,
        )

        model_paths = (config or {}).get("model_paths", {})
        vec_path = model_paths.get("tfidf_vectorizer")
        cen_path = model_paths.get("tfidf_centroids")
        if not vec_path or not cen_path:
            return NullTextBoostProvider()

        similarity_thresholds = config.get("similarity_thresholds")
        max_total_boost = config.get("max_total_boost", 0.10)

        kw = KeywordBoosterProvider(max_total_boost=max_total_boost)
        tf = TfidfBoostProvider(
            vectorizer_path=vec_path,
            centroids_path=cen_path,
            max_total_boost=max_total_boost,
            timeout_ms=int(config.get("timeout_ms", 100)),
            similarity_thresholds=similarity_thresholds,
        )
        try:
            tf._lazy_load()
        except Exception:
            pass

        class MaxOfTwoProvider(TextBoostProvider):
            def apply(self, probabilities, experience_details):
                p1, s1 = kw.apply(list(probabilities), experience_details)
                p2, s2 = tf.apply(list(probabilities), experience_details)
                merged = []
                for a, b in zip(p1, p2, strict=False):
                    merged.append(max(float(a), float(b)))

                def _boost_from_summary(txt: str) -> float:
                    try:
                        if not txt:
                            return 0.0
                        if txt.startswith("+") and "%" in txt:
                            return float(txt[1 : txt.index("%")]) / 100.0
                    except Exception:
                        return 0.0
                    return 0.0

                sum1 = _boost_from_summary(s1)
                sum2 = _boost_from_summary(s2)
                summary = s1 if sum1 >= sum2 else s2
                return merged, summary

        return GatedTextBoostProvider(CachingTextBoostProvider(MaxOfTwoProvider()))
    except Exception:
        return NullTextBoostProvider()
