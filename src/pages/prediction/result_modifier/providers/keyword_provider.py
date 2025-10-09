from __future__ import annotations

from src.pages.prediction.result_modifier.keyword_booster import KeywordBooster
from src.pages.prediction.result_modifier.text_boost_provider import TextBoostProvider


class KeywordBoosterProvider(TextBoostProvider):
    def __init__(self, max_total_boost: float = 0.10):
        self._max_total_boost = max_total_boost

    def apply(
        self, probabilities: list[float], experience_details: dict[str, str]
    ) -> tuple[list[float], str]:
        if not probabilities:
            return probabilities, ""

        boosted = KeywordBooster.apply_keyword_boost(probabilities, experience_details)

        if self._max_total_boost is not None and self._max_total_boost < 0.10:
            ratio_list = []
            for b, p in zip(boosted, probabilities, strict=False):
                if p > 0:
                    ratio_list.append((b / p) - 1)
            if ratio_list:
                total_boost = max(0.0, sum([max(0.0, r) for r in ratio_list]) / len(ratio_list))
                if total_boost > self._max_total_boost:
                    factor = (1 + self._max_total_boost) / (1 + total_boost)
                    boosted = [
                        min(p * factor * (1 + total_boost), 1.0) if p > 0 else p
                        for p in probabilities
                    ]

        summary = KeywordBooster.get_boost_summary(experience_details)
        return boosted, summary
