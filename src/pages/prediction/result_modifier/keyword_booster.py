from typing import Any, Dict

from src.pages.prediction.result_modifier.config import KEYWORD_BOOSTER_CACHE_SIZE
from src.pages.prediction.result_modifier.keyword_config import (
    BOOST_MULTIPLIERS,
    KEYWORD_WEIGHTS,
    MAX_SINGLE_EXPERIENCE_BOOST,
)
from src.pages.prediction.result_modifier.utils import (
    generate_content_hash,
    has_valid_experience_details,
    is_effectively_empty,
)


class KeywordBooster:
    _boost_cache: Dict[str, Any] = {}

    @classmethod
    def calculate_keyword_boost(cls, experience_details: dict[str, str]) -> dict[str, float]:
        if not has_valid_experience_details(experience_details):
            return {"research": 0.0, "award": 0.0, "internship": 0.0, "paper": 0.0}

        cache_key = cls._generate_cache_key(experience_details)
        if cache_key in cls._boost_cache:
            return cls._boost_cache[cache_key]

        boosts = {}
        for experience_type in ["research", "award", "internship", "paper"]:
            detail_key = f"{experience_type}_details"
            details = experience_details.get(detail_key, "").strip()

            if is_effectively_empty(details):
                boosts[experience_type] = 0.0
                continue

            boost = cls._calculate_single_experience_boost(experience_type, details)
            boosts[experience_type] = boost

        cls._boost_cache[cache_key] = boosts

        if len(cls._boost_cache) > KEYWORD_BOOSTER_CACHE_SIZE:
            keys_to_remove = list(cls._boost_cache.keys())[: KEYWORD_BOOSTER_CACHE_SIZE // 2]
            for key in keys_to_remove:
                del cls._boost_cache[key]

        return boosts

    @classmethod
    def _generate_cache_key(cls, experience_details: dict[str, str]) -> str:
        content = []
        for exp_type in ["research", "award", "internship", "paper"]:
            detail_key = f"{exp_type}_details"
            details = experience_details.get(detail_key, "").strip()
            content.append(f"{exp_type}:{details}")
        combined = "|".join(content)
        return generate_content_hash(combined)

    @classmethod
    def _calculate_single_experience_boost(cls, experience_type: str, details: str) -> float:
        if experience_type not in KEYWORD_WEIGHTS:
            return 0.0

        keywords_config = KEYWORD_WEIGHTS[experience_type]
        total_boost = 0.0
        details_lower = details.lower()

        for tier in ["top_tier", "high_tier", "medium_tier", "general"]:
            keywords = keywords_config.get(tier, [])
            tier_boost = 0.0

            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in details_lower:
                    tier_boost = max(tier_boost, BOOST_MULTIPLIERS[tier])

            total_boost += tier_boost

        return min(total_boost, MAX_SINGLE_EXPERIENCE_BOOST)

    @classmethod
    def apply_keyword_boost(
        cls, probabilities: list[float], experience_details: dict[str, str]
    ) -> list[float]:
        if not experience_details or not probabilities:
            return probabilities

        boosts = cls.calculate_keyword_boost(experience_details)
        total_boost = sum(boosts.values())

        if total_boost == 0:
            return probabilities

        boosted_probabilities = []
        for prob in probabilities:
            boosted_prob = min(prob * (1 + total_boost), 1.0)
            boosted_probabilities.append(boosted_prob)

        return boosted_probabilities

    @classmethod
    def get_boost_summary(cls, experience_details: dict[str, str]) -> str:
        boosts = cls.calculate_keyword_boost(experience_details)

        if not any(boosts.values()):
            return ""

        summary_parts = []
        for exp_type, boost in boosts.items():
            if boost > 0:
                exp_names = {
                    "research": "科研项目",
                    "award": "获奖情况",
                    "internship": "实习经历",
                    "paper": "论文发表",
                }
                summary_parts.append(f"{exp_names[exp_type]}: +{boost:.1%}")

        total_boost = sum(boosts.values())
        return f"+{total_boost:.1%} ({', '.join(summary_parts)})"

    @classmethod
    def clear_cache(cls):
        cls._boost_cache.clear()

    @classmethod
    def get_cache_stats(cls) -> dict[str, int]:
        return {"cache_size": len(cls._boost_cache), "cache_limit": KEYWORD_BOOSTER_CACHE_SIZE}
