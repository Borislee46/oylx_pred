from functools import lru_cache
from typing import Any, Optional

from src.pages.prediction.result_modifier.config import UNIVERSITY_DIFFICULTY_ORDER
from src.pages.prediction.school_combination_optimizer_algorithm.school_selector import (
    reduce_schools_balanced,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    clip_probability,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


@lru_cache(maxsize=1)
def _get_difficulty_map() -> dict[str, int]:
    return {uni: idx for idx, uni in enumerate(UNIVERSITY_DIFFICULTY_ORDER)}


def adjust_probability_by_university_difficulty(
    schools: list[dict[str, Any]],
    adaptive_thresholds: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    if not schools:
        return schools

    difficulty_map = _get_difficulty_map()
    target_thresh = adaptive_thresholds.get("target_lower", 0.55) if adaptive_thresholds else 0.55
    total_universities = len(UNIVERSITY_DIFFICULTY_ORDER)

    adjusted_schools = []
    for school in schools:
        university = school.get("university", "")
        current_prob = clip_probability(school.get("probability", 0.0))

        difficulty_rank = difficulty_map.get(university, total_universities)
        normalized_rank = difficulty_rank / total_universities if total_universities > 0 else 0.5

        if normalized_rank >= 0.3:
            if current_prob < target_thresh:
                adjustment_factor = (normalized_rank - 0.3) / 0.7 if normalized_rank > 0.3 else 0.1
                boost_amount = max(0.08, 0.15 * adjustment_factor)
                adjusted_prob = min(1.0, current_prob + boost_amount)

                if adjusted_prob < target_thresh:
                    adjusted_prob = target_thresh + 0.02

                school = {**school, "probability": min(1.0, adjusted_prob)}
            elif current_prob < target_thresh + 0.15:
                adjustment_factor = (normalized_rank - 0.3) / 0.7 if normalized_rank > 0.3 else 0.1
                boost_amount = 0.06 * adjustment_factor
                adjusted_prob = min(1.0, current_prob + boost_amount)
                school = {**school, "probability": adjusted_prob}

        adjusted_schools.append(school)

    return adjusted_schools


def enforce_school_limits(
    schools: list[dict[str, Any]],
    min_schools: int,
    max_schools: int,
    context_adaptive_thresholds: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    num_selected = len(schools)

    if num_selected < min_schools:
        return []
    elif num_selected > max_schools:
        return reduce_schools_balanced(schools, max_schools, context_adaptive_thresholds)

    return schools
