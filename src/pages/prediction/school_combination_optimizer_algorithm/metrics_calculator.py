# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
from functools import lru_cache
from typing import Any

import numpy as np

from src.pages.prediction.core.utils import is_new_major
from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    BALANCE_RATIOS,
    PRESTIGE_WEIGHT,
    SCHOOL_CATEGORY_THRESHOLDS,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    clip_probability,
)
from src.utils.school_level_service import (
    SCHOOL_LEVEL_PRIORITY,
    get_school_level_service,
)


def calculate_metrics(
    schools: list[dict[str, Any]],
    background_major: str = "",
    adaptive_thresholds: dict[str, float] = None,
    bg_target_similarity_cache: dict = None,
    new_major_cache: dict = None,
    background_faculty: str = None,
    major_category_cache: dict = None,
) -> dict[str, Any]:
    if not schools:
        return _get_empty_metrics()

    probabilities = np.array(
        [clip_probability(school.get("probability", 0.0)) for school in schools]
    )
    rejection_prob = np.prod(1.0 - probabilities)

    current_thresholds = adaptive_thresholds or SCHOOL_CATEGORY_THRESHOLDS
    safety_thresh_val = current_thresholds["safety"]
    target_thresh_val = current_thresholds["target_lower"]

    safety_count = np.sum(probabilities >= safety_thresh_val)
    target_count = np.sum(
        (probabilities >= target_thresh_val) & (probabilities < safety_thresh_val)
    )
    reach_count = np.sum(probabilities < target_thresh_val)

    prestige_score = _calculate_prestige_score(schools)
    balance_score = _calculate_balance_score(
        len(schools), safety_count, target_count, reach_count, prestige_score
    )

    major_similarity = _calculate_major_similarity(schools, background_major)
    new_major_count, new_major_ratio = _calculate_new_major_stats(schools, new_major_cache)

    return {
        "rejection_probability": rejection_prob,
        "admission_probability": 1.0 - rejection_prob,
        "diversity": len({school["university"] for school in schools}),
        "safety_count": safety_count,
        "target_count": target_count,
        "reach_count": reach_count,
        "balance_score": balance_score,
        "major_similarity": major_similarity,
        "new_major_ratio": new_major_ratio,
        "new_major_count": new_major_count,
        "simulated_rejection_probability": None,
        "simulated_admission_probability": None,
    }


def _get_empty_metrics() -> dict[str, Any]:
    return {
        "rejection_probability": 1.0,
        "admission_probability": 0.0,
        "diversity": 0,
        "safety_count": 0,
        "target_count": 0,
        "reach_count": 0,
        "balance_score": 0.0,
        "major_similarity": 0.0,
        "new_major_ratio": 0.0,
        "new_major_count": 0,
        "simulated_rejection_probability": None,
        "simulated_admission_probability": None,
    }


def _calculate_prestige_score(schools: list[dict[str, Any]]) -> float:
    if not schools:
        return 0.0
    priorities = [_get_priority_cached(school.get("university", "")) for school in schools]
    return max(0.0, min(1.0, (12 - np.mean(priorities)) / 11))


@lru_cache(maxsize=5000)
def _get_priority_cached(university: str) -> int:
    svc = get_school_level_service()
    info = svc.get_school_info(university or "")
    return int(info.get("priority", SCHOOL_LEVEL_PRIORITY.get("未知", 12)))


def _calculate_balance_score(
    total: int,
    safety_count: int,
    target_count: int,
    reach_count: int,
    prestige_score: float,
) -> float:
    ideal_safety = total * BALANCE_RATIOS["safety"]
    ideal_target = total * BALANCE_RATIOS["target"]
    ideal_reach = total * BALANCE_RATIOS["reach"]

    balance_score = 10 - (
        (safety_count - ideal_safety) ** 2
        + (target_count - ideal_target) ** 2
        + (reach_count - ideal_reach) ** 2
    )
    return balance_score + PRESTIGE_WEIGHT * prestige_score


def _calculate_major_similarity(
    schools: list[dict[str, Any]],
    background_major: str,
) -> float:
    if not background_major:
        return 0.0

    similarities = [
        float(school.get("similarity", 0.0))
        for school in schools
        if "similarity" in school and school["similarity"] is not None
    ]

    return np.mean(similarities) if similarities else 0.0


def _calculate_new_major_stats(
    schools: list[dict[str, Any]], new_major_cache: dict | None
) -> tuple[int, float]:
    if new_major_cache is None:
        new_major_count = sum(
            1
            for school in schools
            if school.get("university")
            and school.get("major")
            and is_new_major(school["university"], school["major"])
        )
    else:
        new_major_count = sum(
            1
            for school in schools
            if new_major_cache.get(f"{school.get('university')}|{school.get('major')}")
        )

    new_major_ratio = new_major_count / len(schools) if schools else 0.0
    return new_major_count, new_major_ratio
