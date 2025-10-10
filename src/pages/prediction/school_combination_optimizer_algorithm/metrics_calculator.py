from functools import lru_cache
from typing import Any

import numpy as np

from src.pages.prediction.prediction_utils import get_cached_major_similarity, is_new_major
from src.pages.prediction.school_combination_optimizer_algorithm.common_utils import (
    clip_probability,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer_config import (
    BALANCE_RATIOS,
    PRESTIGE_WEIGHT,
    SCHOOL_CATEGORY_THRESHOLDS,
)
from src.utils.school_level_service import SCHOOL_LEVEL_PRIORITY, get_school_level_service


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
    admission_prob = 1.0 - rejection_prob

    universities = {school["university"] for school in schools}
    diversity = len(universities)

    prestige_score = _calculate_prestige_score(schools)

    current_thresholds = adaptive_thresholds or SCHOOL_CATEGORY_THRESHOLDS
    safety_thresh_val = current_thresholds["safety"]
    target_thresh_val = current_thresholds["target_lower"]

    safety_count = np.sum(probabilities >= safety_thresh_val)
    target_count = np.sum(
        (probabilities >= target_thresh_val) & (probabilities < safety_thresh_val)
    )
    reach_count = np.sum(probabilities < target_thresh_val)

    balance_score = _calculate_balance_score(
        len(schools), safety_count, target_count, reach_count, prestige_score
    )

    major_similarity = _calculate_major_similarity(
        schools, background_major, bg_target_similarity_cache
    )

    new_major_count, new_major_ratio = _calculate_new_major_stats(schools, new_major_cache)

    (
        major_category_score,
        cross_major_ratio,
        major_category_diversity,
    ) = _calculate_category_stats(
        schools,
        background_major,
        background_faculty,
        major_category_cache,
    )

    return {
        "rejection_probability": rejection_prob,
        "admission_probability": admission_prob,
        "diversity": diversity,
        "safety_count": safety_count,
        "target_count": target_count,
        "reach_count": reach_count,
        "balance_score": balance_score,
        "major_similarity": major_similarity,
        "new_major_ratio": new_major_ratio,
        "new_major_count": new_major_count,
        "simulated_rejection_probability": None,
        "simulated_admission_probability": None,
        "major_category_score": major_category_score,
        "cross_major_ratio": cross_major_ratio,
        "major_category_diversity": major_category_diversity,
    }


def _get_empty_metrics() -> dict[str, Any]:
    return {
        "rejection_probability": 1.0,
        "admission_probability": 0.0,
        "diversity": 0,
        "safety_count": 0,
        "target_count": 0,
        "reach_count": 0,
        "balance_score": 0,
        "major_similarity": 0,
        "new_major_ratio": 0,
        "new_major_count": 0,
        "simulated_rejection_probability": 1.0,
        "simulated_admission_probability": 0.0,
        "major_category_score": 0,
        "cross_major_ratio": 0,
        "major_category_diversity": 0,
    }


def _calculate_prestige_score(schools: list[dict[str, Any]]) -> float:
    try:
        svc = get_school_level_service()
        priorities = []
        for school in schools:
            uni = school.get("university", "")
            priorities.append(_get_priority_cached(uni))
        if priorities:
            avg_priority = float(np.mean(priorities))
            return max(0.0, min(1.0, (12 - avg_priority) / 11))
        return 0.0
    except Exception:
        return 0.0


@lru_cache(maxsize=5000)
def _get_priority_cached(university: str) -> int:
    try:
        svc = get_school_level_service()
        info = svc.get_school_info(university or "")
        return int(info.get("priority", SCHOOL_LEVEL_PRIORITY.get("未知", 12)))
    except Exception:
        return int(SCHOOL_LEVEL_PRIORITY.get("未知", 12))


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
    bg_target_similarity_cache: dict | None,
) -> float:
    if not background_major:
        return 0.0

    cache = bg_target_similarity_cache or {}
    similarities = []

    for school in schools:
        target_major = school.get("major", "")
        if not target_major:
            similarities.append(0.0)
            continue

        cache_key = f"{background_major}|{target_major}"
        if cache_key in cache:
            similarity = cache[cache_key]
        else:
            similarity = get_cached_major_similarity(
                target_major=target_major,
                background_major=background_major,
                cache=cache,
            )
        similarities.append(similarity)

    return np.mean(similarities) if similarities else 0.0


def _calculate_new_major_stats(
    schools: list[dict[str, Any]],
    new_major_cache: dict | None,
) -> tuple[int, float]:
    new_major_count = 0

    if new_major_cache is not None:
        for school in schools:
            university = school.get("university", "")
            major = school.get("major", "")
            cache_key = f"{university}|{major}"
            if new_major_cache.get(cache_key, False):
                new_major_count += 1
    else:
        for school in schools:
            university = school.get("university", "")
            major = school.get("major", "")
            if university and major and is_new_major(university, major):
                new_major_count += 1

    new_major_ratio = new_major_count / len(schools) if len(schools) > 0 else 0.0
    return new_major_count, new_major_ratio


def _calculate_category_stats(
    schools: list[dict[str, Any]],
    background_major: str,
    background_faculty: str | None,
    major_category_cache: dict | None,
) -> tuple[float, float, int]:
    if not background_faculty or major_category_cache is None:
        return 1.0, 0.0, 0

    category_counts: dict[str, int] = {}
    same_category_count = 0

    for school in schools:
        university = school.get("university", "")
        major = school.get("major", "")
        cache_key = f"{university}|{major}"
        target_category = major_category_cache.get(cache_key, "")

        if target_category:
            category_counts[target_category] = category_counts.get(target_category, 0) + 1
            if target_category == background_faculty:
                same_category_count += 1

    cross_major_count = len(schools) - same_category_count
    cross_major_ratio = cross_major_count / len(schools) if len(schools) > 0 else 0.0
    major_category_diversity = len(category_counts)

    major_category_score = 1.0 - (cross_major_ratio * 0.5)

    return major_category_score, cross_major_ratio, major_category_diversity
