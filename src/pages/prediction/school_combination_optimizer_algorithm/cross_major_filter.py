from typing import Any

from src.pages.prediction.prediction_utils import get_cached_major_similarity
from src.pages.prediction.school_combination_optimizer_algorithm.domain_rules import (
    get_exclude_rules_for_background,
)
from src.pages.prediction.school_combination_optimizer_algorithm.major_category_config import (
    RELATED_GROUPS,
)


def filter_schools_by_cross_major_feasibility(
    schools: list[dict[str, Any]],
    background_major: str,
    background_major_category: str | None,
    major_category_cache: dict[str, str] | None,
    bg_target_similarity_cache: dict | None,
    recall_filter_cfg: dict,
) -> list[dict[str, Any]]:
    if (
        not schools
        or not background_major
        or not background_major_category
        or not major_category_cache
    ):
        return schools

    same_group_min = recall_filter_cfg.get("same_group_similarity_min", 0.7)
    cross_group_min = recall_filter_cfg.get("cross_group_similarity_min", 0.85)
    global_min = recall_filter_cfg.get("global_min_similarity", 0.65)

    filtered: list[dict[str, Any]] = []
    cache = bg_target_similarity_cache or {}

    for s in schools:
        university = s.get("university", "")
        major = s.get("major", "")
        if not university or not major:
            filtered.append(s)
            continue

        cache_key = f"{university}|{major}"
        target_category = major_category_cache.get(cache_key, "")
        if not target_category:
            target_category = "__UNKNOWN__"

        rules = get_exclude_rules_for_background(background_major_category)
        if rules:
            excluded_categories = rules.get("categories", set())
            excluded_keywords = rules.get("keywords", [])
            if target_category in excluded_categories:
                continue
            if excluded_keywords and _contains_any(major, excluded_keywords):
                continue

        if target_category == background_major_category:
            filtered.append(s)
            continue

        target_major = s.get("major", "")
        sim = get_cached_major_similarity(
            target_major=target_major,
            background_major=background_major,
            cache=cache,
        )

        if sim < global_min:
            continue

        if target_category != "__UNKNOWN__" and _in_same_group(
            background_major_category, target_category
        ):
            if sim >= same_group_min:
                filtered.append(s)
        else:
            if sim >= cross_group_min:
                filtered.append(s)

    return filtered


def _in_same_group(cat_a: str, cat_b: str) -> bool:
    for members in RELATED_GROUPS.values():
        if cat_a in members and cat_b in members:
            return True
    return False


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = (text or "").lower()
    for kw in keywords:
        if kw in t:
            return True
    return False
