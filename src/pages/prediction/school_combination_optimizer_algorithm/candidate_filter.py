from typing import Any

from src.pages.prediction.school_combination_optimizer_algorithm.common_utils import (
    normalize_school_name,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer_config import (
    PRIORITY_THRESHOLD_NORMAL_BG_DEFAULT,
    PRIORITY_THRESHOLD_NORMAL_BG_GPA_GE_3_0,
    PRIORITY_THRESHOLD_TOP_BG_DEFAULT,
    PRIORITY_THRESHOLD_TOP_BG_GPA_GE_2_8,
    PRIORITY_THRESHOLD_TOP_BG_GPA_GE_3_2,
    TOP5_SCHOOLS,
    TOP8_SCHOOLS,
    TOP_BG_LEVELS_SET,
)
from src.utils.school_level_service import SCHOOL_LEVEL_PRIORITY, get_school_level_service


def filter_candidates_by_background(
    all_schools_data: list[dict[str, Any]],
    school_level: str | None,
    gpa: float | None,
    min_schools: int = 1,
) -> list[dict[str, Any]]:
    is_high_bg_high_gpa = (
        school_level in {"985", "211", "1-50", "51-100"} and gpa is not None and gpa >= 3.2
    )

    max_allowed_priority = _determine_max_allowed_priority(school_level, gpa)
    if max_allowed_priority is not None:
        all_schools_data = _filter_by_priority(
            all_schools_data, max_allowed_priority, preserve_top_schools=is_high_bg_high_gpa
        )

    if is_high_bg_high_gpa:
        all_schools_data = _apply_top8_priority(all_schools_data, min_schools)

    return all_schools_data


def _determine_max_allowed_priority(school_level: str | None, gpa: float | None) -> int | None:
    if not school_level:
        return None

    if school_level in TOP_BG_LEVELS_SET:
        if gpa is not None and gpa >= 3.2:
            return PRIORITY_THRESHOLD_TOP_BG_GPA_GE_3_2
        elif gpa is not None and gpa >= 2.8:
            return PRIORITY_THRESHOLD_TOP_BG_GPA_GE_2_8
        else:
            return PRIORITY_THRESHOLD_TOP_BG_DEFAULT

    if school_level == "普通本科":
        if gpa is not None and gpa >= 3.0:
            return PRIORITY_THRESHOLD_NORMAL_BG_GPA_GE_3_0
        else:
            return PRIORITY_THRESHOLD_NORMAL_BG_DEFAULT

    return None


def _filter_by_priority(
    schools: list[dict[str, Any]], max_priority: int, preserve_top_schools: bool = False
) -> list[dict[str, Any]]:
    try:
        svc = get_school_level_service()
        norm_top5 = (
            {normalize_school_name(u) for u in TOP5_SCHOOLS} if preserve_top_schools else set()
        )

        filtered = []
        for s in schools:
            uni = s.get("university", "")
            norm_uni = normalize_school_name(uni)

            if preserve_top_schools and norm_uni in norm_top5:
                filtered.append(s)
                continue

            info = svc.get_school_info(uni)
            priority = info.get("priority", SCHOOL_LEVEL_PRIORITY.get("未知", 12))
            if priority <= max_priority:
                filtered.append(s)
        return filtered if filtered else schools
    except Exception:
        return schools


def _apply_top8_priority(
    all_schools_data: list[dict[str, Any]], min_schools: int
) -> list[dict[str, Any]]:
    norm_top8 = {normalize_school_name(u) for u in TOP8_SCHOOLS}

    top8_filtered = [
        s
        for s in (all_schools_data or [])
        if normalize_school_name(s.get("university")) in norm_top8
    ]

    if not top8_filtered:
        return all_schools_data

    if len(top8_filtered) >= max(1, int(min_schools)):
        return top8_filtered

    need = max(0, int(min_schools) - len(top8_filtered))
    others = [
        s
        for s in (all_schools_data or [])
        if normalize_school_name(s.get("university")) not in norm_top8
    ]
    others_sorted = sorted(others, key=lambda x: x.get("probability", 0.0), reverse=True)
    return top8_filtered + others_sorted[:need]
