from typing import Any

from src.pages.prediction.school_combination_optimizer_algorithm.common_utils import (
    normalize_school_name,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer_config import (
    MIN_SAFETY_SCHOOL_COUNT_DEFAULT,
    MIN_SAFETY_SCHOOL_COUNT_HIGH_BG,
    PRIORITY_THRESHOLD_NORMAL_BG_DEFAULT,
    PRIORITY_THRESHOLD_NORMAL_BG_GPA_GE_3_0,
    PRIORITY_THRESHOLD_TOP_BG_DEFAULT,
    PRIORITY_THRESHOLD_TOP_BG_GPA_GE_2_8,
    PRIORITY_THRESHOLD_TOP_BG_GPA_GE_3_2,
    TOP5_SCHOOLS,
    TOP8_SCHOOLS,
    TOP_BG_LEVELS_SET,
    SCHOOL_CATEGORY_THRESHOLDS,
)
from src.utils.logger import setup_logger
from src.utils.school_level_service import SCHOOL_LEVEL_PRIORITY, get_school_level_service
from src.utils.app_data_loader import load_raw_cases_data

logger = setup_logger("page3", "prediction")


MIN_SAMPLE_SIZE_THRESHOLD = 30


def _build_target_combo_sample_counts() -> dict[tuple[str, str], int]:
    try:
        df = load_raw_cases_data()
        if df is None or df.empty:
            return {}

        required_cols = {"target_university", "target_major"}
        if not required_cols.issubset(set(df.columns)):
            return {}

        counts_series = (
            df.groupby(["target_university", "target_major"]).size()
            if not df.empty
            else None
        )
        if counts_series is None or counts_series.empty:
            return {}

        counts_dict: dict[tuple[str, str], int] = {}
        for (uni, maj), cnt in counts_series.items():
            counts_dict[(str(uni), str(maj))] = int(cnt)
        return counts_dict
    except Exception:
        return {}


def filter_candidates_by_background(
    all_schools_data: list[dict[str, Any]],
    school_level: str | None,
    gpa: float | None,
    min_schools: int = 1,
    background_faculty: str | None = None,
    adaptive_thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    sample_counts = _build_target_combo_sample_counts()
    if sample_counts:
        try:
            all_schools_data = [
                s
                for s in (all_schools_data or [])
                if sample_counts.get((str(s.get("university", "")), str(s.get("major", ""))), 0)
                >= MIN_SAMPLE_SIZE_THRESHOLD
            ]
        except Exception:
            pass
    is_high_bg_high_gpa = (
        school_level in {"985", "211", "1-50", "51-100"} and gpa is not None and gpa >= 3.2
    )

    original_schools_data = all_schools_data[:]

    max_allowed_priority = _determine_max_allowed_priority(school_level, gpa)
    if max_allowed_priority is not None:
        filtered_schools = _filter_by_priority(
            all_schools_data, max_allowed_priority, preserve_top_schools=is_high_bg_high_gpa
        )

    else:
        filtered_schools = all_schools_data

    if is_high_bg_high_gpa:
        filtered_schools = _apply_top8_priority(filtered_schools, min_schools)

    safety_threshold = (
        (adaptive_thresholds or SCHOOL_CATEGORY_THRESHOLDS).get("safety", SCHOOL_CATEGORY_THRESHOLDS["safety"])
    )

    filtered_schools = _ensure_safety_schools(
        filtered_schools=filtered_schools,
        original_schools=original_schools_data,
        is_high_bg=is_high_bg_high_gpa,
        background_faculty=background_faculty,
        safety_threshold=safety_threshold,
    )

    return filtered_schools


def _ensure_safety_schools(
    filtered_schools: list[dict[str, Any]],
    original_schools: list[dict[str, Any]],
    is_high_bg: bool,
    background_faculty: str | None,
    safety_threshold: float,
) -> list[dict[str, Any]]:
    min_safety_needed = (
        MIN_SAFETY_SCHOOL_COUNT_HIGH_BG if is_high_bg else MIN_SAFETY_SCHOOL_COUNT_DEFAULT
    )

    current_safety_count = sum(
        1 for s in filtered_schools if s.get("probability", 0.0) >= safety_threshold
    )

    if current_safety_count >= min_safety_needed:
        return filtered_schools

    needed = min_safety_needed - current_safety_count

    potential_safety_schools = [
        s for s in original_schools if s.get("probability", 0.0) >= safety_threshold
    ]

    existing_schools_set = {(s.get("university"), s.get("major")) for s in filtered_schools}
    potential_safety_schools = [
        s
        for s in potential_safety_schools
        if (s.get("university"), s.get("major")) not in existing_schools_set
    ]

    same_faculty_safety = []
    other_faculty_safety = []
    if background_faculty:
        from src.pages.prediction.school_combination_optimizer_algorithm.problem_initializer import (
            build_major_category_cache,
        )
        from src.utils.app_data_loader import load_school_major_details_df

        details_df = load_school_major_details_df()
        major_category_cache = build_major_category_cache(details_df)

        for s in potential_safety_schools:
            uni = s.get("university", "")
            maj = s.get("major", "")
            cache_key = f"{uni}|{maj}"
            faculty = major_category_cache.get(cache_key)
            if faculty == background_faculty:
                same_faculty_safety.append(s)
            else:
                other_faculty_safety.append(s)
    else:
        other_faculty_safety = potential_safety_schools

    same_faculty_safety.sort(key=lambda s: s.get("probability", 0.0), reverse=True)
    other_faculty_safety.sort(key=lambda s: s.get("probability", 0.0), reverse=True)

    sorted_safety_pool = same_faculty_safety + other_faculty_safety

    schools_to_add = sorted_safety_pool[:needed]

    return filtered_schools + schools_to_add


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
