from collections import defaultdict
from typing import Any

from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    MIN_SAFETY_SCHOOL_COUNT_DEFAULT,
    MIN_SAFETY_SCHOOL_COUNT_HIGH_BG,
    PRIORITY_THRESHOLD_NORMAL_BG_DEFAULT,
    PRIORITY_THRESHOLD_NORMAL_BG_GPA_GE_3_0,
    PRIORITY_THRESHOLD_TOP_BG_DEFAULT,
    PRIORITY_THRESHOLD_TOP_BG_GPA_GE_2_8,
    PRIORITY_THRESHOLD_TOP_BG_GPA_GE_3_2,
    SCHOOL_CATEGORY_THRESHOLDS,
    TOP5_SCHOOLS,
    TOP8_SCHOOLS,
    TOP_BG_LEVELS_SET,
    get_allowed_target_faculties,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    normalize_major_name,
    normalize_school_name,
)
from src.utils.app_data_loader import load_raw_cases_data
from src.utils.school_level_service import (
    SCHOOL_LEVEL_PRIORITY,
    get_school_level_service,
)

MIN_SAMPLE_SIZE_THRESHOLD = 30


def deduplicate_majors(schools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not schools:
        return []

    grouped = defaultdict(list)
    for school in schools:
        if (uni := school.get("university")) and (major := school.get("major")):
            major_key = school.get("major_norm") or normalize_major_name(major)
            grouped[(uni, major_key)].append(school)

    return [
        (
            school_group[0]
            if len(school_group) == 1
            else max(school_group, key=lambda s: s.get("probability", 0.0))
        )
        for school_group in grouped.values()
    ]


def deduplicate_universities_by_similarity(
    schools: list[dict[str, Any]],
    background_major: str,
    similarity_cache: dict,
    target_universities: set[str],
) -> list[dict[str, Any]]:
    if not schools or not background_major or not target_universities:
        return schools

    grouped_by_uni = defaultdict(list)
    others = []

    for s in schools:
        uni = s.get("university", "")
        if uni in target_universities:
            grouped_by_uni[uni].append(s)
        else:
            others.append(s)

    def score(item: dict[str, Any]) -> tuple[float, float]:
        return float(item.get("similarity", 0.0)), float(item.get("probability", 0.0) or 0.0)

    picked = [
        items[0] if len(items) == 1 else max(items, key=score) for items in grouped_by_uni.values()
    ]

    return others + picked


def filter_schools_by_faculty_rules(
    schools: list[dict[str, Any]],
    background_faculty: str | None,
) -> list[dict[str, Any]]:
    if not schools or not background_faculty:
        return schools

    allowed_faculties = get_allowed_target_faculties(background_faculty)
    if not allowed_faculties:
        return schools

    filtered = []
    for school in schools:
        faculty = school.get("faculty", "").strip()
        if not faculty:
            filtered.append(school)
        elif faculty in allowed_faculties:
            filtered.append(school)
    return filtered


def _build_target_combo_sample_counts() -> dict[tuple[str, str], int]:
    df = load_raw_cases_data()
    if df is None or df.empty or not {"target_university", "target_major"}.issubset(df.columns):
        return {}

    counts_series = df.groupby(["target_university", "target_major"]).size()
    return (
        {(str(uni), str(maj)): int(cnt) for (uni, maj), cnt in counts_series.items()}
        if not counts_series.empty
        else {}
    )


def _filter_by_sample_size(schools_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_counts = _build_target_combo_sample_counts()
    if not sample_counts:
        return schools_data

    return [
        s
        for s in schools_data
        if sample_counts.get((str(s.get("university", "")), str(s.get("major", ""))), 0)
        >= MIN_SAMPLE_SIZE_THRESHOLD
    ]


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

    if school_level in {"普通本科", "101-200", "201-300", "301-500", "500之后"}:
        return (
            PRIORITY_THRESHOLD_NORMAL_BG_GPA_GE_3_0
            if gpa is not None and gpa >= 3.0
            else PRIORITY_THRESHOLD_NORMAL_BG_DEFAULT
        )

    return None


def _filter_by_priority(
    schools: list[dict[str, Any]], max_priority: int, preserve_top_schools: bool = False
) -> list[dict[str, Any]]:
    svc = get_school_level_service()
    norm_top5 = {normalize_school_name(u) for u in TOP5_SCHOOLS} if preserve_top_schools else set()

    filtered = [
        s
        for s in schools
        if _should_include_school(s, svc, max_priority, norm_top5, preserve_top_schools)
    ]
    return filtered or schools


def _should_include_school(
    school: dict[str, Any],
    svc: Any,
    max_priority: int,
    top_schools: set[str],
    preserve_top_schools: bool,
) -> bool:
    uni = school.get("university", "")

    if preserve_top_schools and normalize_school_name(uni) in top_schools:
        return True

    info = svc.get_school_info(uni)
    priority = info.get("priority", SCHOOL_LEVEL_PRIORITY.get("未知", 12))
    return priority <= max_priority


def _apply_top8_priority(
    all_schools_data: list[dict[str, Any]], min_schools: int
) -> list[dict[str, Any]]:
    norm_top8 = {normalize_school_name(u) for u in TOP8_SCHOOLS}

    top8_schools = [
        s for s in all_schools_data if normalize_school_name(s.get("university")) in norm_top8
    ]

    if not top8_schools or len(top8_schools) >= max(1, min_schools):
        return top8_schools or all_schools_data

    other_schools = [
        s for s in all_schools_data if normalize_school_name(s.get("university")) not in norm_top8
    ]
    other_schools.sort(key=lambda x: x.get("probability", 0.0), reverse=True)

    return top8_schools + other_schools[: max(0, min_schools - len(top8_schools))]


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

    schools_to_add = _select_additional_safety_schools(
        filtered_schools,
        original_schools,
        min_safety_needed - current_safety_count,
        background_faculty,
        safety_threshold,
    )

    return filtered_schools + schools_to_add


def _select_additional_safety_schools(
    current_schools: list[dict[str, Any]],
    all_schools: list[dict[str, Any]],
    needed_count: int,
    background_faculty: str | None,
    safety_threshold: float,
) -> list[dict[str, Any]]:
    existing_set = {(s.get("university"), s.get("major")) for s in current_schools}

    potential_safety = [
        s
        for s in all_schools
        if s.get("probability", 0.0) >= safety_threshold
        and (s.get("university"), s.get("major")) not in existing_set
    ]

    if not background_faculty:
        return sorted(potential_safety, key=lambda s: s.get("probability", 0.0), reverse=True)[
            :needed_count
        ]

    same_faculty = [s for s in potential_safety if s.get("faculty", "") == background_faculty]
    other_faculty = [s for s in potential_safety if s.get("faculty", "") != background_faculty]

    same_faculty.sort(key=lambda s: s.get("probability", 0.0), reverse=True)
    other_faculty.sort(key=lambda s: s.get("probability", 0.0), reverse=True)

    return (same_faculty + other_faculty)[:needed_count]


def filter_candidates_by_background(
    all_schools_data: list[dict[str, Any]],
    school_level: str | None,
    gpa: float | None,
    min_schools: int = 1,
    background_faculty: str | None = None,
    adaptive_thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    all_schools_data = _filter_by_sample_size(all_schools_data)

    is_high_bg_high_gpa = (
        school_level in {"985", "211", "1-50", "51-100"} and gpa is not None and gpa >= 3.2
    )

    max_allowed_priority = _determine_max_allowed_priority(school_level, gpa)
    filtered_schools = (
        _filter_by_priority(
            all_schools_data,
            max_allowed_priority,
            preserve_top_schools=is_high_bg_high_gpa,
        )
        if max_allowed_priority is not None
        else all_schools_data
    )

    if is_high_bg_high_gpa:
        filtered_schools = _apply_top8_priority(filtered_schools, min_schools)

    safety_threshold = (adaptive_thresholds or SCHOOL_CATEGORY_THRESHOLDS).get(
        "safety", SCHOOL_CATEGORY_THRESHOLDS["safety"]
    )

    return _ensure_safety_schools(
        filtered_schools=filtered_schools,
        original_schools=all_schools_data,
        is_high_bg=is_high_bg_high_gpa,
        background_faculty=background_faculty,
        safety_threshold=safety_threshold,
    )
