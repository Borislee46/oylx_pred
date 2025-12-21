from typing import Any

from src.pages.prediction.result_modifier.config import FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR
from src.pages.prediction.result_modifier.utils import clip_probability

CROSS_FACULTY_RULES: dict[str, set[str]] = {
    "文学院": {"文学院", "社会科学院", "教育学院", "商学院", "艺术学院"},
    "社会科学院": {
        "社会科学院",
        "文学院",
        "商学院",
        "教育学院",
        "艺术学院",
        "建筑学院",
    },
    "法学院": {"法学院"},
    "教育学院": {"教育学院", "文学院", "社会科学院"},
    "商学院": {"商学院", "社会科学院", "文学院"},
    "理学院": {
        "理学院",
        "工程学院",
        "商学院",
        "经济金融学院",
        "科学学院",
        "计算机学院",
    },
    "工程学院": {
        "工程学院",
        "理学院",
        "商学院",
        "计算机学院",
        "建筑学院",
        "设计学院",
        "科学学院",
    },
    "计算机学院": {"计算机学院", "工程学院", "理学院", "商学院"},
    "艺术学院": {"艺术学院", "社会科学院", "文学院", "设计学院", "建筑学院"},
    "医学院": {"医学院"},
    "建筑学院": {"建筑学院", "工程学院", "设计学院", "艺术学院"},
    "设计学院": {"设计学院", "艺术学院", "建筑学院", "社会科学院"},
}


def get_allowed_target_faculties(background_faculty: str | None) -> set[str]:
    if not background_faculty:
        return set()
    return CROSS_FACULTY_RULES.get(background_faculty, set())


def filter_schools_by_allowed_faculties(
    schools: list[dict[str, Any]], allowed_faculties: set[str]
) -> list[dict[str, Any]]:
    if not schools or not allowed_faculties:
        return schools
    return [
        school
        for school in schools
        if not (faculty := school.get("faculty", "").strip()) or faculty in allowed_faculties
    ]


def get_allowed_target_faculties_from_background_faculties(
    background_faculties: list[str] | None, max_allowed: int = 6
) -> set[str]:
    if not background_faculties:
        return set()

    allowed: set[str] = set()
    for bg in background_faculties:
        if not bg:
            continue
        allowed |= CROSS_FACULTY_RULES.get(bg, {bg})
        if max_allowed > 0 and len(allowed) >= max_allowed:
            break

    if max_allowed > 0 and len(allowed) > max_allowed:
        return set(list(allowed)[:max_allowed])
    return allowed


def filter_schools_by_faculty_rules(
    schools: list[dict[str, Any]],
    background_faculty: str | None,
) -> list[dict[str, Any]]:
    if not schools or not background_faculty:
        return schools

    allowed_faculties = get_allowed_target_faculties(background_faculty)
    if not allowed_faculties:
        return schools

    return filter_schools_by_allowed_faculties(schools, allowed_faculties)


def apply_out_of_scope_faculty_penalty(
    schools: list[dict[str, Any]],
    background_faculty: str | None,
    factor: float = FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR,
) -> list[dict[str, Any]]:
    if not schools or not background_faculty:
        return schools

    allowed_faculties = get_allowed_target_faculties(background_faculty)
    if not allowed_faculties:
        return schools

    adjusted: list[dict[str, Any]] = []
    for s in schools:
        if not isinstance(s, dict):
            continue
        faculty = str(s.get("faculty", "")).strip()
        if faculty and faculty not in allowed_faculties:
            prob = s.get("probability", 0.0)
            adjusted_prob = clip_probability(prob) * factor
            s = s.copy()
            s["probability"] = clip_probability(adjusted_prob)
        adjusted.append(s)
    return adjusted
