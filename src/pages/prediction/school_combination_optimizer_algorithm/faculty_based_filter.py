from typing import Any

from src.pages.prediction.school_combination_optimizer_algorithm.faculty_rules import (
    get_allowed_target_faculties,
)


def filter_schools_by_faculty_rules(
    schools: list[dict[str, Any]],
    background_faculty: str | None,
    major_category_cache: dict[str, str] | None,
) -> list[dict[str, Any]]:
    if not schools or not background_faculty or not major_category_cache:
        return schools

    allowed_faculties = get_allowed_target_faculties(background_faculty)

    if not allowed_faculties:
        return schools

    filtered_schools: list[dict[str, Any]] = []
    for school in schools:
        university = school.get("university", "")
        major = school.get("major", "")

        cache_key = f"{university}|{major}"
        target_faculty = major_category_cache.get(cache_key)

        if not target_faculty:
            filtered_schools.append(school)
            continue

        if target_faculty in allowed_faculties:
            filtered_schools.append(school)

    return filtered_schools
