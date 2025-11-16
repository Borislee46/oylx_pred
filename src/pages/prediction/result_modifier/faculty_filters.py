from typing import Any

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


def filter_schools_by_faculty_rules(
    schools: list[dict[str, Any]],
    background_faculty: str | None,
) -> list[dict[str, Any]]:
    if not schools or not background_faculty:
        return schools

    allowed_faculties = get_allowed_target_faculties(background_faculty)
    if not allowed_faculties:
        return schools

    return [
        school
        for school in schools
        if not (faculty := school.get("faculty", "").strip()) or faculty in allowed_faculties
    ]

