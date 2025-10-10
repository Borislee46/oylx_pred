import re
from collections import defaultdict
from typing import Any


def _normalize_major_name(major_name: str) -> str:
    return re.sub(r'\s*\(.*\)\s*', '', major_name).strip()


def deduplicate_majors(
    schools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not schools:
        return []

    grouped_schools = defaultdict(list)
    for school in schools:
        university = school.get("university")
        major = school.get("major")
        if not university or not major:
            continue

        normalized_major = _normalize_major_name(major)
        group_key = (university, normalized_major)
        grouped_schools[group_key].append(school)

    final_schools = []
    for school_group in grouped_schools.values():
        if len(school_group) == 1:
            final_schools.append(school_group[0])
        else:
            best_school = max(school_group, key=lambda s: s.get("probability", 0.0))
            final_schools.append(best_school)

    return final_schools
