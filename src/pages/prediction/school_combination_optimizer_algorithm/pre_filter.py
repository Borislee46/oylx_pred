import re
from collections import defaultdict
from typing import Any


def _normalize_major_name(major_name: str) -> str:
    return re.sub(r"\s*\(.*\)\s*", "", major_name).strip()


def deduplicate_majors(schools: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def deduplicate_universities_by_similarity(
    schools: list[dict[str, Any]],
    background_major: str,
    similarity_cache: dict,
    target_universities: set[str],
) -> list[dict[str, Any]]:
    if not schools or not background_major or not similarity_cache or not target_universities:
        return schools

    from src.pages.prediction.prediction_utils import get_cached_major_similarity

    grouped_by_uni: dict[str, list[dict[str, Any]]] = defaultdict(list)
    others: list[dict[str, Any]] = []

    for s in schools:
        uni = s.get("university", "")
        if uni in target_universities:
            grouped_by_uni[uni].append(s)
        else:
            others.append(s)

    picked: list[dict[str, Any]] = []
    for uni, items in grouped_by_uni.items():
        if len(items) == 1:
            picked.append(items[0])
            continue

        def score(item: dict[str, Any]) -> tuple[float, float]:
            sim = (
                get_cached_major_similarity(
                    target_major=item.get("major", ""),
                    background_major=background_major,
                    cache=similarity_cache,
                )
                or 0.0
            )
            prob = float(item.get("probability", 0.0) or 0.0)
            return sim, prob

        best = max(items, key=score)
        picked.append(best)

    return others + picked
