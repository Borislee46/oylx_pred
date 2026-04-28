from collections import Counter

import pandas as pd

from src.utils.school_constants import SCHOOL_LEVEL_PRIORITY
from src.utils.school_level_service import get_school_level_service


def _find_adjacent_level_fallback(
    target_level: str, level_to_representative: dict[str, str]
) -> str | None:
    target_priority = SCHOOL_LEVEL_PRIORITY.get(target_level, SCHOOL_LEVEL_PRIORITY["未知"])
    best_level = None
    best_diff = None

    for level in level_to_representative.keys():
        diff = abs(
            SCHOOL_LEVEL_PRIORITY.get(level, SCHOOL_LEVEL_PRIORITY["未知"]) - target_priority
        )
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_level = level

    return level_to_representative.get(best_level) if best_level else None


def build_school_level_fallback_mapping(cases_df: pd.DataFrame) -> dict[str, str]:
    if cases_df is None or cases_df.empty or "background_university" not in cases_df.columns:
        return {}

    counts: Counter[str] = Counter()
    for v in cases_df["background_university"].dropna():
        s = str(v).strip()
        if s:
            counts[s] += 1

    if not counts:
        return {}

    service = get_school_level_service()
    level_best: dict[str, tuple[str, int]] = {}

    for uni, cnt in counts.most_common():
        level = service.get_school_level(uni)
        prev = level_best.get(level)
        if prev is None or cnt > prev[1]:
            level_best[level] = (uni, int(cnt))

    level_to_representative = {level: uni for level, (uni, _) in level_best.items()}

    for level in SCHOOL_LEVEL_PRIORITY.keys():
        if level is None:
            continue
        if level not in level_to_representative:
            fallback_school = _find_adjacent_level_fallback(level, level_to_representative)
            if fallback_school:
                level_to_representative[level] = fallback_school

    return level_to_representative
