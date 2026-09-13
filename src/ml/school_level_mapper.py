from collections import Counter

import pandas as pd

from src.utils.schools.constants import SCHOOL_LEVEL_PRIORITY
from src.utils.schools.level_service import get_school_level_service

_MIN_SAMPLES_FOR_REPRESENTATIVE = 5


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


def _pick_representative_by_admission_median(
    uni_stats: dict[str, tuple[int, float]],
) -> str | None:
    eligible = {
        u: (c, r) for u, (c, r) in uni_stats.items() if c >= _MIN_SAMPLES_FOR_REPRESENTATIVE
    }
    pool = eligible or uni_stats
    if len(pool) < 2:
        return None

    rates = [r for _, r in pool.values()]
    median_rate = sorted(rates)[len(rates) // 2]

    best_uni = None
    best_diff = None
    for uni, (_, rate) in pool.items():
        diff = abs(rate - median_rate)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_uni = uni
    return best_uni


def build_school_level_fallback_mapping(
    cases_df: pd.DataFrame,
) -> tuple[dict[str, str], list[str]]:
    if cases_df is None or cases_df.empty or "background_university" not in cases_df.columns:
        return {}, []

    counts: Counter[str] = Counter()
    for v in cases_df["background_university"].dropna():
        s = str(v).strip()
        if s:
            counts[s] += 1

    if not counts:
        return {}, []

    has_admitted = "admitted" in cases_df.columns
    admission_rates: dict[str, float] = {}
    if has_admitted:
        rates_df = (
            cases_df.dropna(subset=["background_university", "admitted"])
            .assign(bg=lambda d: d["background_university"].astype(str).str.strip())
            .groupby("bg")["admitted"]
            .mean()
        )
        admission_rates = {uni: float(r) for uni, r in rates_df.items() if uni in counts}

    service = get_school_level_service()
    level_to_unis: dict[str, dict[str, tuple[int, float]]] = {}

    for uni, cnt in counts.items():
        level = service.get_school_level(uni)
        rate = admission_rates.get(uni, 0.0)
        level_to_unis.setdefault(level, {})[uni] = (int(cnt), rate)

    level_to_representative: dict[str, str] = {}
    for level, unis in level_to_unis.items():
        rep = _pick_representative_by_admission_median(unis)
        if rep is None:
            rep = max(unis.keys(), key=lambda u: unis[u][0])
        level_to_representative[level] = rep

    cross_level_levels: list[str] = []

    for level in SCHOOL_LEVEL_PRIORITY.keys():
        if level is None:
            continue
        if level not in level_to_representative:
            fallback_school = _find_adjacent_level_fallback(level, level_to_representative)
            if fallback_school:
                level_to_representative[level] = fallback_school
                cross_level_levels.append(level)

    return level_to_representative, cross_level_levels
