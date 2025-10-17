import os
from collections import Counter, defaultdict

import pandas as pd

SCHOOL_LEVEL_PRIORITY = {
    "985": 1,
    "1-50": 2,
    "51-100": 3,
    "211": 4,
    "101-200": 5,
    "201-300": 6,
    "301-500": 6,
    "500之后": 7,
    "500+": 7,
    "普通本科": 8,
    "专接本": 9,
    "三本/民办本科": 10,
    "专科": 11,
    "未知": 12,
    None: 12,
}


def _load_school_base_data(path: str = None) -> pd.DataFrame:
    if path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "data", "school_base.feather")

    return pd.read_feather(path)


def _build_school_to_level_mapping(school_base_df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}

    if (
        school_base_df.empty
        or "学校名称" not in school_base_df.columns
        or "school_level" not in school_base_df.columns
    ):
        return mapping

    df_local = school_base_df[["学校名称", "school_level"]].copy()
    df_local["学校名称"] = df_local["学校名称"].astype(str).str.strip()
    df_local["school_level"] = df_local["school_level"].astype(str).str.strip()

    mask_unknown = (
        df_local["school_level"].str.lower().isin(["nan", "none", ""])
        | df_local["school_level"].isna()
    )
    df_local.loc[mask_unknown, "school_level"] = "未知"

    for _, row in df_local.iterrows():
        school_name = row["学校名称"]
        level = row["school_level"]
        mapping[school_name] = level

    return mapping


def _count_schools_by_level(
    cases_df: pd.DataFrame, school_to_level: dict[str, str]
) -> dict[str, Counter]:
    level_school_counts: defaultdict[str, Counter] = defaultdict(Counter)

    if "background_university" not in cases_df.columns:
        return level_school_counts

    for school in cases_df["background_university"].dropna():
        school = str(school).strip()
        if not school:
            continue

        level = school_to_level.get(school, "未知")
        level_school_counts[level][school] += 1

    return level_school_counts


def _find_representative_school_for_level(
    level: str, level_school_counts: dict[str, Counter]
) -> str | None:
    if level not in level_school_counts or not level_school_counts[level]:
        return None

    most_common = level_school_counts[level].most_common(1)
    if most_common:
        return most_common[0][0]

    return None


def _find_adjacent_level_fallback(
    target_level: str, level_to_representative: dict[str, str], priority_map: dict[str | None, int]
) -> str | None:
    target_priority = priority_map.get(target_level, priority_map.get("未知", 12))

    available_levels = [
        (level, abs(priority_map.get(level, 12) - target_priority))
        for level in level_to_representative.keys()
    ]

    if not available_levels:
        return None

    available_levels.sort(key=lambda x: x[1])
    nearest_level = available_levels[0][0]

    return level_to_representative.get(nearest_level)


def build_school_level_fallback_mapping(
    cases_df: pd.DataFrame, school_base_path: str = None
) -> dict[str, str]:
    school_base_df = _load_school_base_data(school_base_path)
    if school_base_df.empty:
        return {}

    school_to_level = _build_school_to_level_mapping(school_base_df)

    level_school_counts = _count_schools_by_level(cases_df, school_to_level)

    level_to_representative = {}
    for level in SCHOOL_LEVEL_PRIORITY.keys():
        if level is None:
            continue

        representative = _find_representative_school_for_level(level, level_school_counts)
        if representative:
            level_to_representative[level] = representative

    for level in SCHOOL_LEVEL_PRIORITY.keys():
        if level is None:
            continue

        if level not in level_to_representative:
            fallback_school = _find_adjacent_level_fallback(
                level, level_to_representative, SCHOOL_LEVEL_PRIORITY
            )
            if fallback_school:
                level_to_representative[level] = fallback_school

    return level_to_representative
