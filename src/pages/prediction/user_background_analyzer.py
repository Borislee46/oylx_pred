import pandas as pd
import streamlit as st

from src.utils.school_level_service import get_school_level_service


@st.cache_data(ttl=1800, show_spinner=False)
def _get_substitution_map(cases_df: pd.DataFrame):
    if cases_df is None or cases_df.empty:
        return {}, None

    if "background_university" not in cases_df.columns:
        return {}, None

    counts = cases_df["background_university"].dropna().astype(str).str.strip()
    counts = counts[counts != ""].value_counts()
    fallback_uni = str(counts.index[0]) if not counts.empty else None
    if counts.empty:
        return {}, fallback_uni

    service = get_school_level_service()

    level_best: dict[str, tuple[str, int]] = {}
    for uni, cnt in counts.items():
        level = service.get_school_level(uni)
        prev = level_best.get(level)
        if prev is None or cnt > prev[1]:
            level_best[level] = (uni, int(cnt))

    return {level: uni for level, (uni, _) in level_best.items()}, fallback_uni


def find_substitute_university(selected_uni: str, cases_df: pd.DataFrame) -> str | None:
    level_to_substitute, fallback_uni = _get_substitution_map(cases_df)

    service = get_school_level_service()
    selected_uni_level = service.get_school_level(selected_uni)

    return level_to_substitute.get(selected_uni_level, fallback_uni)
