import pandas as pd
import streamlit as st

from src.utils.school_level_service import get_school_level_service


@st.cache_data(ttl=1800)
def _get_substitution_map(cases_df: pd.DataFrame):
    if cases_df is None or cases_df.empty:
        return {}, None

    fallback_uni = (
        cases_df["background_university"].mode().iloc[0]
        if not cases_df["background_university"].mode().empty
        else None
    )

    service = get_school_level_service()
    all_unis = cases_df["background_university"].unique()
    uni_level_map = {uni: service.get_school_level(uni) for uni in all_unis}

    uni_counts = cases_df["background_university"].value_counts().reset_index()
    uni_counts.columns = ["background_university", "count"]

    uni_level_df = pd.DataFrame(
        list(uni_level_map.items()), columns=["background_university", "level"]
    ).dropna()

    merged_df = pd.merge(uni_counts, uni_level_df, on="background_university")

    if merged_df.empty:
        return {}, fallback_uni

    most_frequent_unis = merged_df.loc[merged_df.groupby("level")["count"].idxmax()]

    level_to_substitute_map = pd.Series(
        most_frequent_unis.background_university.values, index=most_frequent_unis.level
    ).to_dict()

    return level_to_substitute_map, fallback_uni


def find_substitute_university(selected_uni: str, cases_df: pd.DataFrame) -> str | None:
    level_to_substitute, fallback_uni = _get_substitution_map(cases_df)

    service = get_school_level_service()
    selected_uni_level = service.get_school_level(selected_uni)

    return level_to_substitute.get(selected_uni_level, fallback_uni)
