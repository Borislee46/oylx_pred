import pandas as pd
import streamlit as st

from src.utils.school_level_service import get_school_level_service


def get_school_level_for_analyzer(background_uni_name: str) -> str:
    service = get_school_level_service()
    return service.get_school_level(background_uni_name)


@st.cache_data(ttl=1800, show_spinner=False)
def _get_cases_df_fingerprint(cases_df: pd.DataFrame) -> int:
    try:
        from pandas.util import hash_pandas_object

        return int(hash_pandas_object(cases_df[["background_university"]]).sum())
    except Exception:
        return len(cases_df)


@st.cache_data(ttl=1800, show_spinner=False)
def _get_substitution_map(fingerprint: int):
    from src.utils.app_data_loader import load_raw_cases_data

    cases_df = load_raw_cases_data()
    if cases_df is None or cases_df.empty:
        return {}, None

    most_frequent_uni_overall = cases_df["background_university"].mode()
    fallback_uni = (
        most_frequent_uni_overall.iloc[0] if not most_frequent_uni_overall.empty else None
    )

    all_unis_in_cases = cases_df["background_university"].unique()
    uni_level_map = {uni: get_school_level_for_analyzer(uni) for uni in all_unis_in_cases}

    uni_level_df = pd.DataFrame(
        list(uni_level_map.items()), columns=["background_university", "level"]
    ).dropna()

    if uni_level_df.empty:
        return {}, fallback_uni

    uni_counts = cases_df["background_university"].value_counts().reset_index()
    uni_counts.columns = ["background_university", "count"]

    merged_df = pd.merge(uni_counts, uni_level_df, on="background_university")

    if merged_df.empty:
        return {}, fallback_uni

    idx = merged_df.groupby("level")["count"].idxmax()
    most_frequent_unis = merged_df.loc[idx]

    level_to_substitute_map = pd.Series(
        most_frequent_unis.background_university.values, index=most_frequent_unis.level
    ).to_dict()

    return level_to_substitute_map, fallback_uni


def find_substitute_university(selected_uni: str, cases_df: pd.DataFrame) -> str | None:
    fingerprint = _get_cases_df_fingerprint(cases_df)
    level_to_substitute, fallback_uni = _get_substitution_map(fingerprint)

    selected_uni_level = get_school_level_for_analyzer(selected_uni)

    if selected_uni_level and selected_uni_level in level_to_substitute:
        return level_to_substitute[selected_uni_level]

    return fallback_uni
