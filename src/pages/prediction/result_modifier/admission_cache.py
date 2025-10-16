import pandas as pd
import streamlit as st


@st.cache_data
def get_admitted_combinations_for_major(
    cases_df_tuple: tuple, background_major: str
) -> set[tuple[str, str]]:
    if not cases_df_tuple:
        return set()

    try:
        bg_major_clean = str(background_major).strip()
        admitted_combinations = set()
        for row in cases_df_tuple:
            if row[0] == 1 and row[3] == bg_major_clean:
                admitted_combinations.add((row[1], row[2]))
        return admitted_combinations
    except Exception:
        return set()


def get_admitted_combinations_from_dataframe(
    cases_df: pd.DataFrame, background_major: str
) -> set[tuple[str, str]]:
    if cases_df is None or cases_df.empty:
        return set()

    required_cols = [
        "admitted",
        "target_university",
        "target_major",
        "background_major",
    ]
    if not all(col in cases_df.columns for col in required_cols):
        return set()

    cases_df_tuple = tuple(cases_df[required_cols].itertuples(index=False, name=None))
    return get_admitted_combinations_for_major(cases_df_tuple, background_major)
