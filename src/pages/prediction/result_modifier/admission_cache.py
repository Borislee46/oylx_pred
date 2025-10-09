import pandas as pd
import streamlit as st


@st.cache_data
def get_admitted_combinations_for_major(
    cases_df_tuple: tuple, background_major: str
) -> set[tuple[str, str]]:
    cases_df = pd.DataFrame(
        list(cases_df_tuple),
        columns=["admitted", "target_university", "target_major", "background_major"],
    )
    if cases_df.empty:
        return set()

    try:
        bg_major_clean = str(background_major).strip()
        filtered_cases = cases_df[
            (cases_df["admitted"] == 1) & (cases_df["background_major"] == bg_major_clean)
        ]
        return set(tuple(x) for x in filtered_cases[["target_university", "target_major"]].values)
    except Exception:
        return set()


def get_admitted_combinations_from_dataframe(
    cases_df: pd.DataFrame, background_major: str
) -> set[tuple[str, str]]:
    if cases_df is None or cases_df.empty:
        return set()

    required_cols = ["admitted", "target_university", "target_major", "background_major"]
    if not all(col in cases_df.columns for col in required_cols):
        return set()

    cases_df_tuple = tuple(cases_df[required_cols].itertuples(index=False, name=None))
    return get_admitted_combinations_for_major(cases_df_tuple, background_major)
