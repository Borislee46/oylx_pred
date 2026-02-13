import pandas as pd

from src.pages.prediction.result_modifier.streamlit_cache import cache_data
from src.pages.prediction.result_modifier.utils import compute_dataframe_hash
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


@cache_data(show_spinner=False)
def get_admitted_combinations_cached(
    df_hash: str, cases_df: pd.DataFrame, background_major: str
) -> set[tuple[str, str]]:
    bg_major_clean = str(background_major).strip()
    mask = (cases_df["admitted"] == 1) & (cases_df["background_major"] == bg_major_clean)
    admitted = cases_df[mask][["target_university", "target_major"]]

    return set(
        zip(
            admitted["target_university"].astype(str),
            admitted["target_major"].astype(str),
            strict=True,
        )
    )


def get_admitted_combinations_from_dataframe(
    cases_df: pd.DataFrame, background_major: str
) -> set[tuple[str, str]]:
    if cases_df is None or cases_df.empty:
        return set()

    required_cols = ["admitted", "target_university", "target_major", "background_major"]
    if not all(col in cases_df.columns for col in required_cols):
        return set()

    df_hash = compute_dataframe_hash(cases_df[required_cols])
    return get_admitted_combinations_cached(df_hash, cases_df[required_cols], background_major)
