import pandas as pd

from src.pages.prediction.result_modifier.streamlit_cache import cache_data
from src.pages.prediction.result_modifier.utils import compute_dataframe_hash


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


@cache_data(show_spinner=False)
def get_cross_major_stats_cached(
    df_hash: str, cases_df: pd.DataFrame, background_major: str
) -> dict[tuple[str, str], dict]:
    """Per-target admission statistics for shrinkage-adjusted cross-major penalty.

    Returns dict keyed by (university, major) with per-target:
      n_total, admitted_total  — all applicants to this target
      n_cross, admitted_cross   — applicants with this background_major
    """
    bg = str(background_major).strip()
    df = cases_df[["background_major", "target_university", "target_major", "admitted"]]

    total = df.groupby(["target_university", "target_major"], as_index=True)
    total_agg = total.agg(n_total=("admitted", "count"), admitted_total=("admitted", "sum"))

    cross = df[df["background_major"] == bg]
    if cross.empty:
        cross_agg = pd.DataFrame(columns=["n_cross", "admitted_cross"])
    else:
        cross_agg = cross.groupby(["target_university", "target_major"], as_index=True).agg(
            n_cross=("admitted", "count"), admitted_cross=("admitted", "sum")
        )

    result: dict[tuple[str, str], dict] = {}
    for idx, t_row in total_agg.iterrows():
        univ, major = str(idx[0]), str(idx[1])
        key = (univ, major)
        c_row = cross_agg.loc[idx] if idx in cross_agg.index else None
        result[key] = {
            "n_total": int(t_row["n_total"]),
            "admitted_total": int(t_row["admitted_total"]),
            "n_cross": int(c_row["n_cross"]) if c_row is not None else 0,
            "admitted_cross": int(c_row["admitted_cross"]) if c_row is not None else 0,
        }

    return result


def get_cross_major_admission_stats(
    cases_df: pd.DataFrame, background_major: str
) -> dict[tuple[str, str], dict]:
    if cases_df is None or cases_df.empty:
        return {}
    required = ["background_major", "target_university", "target_major", "admitted"]
    if not all(c in cases_df.columns for c in required):
        return {}
    df_hash = compute_dataframe_hash(cases_df[required])
    return get_cross_major_stats_cached(df_hash, cases_df[required], background_major)
