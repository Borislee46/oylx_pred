import pandas as pd

from src.adjustment.utils import cache_data, compute_dataframe_hash
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


@cache_data(
    show_spinner=False,
    hash_funcs={pd.DataFrame: compute_dataframe_hash},
)
def get_admitted_combinations_cached(
    cases_df: pd.DataFrame, background_major: str
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

    return get_admitted_combinations_cached(cases_df[required_cols], background_major)


@cache_data(
    show_spinner=False,
    hash_funcs={pd.DataFrame: compute_dataframe_hash},
)
def get_cross_major_stats_cached(
    cases_df: pd.DataFrame, background_major: str
) -> dict[tuple[str, str], dict]:
    bg = str(background_major).strip()
    df = cases_df[["background_major", "target_university", "target_major", "admitted"]]

    total = df.groupby(["target_university", "target_major"], as_index=True)
    total_agg = total.agg(n_total=("admitted", "count"), admitted_total=("admitted", "sum"))

    cross = df[df["background_major"] == bg]
    cross_agg = cross.groupby(["target_university", "target_major"], as_index=True).agg(
        n_cross=("admitted", "count"), admitted_cross=("admitted", "sum")
    )

    merged = total_agg.join(cross_agg, how="left").fillna(0)
    result: dict[tuple[str, str], dict] = {}
    for idx, row in merged.iterrows():
        univ, major = str(idx[0]), str(idx[1])
        result[(univ, major)] = {
            "n_total": int(row["n_total"]),
            "admitted_total": int(row["admitted_total"]),
            "n_cross": int(row["n_cross"]),
            "admitted_cross": int(row["admitted_cross"]),
        }

    return result


def get_cross_major_admission_stats(
    cases_df: pd.DataFrame, background_major: str
) -> dict[tuple[str, str], dict]:
    if cases_df is None or cases_df.empty:
        logger.debug("跨专业录取统计: cases_df为空")
        return {}
    required = ["background_major", "target_university", "target_major", "admitted"]
    if not all(c in cases_df.columns for c in required):
        logger.warning(
            "跨专业录取统计: 缺失必要列 | have=%s need=%s",
            [c for c in cases_df.columns if c in required],
            required,
        )
        return {}
    stats = get_cross_major_stats_cached(cases_df[required], background_major)
    logger.debug(
        "跨专业录取统计 | bg_major=%s n_targets=%d",
        background_major,
        len(stats),
    )
    return stats


@cache_data(
    show_spinner=False,
    hash_funcs={pd.DataFrame: compute_dataframe_hash},
)
def get_baseline_admit_lookup_cached(
    cases_df: pd.DataFrame,
) -> dict[tuple[str, str], tuple[float, int]]:
    grouped = cases_df.groupby(["target_university", "target_major"], as_index=True)[
        "admitted"
    ].agg(["mean", "count"])
    return {
        (str(univ), str(major)): (float(row["mean"]), int(row["count"]))
        for (univ, major), row in grouped.iterrows()
    }


def get_baseline_admit_lookup(
    cases_df: pd.DataFrame,
) -> dict[tuple[str, str], tuple[float, int]]:
    if cases_df is None or cases_df.empty:
        return {}
    required = ["target_university", "target_major", "admitted"]
    if not all(col in cases_df.columns for col in required):
        return {}
    subset = cases_df[required]
    return get_baseline_admit_lookup_cached(subset)
