import pandas as pd

from src.adjustment.utils import cache_data, compute_dataframe_hash
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


@cache_data(
    show_spinner=False,
    # df_hash 是权威缓存键；避免 streamlit 对整帧 DataFrame 做 O(n) 内容哈希。
    hash_funcs={pd.DataFrame: compute_dataframe_hash},
)
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


@cache_data(
    show_spinner=False,
    # df_hash 是权威缓存键；避免 streamlit 对整帧 DataFrame 做 O(n) 内容哈希。
    hash_funcs={pd.DataFrame: compute_dataframe_hash},
)
def get_cross_major_stats_cached(
    df_hash: str, cases_df: pd.DataFrame, background_major: str
) -> dict[tuple[str, str], dict]:
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
    df_hash = compute_dataframe_hash(cases_df[required])
    stats = get_cross_major_stats_cached(df_hash, cases_df[required], background_major)
    logger.debug(
        "跨专业录取统计 | bg_major=%s n_targets=%d",
        background_major,
        len(stats),
    )
    return stats


@cache_data(
    show_spinner=False,
    # df_hash 是权威缓存键；避免 streamlit 对整帧 DataFrame 做 O(n) 内容哈希。
    hash_funcs={pd.DataFrame: compute_dataframe_hash},
)
def get_baseline_admit_lookup_cached(
    df_hash: str,
    cases_df: pd.DataFrame,
) -> dict[tuple[str, str], tuple[float, int]]:
    """Full (university, major) → (admit_rate, sample_count) lookup for trace baseline."""
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
    df_hash = compute_dataframe_hash(subset)
    return get_baseline_admit_lookup_cached(df_hash, subset)
