import hashlib
import string
from typing import Any

import pandas as pd
import streamlit as st
from numba import jit

from src.adjustment.config import (
    CROSS_MAJOR_PENALTY_FACTOR,
    CROSS_MAJOR_SIGMOID_MIDPOINT,
    CROSS_MAJOR_SIGMOID_STEEPNESS,
    CROSS_MAJOR_SIMILARITY_MIN,
    SCHOOL_STATS_MIN_N,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability, sigmoid_k

logger = setup_logger("page3", "prediction")


def has_streamlit_runtime() -> bool:
    runtime = getattr(st, "runtime", None)
    exists = getattr(runtime, "exists", None)
    return bool(exists and exists())


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty_df"
    try:
        meta_str = f"{df.shape}-{tuple(map(str, df.columns))}-{tuple(map(str, df.dtypes))}"
        n = len(df)
        if n <= 32:
            idxs = list(range(n))
        else:
            # 32 个均匀采样点 + 尾行，降低“仅未采样行变化”导致缓存命中的概率。
            step = max(1, n // 32)
            idxs = list(range(0, n, step))[:32]
            idxs.append(n - 1)
            idxs = sorted(set(idxs))

        sample = df.iloc[idxs]
        values_hash = pd.util.hash_pandas_object(sample, index=True).to_numpy().tobytes()
        sample_digest = hashlib.md5(values_hash).hexdigest()
        return hashlib.md5(f"{meta_str}-{sample_digest}".encode()).hexdigest()
    except (IndexError, KeyError, TypeError, ValueError, AttributeError):
        return "fallback_hash"


INVALID_TOKENS = {
    "",
    "无",
    "暂无",
    "没有",
    "na",
    "n/a",
    "none",
    "null",
    "-",
    "--",
    "—",
    "/",
    "／",
    ".",
}

PUNCTUATION_CHARS = string.punctuation + "·—-_/／\\|~`'\"，。；：、"


def is_effectively_empty(text: Any) -> bool:
    if not text:
        return True
    t = str(text).strip()
    if not t:
        return True
    tl = t.lower()
    if tl in INVALID_TOKENS:
        return True
    return not tl.strip(PUNCTUATION_CHARS)


def get_probability(case: dict[str, Any], default: float = 0.0) -> float:
    v = case.get("probability", default)
    val = float(v) if v is not None else default
    return clip_probability(val)


# 2026-07-20: get_similarity / deduplicate_results 仅被已关停的 Agent 排名链
# (strategies.py / ranker.py) 使用。保留为薄桩以维持 engine.py 的模块级导入；
# 待 Agent 链整体迁出后一并删除。
def get_similarity(case: dict[str, Any]) -> float:
    v = case.get("similarity", 0.0)
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from src.adjustment.engine import case_key

    seen: set = set()
    deduped: list[dict[str, Any]] = []
    for r in results:
        k = case_key(r)
        if k is not None and k not in seen:
            seen.add(k)
            deduped.append(r)
    return deduped


@jit(nopython=True, cache=True)
def cross_major_penalty_factor_sigmoid(
    similarity: float,
    k: float = CROSS_MAJOR_SIGMOID_STEEPNESS,
    midpoint: float = CROSS_MAJOR_SIGMOID_MIDPOINT,
) -> float:
    if similarity >= 1.0:
        return 1.0
    if similarity <= CROSS_MAJOR_SIMILARITY_MIN:
        return CROSS_MAJOR_PENALTY_FACTOR
    return 0.5 + 0.5 * sigmoid_k(similarity, k, midpoint)


def has_any_experience(experience_details: dict[str, str] | None) -> bool:
    if not experience_details:
        return False
    keys = ("research_details", "award_details", "internship_details", "paper_details")
    return any(not is_effectively_empty(experience_details.get(k, "")) for k in keys)


_SCHOOL_STATS_CACHE: tuple[str, dict[str, dict[str, float]]] | None = None


def _df_fingerprint(df: pd.DataFrame) -> str:
    """轻量 DataFrame 指纹：长度 + 列名 + admitted 汇总 + 关键数值列均值。

    不计入全量行哈希以避免 O(n) 开销。
    """
    cols = tuple(sorted(df.columns))
    try:
        n_admitted = int(df["admitted"].sum()) if "admitted" in df.columns else -1
        n_total = int(df["admitted"].count()) if "admitted" in df.columns else -1
        n_uniq_uni = (
            int(df["target_university"].nunique()) if "target_university" in df.columns else -1
        )
    except (TypeError, ValueError):
        n_admitted = -1
        n_total = -1
        n_uniq_uni = -1

    def _num_mean(col: str) -> str:
        if col not in df.columns:
            return "-"
        try:
            s = df[col]
            if not pd.api.types.is_numeric_dtype(s):
                s = pd.to_numeric(s, errors="coerce")
            s = s.dropna()
            return f"{s.mean():.6f}" if len(s) else "-"
        except (TypeError, ValueError):
            return "-"

    return (
        f"{len(df)}|{cols}|adm={n_admitted}|n={n_total}|uniq={n_uniq_uni}"
        f"|gpa={_num_mean('gpa')}|lang={_num_mean('language_score')}"
        f"|ielts={_num_mean('ielts')}|toefl={_num_mean('toefl')}"
    )


def compute_school_stats(cases_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    global _SCHOOL_STATS_CACHE
    if cases_df is None or cases_df.empty:
        logger.warning("学校统计: cases_df为空，返回空字典")
        return {}

    fp = _df_fingerprint(cases_df)
    if _SCHOOL_STATS_CACHE is not None and _SCHOOL_STATS_CACHE[0] == fp:
        logger.debug(
            "学校统计: 命中缓存 | n_schools=%d fingerprint=%s", len(_SCHOOL_STATS_CACHE[1]), fp
        )
        return _SCHOOL_STATS_CACHE[1]

    df = cases_df
    required = ["target_university", "admitted"]
    if not all(c in df.columns for c in required):
        logger.warning(
            "学校统计: 缺失必要列 | need=%s have=%s",
            required,
            list(df.columns),
        )
        return {}

    schools = df.groupby("target_university").agg(
        admit_rate=("admitted", "mean"),
        n=("admitted", "count"),
    )
    schools = schools[schools["n"] >= SCHOOL_STATS_MIN_N]

    if "gpa" in df.columns:
        gpa_series = df[df["admitted"] == 1].groupby("target_university")["gpa"].mean()
        schools["avg_gpa_admitted"] = gpa_series

    schools["d_admit"] = 1.0 - schools["admit_rate"]
    if "avg_gpa_admitted" in schools.columns:
        gpa_col = schools["avg_gpa_admitted"]
        schools["d_gpa"] = (
            (gpa_col - gpa_col.min()) / (gpa_col.max() - gpa_col.min())
            if gpa_col.max() > gpa_col.min()
            else 0
        )
    else:
        schools["d_gpa"] = 0
    schools["d_raw"] = (schools["d_admit"] + schools["d_gpa"]) / 2
    d_min, d_max = schools["d_raw"].min(), schools["d_raw"].max()
    schools["difficulty"] = (
        (schools["d_raw"] - d_min) / (d_max - d_min) if d_max > d_min else schools["d_raw"]
    )

    result = {}
    for uni, row in schools.iterrows():
        result[str(uni)] = {
            "difficulty": float(row["difficulty"]),
            "admit_rate": float(row["admit_rate"]),
            "n": int(row["n"]),
        }
    _SCHOOL_STATS_CACHE = (fp, result)
    logger.info(
        "学校统计计算完成 | n_schools=%d min_n=%d difficulty_range=[%.2f, %.2f]",
        len(result),
        SCHOOL_STATS_MIN_N,
        min(r["difficulty"] for r in result.values()) if result else 0,
        max(r["difficulty"] for r in result.values()) if result else 0,
    )
    return result


from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def _noop_decorator(func: Callable[P, R]) -> Callable[P, R]:
    return func


def cache_data(*args: Any, **kwargs: Any):
    try:
        import streamlit as st

        if has_streamlit_runtime():
            return st.cache_data(*args, **kwargs)
    except ImportError:
        pass

    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        return _noop_decorator(func)

    return decorator


def cache_resource(*args: Any, **kwargs: Any):
    try:
        import streamlit as st

        if has_streamlit_runtime():
            return st.cache_resource(*args, **kwargs)
    except ImportError:
        pass

    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        return _noop_decorator(func)

    return decorator
