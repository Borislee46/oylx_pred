from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.utils.logger import setup_logger
from src.utils.numeric import wilson_score_ci

logger = setup_logger("page3", "prediction")


from src.adjustment.config import (
    BETA_BINOMIAL_PRIOR_STRENGTH,
)


@dataclass
class FallbackResult:
    probability: float
    ci_lower: float
    ci_upper: float
    sample_count: int
    effective_n: float
    fallback_level: int
    level_description: str
    shrinkage_weight: float = 0.0


def _build_fallback_index(cases_df: pd.DataFrame, bg_uni: str, bg_major: str) -> dict[str, Any]:
    """批内一次性预计算：归一化列 + 各级别 (k, n) 计数。

    旧实现为每个组合×每个层级重复 astype(str).str.strip() + boolean mask，
    40 个组合实测 ~2.7s（O(组合×层级×n)）。这里改为一次归一化 + 4 次 groupby，
    组合查询全部 O(1)。

    过滤语义与旧 mask 实现保持一致：仅当列存在且值非空（truthy）时启用该维度，
    比较使用原始值（不做 strip），NaN 经 astype(str) 后不会命中。
    """
    if "admitted" not in cases_df.columns:
        return {"available": False, "n_total": 0, "k_total": 0}

    admitted = cases_df["admitted"]
    n_total = int(admitted.count())
    k_total = int(admitted.sum())

    tu_exists = "target_university" in cases_df.columns
    tm_exists = "target_major" in cases_df.columns
    bu_exists = "background_university" in cases_df.columns
    bm_exists = "background_major" in cases_df.columns

    if not tu_exists:
        return {
            "available": False,
            "n_total": n_total,
            "k_total": k_total,
            "tu_exists": False,
            "tm_exists": tm_exists,
        }

    norm: dict[str, pd.Series] = {"tu": cases_df["target_university"].astype(str).str.strip()}
    if tm_exists:
        norm["tm"] = cases_df["target_major"].astype(str).str.strip()
    if bu_exists:
        norm["bu"] = cases_df["background_university"].astype(str).str.strip()
    if bm_exists:
        norm["bm"] = cases_df["background_major"].astype(str).str.strip()

    frame = pd.DataFrame(norm)
    frame["admitted"] = admitted

    def _counts(rows: pd.DataFrame, keys: list[str]) -> dict[tuple, tuple[int, int]]:
        if rows is None or rows.empty or not keys:
            return {}
        grouped = rows.groupby(keys, sort=False)["admitted"].agg(["sum", "count"])
        out: dict[tuple, tuple[int, int]] = {}
        for key, row in grouped.iterrows():
            k = key if isinstance(key, tuple) else (key,)
            out[k] = (int(row["sum"]), int(row["count"]))
        return out

    combo_keys = ["tu"] if not tm_exists else ["tu", "tm"]

    # L3: 仅目标院校；L2: 目标校+专业
    l3_map = _counts(frame, ["tu"])
    l2_map = _counts(frame, combo_keys)

    # L1: 额外限制背景院校（bg_uni 是本次调用的常量）
    rows1 = frame
    if bu_exists and bool(bg_uni):
        rows1 = frame[frame["bu"] == bg_uni]
    l1_map = _counts(rows1, combo_keys)

    # L0: 再限制背景专业
    rows0 = rows1
    if bm_exists and bool(bg_major):
        rows0 = rows1[rows1["bm"] == bg_major]
    l0_map = _counts(rows0, combo_keys)

    return {
        "available": True,
        "n_total": n_total,
        "k_total": k_total,
        "tu_exists": tu_exists,
        "tm_exists": tm_exists,
        "l0": l0_map,
        "l1": l1_map,
        "l2": l2_map,
        "l3": l3_map,
    }


def _resolve_fallback(
    index: dict[str, Any],
    bg_uni: str,
    bg_major: str,
    target_uni: str,
    target_major: str,
    m: float = BETA_BINOMIAL_PRIOR_STRENGTH,
) -> FallbackResult:
    logger.debug(
        "Fallback 层次收缩开始 | bg_uni=%s bg_major=%s target_uni=%s target_major=%s m=%.1f",
        bg_uni,
        bg_major,
        target_uni,
        target_major,
        m,
    )
    n_total = index["n_total"]
    if n_total == 0:
        return FallbackResult(
            probability=0.0,
            ci_lower=0.0,
            ci_upper=1.0,
            sample_count=0,
            effective_n=0.0,
            fallback_level=4,
            level_description="历史数据不可用，无法估算",
        )

    k_total = index["k_total"]
    p_global = k_total / n_total
    available = bool(index.get("available"))

    level_specs = [
        (0, "(bg_uni, bg_major, target_uni, target_major) 精确匹配"),
        (1, "(bg_uni, target_uni, target_major) 三级匹配"),
        (2, "(target_uni, target_major) 目标校+专业"),
        (3, f"({target_uni}) 仅目标院校"),
    ]

    combo_key: tuple = (target_uni,) if not index.get("tm_exists") else (target_uni, target_major)
    raw_counts = [
        index.get("l0", {}).get(combo_key, (0, 0)),
        index.get("l1", {}).get(combo_key, (0, 0)),
        index.get("l2", {}).get(combo_key, (0, 0)),
        index.get("l3", {}).get((target_uni,), (0, 0)),
    ]

    prior_p = p_global
    shrunk_results: list[dict] = []
    for (level, desc), (k, n) in zip(reversed(level_specs), reversed(raw_counts), strict=True):
        if available and n > 0:
            shrunk_k = float(k) + m * prior_p
            shrunk_n = float(n) + m
            shrunk_p = shrunk_k / shrunk_n
            shrinkage_weight = m / (n + m)
            raw_k, raw_n, raw_p = k, n, k / n
        else:
            shrunk_p = prior_p
            shrunk_k = m * prior_p
            shrunk_n = m
            shrinkage_weight = 1.0
            raw_k, raw_n, raw_p = 0, 0, prior_p

        shrunk_results.append(
            {
                "level": level,
                "raw_k": raw_k,
                "raw_n": raw_n,
                "raw_p": raw_p,
                "shrunk_p": shrunk_p,
                "effective_n": shrunk_n,
                "shrinkage_weight": shrinkage_weight,
                "desc": desc,
            }
        )
        prior_p = shrunk_p

    finest = shrunk_results[-1]

    eff_k = finest["shrunk_p"] * finest["effective_n"]
    eff_n = finest["effective_n"]
    lo, hi = wilson_score_ci(eff_k, eff_n)

    finest_with_data = None
    for entry in reversed(shrunk_results):
        if entry["raw_n"] > 0:
            finest_with_data = entry
            break
    if finest_with_data is None:
        finest_with_data = {
            "level": 4,
            "raw_n": n_total,
            "desc": f"全部 {n_total:,} 条历史数据",
        }

    if finest_with_data["level"] == 0:
        desc = f"基于 {finest_with_data['raw_n']} 条精确匹配历史数据的层次收缩估计"
    elif finest_with_data["level"] <= 3:
        desc = f"基于 {finest_with_data['desc']} ({finest_with_data['raw_n']} 条) 的层次收缩估计"
    else:
        desc = f"基于全部 {n_total:,} 条历史数据的全局录取率"

    result = FallbackResult(
        probability=round(finest["shrunk_p"], 6),
        ci_lower=round(lo, 6),
        ci_upper=round(hi, 6),
        sample_count=finest_with_data["raw_n"],
        effective_n=round(finest["effective_n"], 1),
        fallback_level=finest_with_data["level"],
        level_description=desc,
        shrinkage_weight=round(finest["shrinkage_weight"], 4),
    )
    logger.info(
        "Fallback 层次收缩完成 | prob=%.4f CI=[%.4f, %.4f] level=%d n=%d eff_n=%.1f "
        "shrinkage=%.2f%% | %s",
        result.probability,
        result.ci_lower,
        result.ci_upper,
        result.fallback_level,
        result.sample_count,
        result.effective_n,
        result.shrinkage_weight * 100,
        desc,
    )
    return result


def compute_fallback_probabilities(
    combinations: list[tuple[str, str]],
    cases_df: pd.DataFrame,
    bg_uni: str,
    bg_major: str,
    similarity_scores: dict[tuple[str, str], float] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if cases_df is None or cases_df.empty:
        logger.warning("Fallback 概率计算: cases_df 为空，无法生成兜底结果")
        return results

    logger.info(
        "Fallback 概率批量计算开始 | n_combinations=%d bg_uni=%s bg_major=%s",
        len(combinations),
        bg_uni,
        bg_major,
    )
    index = _build_fallback_index(cases_df, bg_uni, bg_major)
    for target_uni, target_major in combinations:
        fb = _resolve_fallback(index, bg_uni, bg_major, target_uni, target_major)
        sim = 1.0
        if similarity_scores:
            sim = similarity_scores.get((target_uni, target_major), 1.0)

        results.append(
            {
                "university": target_uni,
                "major": target_major,
                "probability": fb.probability,
                "similarity": sim,
                "_is_fallback": True,
                "_fallback_level": fb.fallback_level,
                "_fallback_ci_lower": fb.ci_lower,
                "_fallback_ci_upper": fb.ci_upper,
                "_fallback_sample_count": fb.sample_count,
                "_fallback_effective_n": fb.effective_n,
                "_fallback_shrinkage_weight": fb.shrinkage_weight,
                "_fallback_description": fb.level_description,
            }
        )

    avg_prob = sum(r["probability"] for r in results) / len(results) if results else 0.0
    logger.info(
        "Fallback 概率批量计算完成 | n_results=%d avg_prob=%.4f",
        len(results),
        avg_prob,
    )
    return results
