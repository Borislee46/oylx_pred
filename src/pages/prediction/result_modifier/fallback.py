# =============================================================================
# 预测失败兜底 (Prediction Fallback)
# ─────────────────────────────────────────────────────────────────────────────
# 当 XGBoost 无法运行时（缺失 GPA+语言 或 缺失关键标识符），
# 用历史数据的录取率做 cascading population fallback。
#
# Fallback 层级（从精确到宽泛）：
#   Level 0: (bg_uni, bg_major, target_uni, target_major) 直接匹配
#   Level 1: (bg_uni, target_uni, target_major)
#   Level 2: (target_uni, target_major)
#   Level 3: (target_uni)
#   Level 4: 全局录取率
#
# 每层降级时 n 阈值 = 5，低于阈值自动降级到下一层。
# 所有比例用 Wilson score interval 给出 95% CI，避免小样本误导。
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class DataCompleteness(str, Enum):
    complete = "complete"       # GPA + Language + 标识符 齐全
    degraded = "degraded"       # 缺一个数值字段，模型可跑但调整链不完整
    minimal = "minimal"         # 缺 GPA 和 Language，XGBoost 不可用
    insufficient = "insufficient"  # 缺背景院校/专业，完全无法预测


FALLBACK_N_THRESHOLD: int = 5
WILSON_Z: float = 1.96  # 95% CI


def wilson_score_ci(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Unlike the normal approximation, Wilson intervals stay within [0, 1]
    and are well-calibrated even for small n.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


@dataclass
class FallbackResult:
    probability: float
    ci_lower: float
    ci_upper: float
    sample_count: int
    fallback_level: int       # 0=精确, 4=全局
    level_description: str    # 人类可读的层级描述


def check_data_completeness(input_data: dict[str, Any]) -> DataCompleteness:
    bg_uni = str(input_data.get("background_university", "")).strip()
    bg_major = str(input_data.get("background_major", "")).strip()
    if not bg_uni or not bg_major:
        return DataCompleteness.insufficient

    has_gpa = input_data.get("gpa") is not None
    has_lang = input_data.get("language_score") is not None

    if has_gpa and has_lang:
        return DataCompleteness.complete
    elif has_gpa or has_lang:
        return DataCompleteness.degraded
    else:
        return DataCompleteness.minimal


def _col_exists(df: pd.DataFrame, col: str | None) -> bool:
    return col is not None and col in df.columns


def _compute_admit_rate(
    df: pd.DataFrame,
    bg_uni_col: str | None,
    bg_uni_val: str | None,
    bg_major_col: str | None,
    bg_major_val: str | None,
    target_uni_col: str,
    target_uni_val: str,
    target_major_col: str | None,
    target_major_val: str | None,
) -> FallbackResult | None:
    if not _col_exists(df, target_uni_col) or "admitted" not in df.columns:
        return None
    mask = df[target_uni_col].astype(str).str.strip() == target_uni_val
    if target_major_col and target_major_val and _col_exists(df, target_major_col):
        mask &= df[target_major_col].astype(str).str.strip() == target_major_val
    if bg_uni_col and bg_uni_val and _col_exists(df, bg_uni_col):
        mask &= df[bg_uni_col].astype(str).str.strip() == bg_uni_val
    if bg_major_col and bg_major_val and _col_exists(df, bg_major_col):
        mask &= df[bg_major_col].astype(str).str.strip() == bg_major_val

    sub = df.loc[mask, "admitted"]
    n = int(sub.count())
    if n < FALLBACK_N_THRESHOLD:
        return None
    k = int(sub.sum())
    lo, hi = wilson_score_ci(k, n)
    return FallbackResult(
        probability=k / n,
        ci_lower=lo,
        ci_upper=hi,
        sample_count=n,
        fallback_level=-1,
        level_description="",
    )


def _resolve_fallback(
    cases_df: pd.DataFrame,
    bg_uni: str,
    bg_major: str,
    target_uni: str,
    target_major: str,
) -> FallbackResult:
    """Cascading fallback: try each granularity level until n >= threshold."""
    bg_uni_col = "background_university"
    bg_major_col = "background_major"
    target_uni_col = "target_university"
    target_major_col = "target_major"

    levels = [
        (0, bg_uni_col, bg_uni, bg_major_col, bg_major,
         target_uni_col, target_uni, target_major_col, target_major,
         f"基于相同背景院校、专业、目标院校、专业的 {FALLBACK_N_THRESHOLD}+ 历史数据"),
        (1, bg_uni_col, bg_uni, None, None,
         target_uni_col, target_uni, target_major_col, target_major,
         f"基于相同背景院校、目标院校、专业的 {FALLBACK_N_THRESHOLD}+ 历史数据"),
        (2, None, None, None, None,
         target_uni_col, target_uni, target_major_col, target_major,
         f"基于目标院校、专业的 {FALLBACK_N_THRESHOLD}+ 历史数据"),
        (3, None, None, None, None,
         target_uni_col, target_uni, None, None,
         f"基于目标院校的 {FALLBACK_N_THRESHOLD}+ 历史数据"),
    ]

    for level, bu_col, bu_val, bm_col, bm_val, tu_col, tu_val, tm_col, tm_val, desc in levels:
        result = _compute_admit_rate(
            cases_df, bu_col, bu_val, bm_col, bm_val, tu_col, tu_val, tm_col, tm_val,
        )
        if result is not None:
            result.fallback_level = level
            result.level_description = desc
            return result

    n_total = int(cases_df["admitted"].count()) if "admitted" in cases_df.columns else 0
    if n_total == 0:
        return FallbackResult(
            probability=0.0, ci_lower=0.0, ci_upper=1.0,
            sample_count=0, fallback_level=4,
            level_description="历史数据不可用，无法估算",
        )
    k_total = int(cases_df["admitted"].sum())
    lo, hi = wilson_score_ci(k_total, n_total)
    return FallbackResult(
        probability=k_total / n_total,
        ci_lower=lo,
        ci_upper=hi,
        sample_count=n_total,
        fallback_level=4,
        level_description=f"基于全部 {n_total:,} 条历史数据的全局录取率",
    )


def compute_fallback_probabilities(
    combinations: list[tuple[str, str]],
    cases_df: pd.DataFrame,
    bg_uni: str,
    bg_major: str,
    similarity_scores: dict[tuple[str, str], float] | None = None,
) -> list[dict[str, Any]]:
    """Generate fallback prediction results for each (target_uni, target_major).

    Returns results in the same structure as normal model predictions,
    with additional `_is_fallback` and `_fallback_*` fields.
    """
    results: list[dict[str, Any]] = []
    if cases_df is None or cases_df.empty:
        return results

    for target_uni, target_major in combinations:
        fb = _resolve_fallback(cases_df, bg_uni, bg_major, target_uni, target_major)
        sim = 1.0
        if similarity_scores:
            sim = similarity_scores.get((target_uni, target_major), 1.0)

        results.append({
            "university": target_uni,
            "major": target_major,
            "probability": fb.probability,
            "similarity": sim,
            "_is_fallback": True,
            "_fallback_level": fb.fallback_level,
            "_fallback_ci_lower": fb.ci_lower,
            "_fallback_ci_upper": fb.ci_upper,
            "_fallback_sample_count": fb.sample_count,
            "_fallback_description": fb.level_description,
        })

    return results
