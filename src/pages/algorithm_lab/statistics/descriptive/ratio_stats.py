"""比率统计 (Ratio Statistics) — 全参数实现

复刻 SPSS RATIO STATISTICS 过程，核心用于 IAAO 批量评估标准:
- 集中趋势: 中位数比率、加权均值、算术均值
- 离散度: AAD, COD (离散系数), PRD (价格相关差异)
- 集中度: 落入指定比率区间的个案百分比
- 行业阈值 (IAAO): COD 5%-15% 良好, >20% 差; PRD 0.98-1.03 理想
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ═══════════════════════════════════════════
# IAAO 行业标准阈值
# ═══════════════════════════════════════════

IAAO_STANDARDS = {
    "COD": {
        "优秀": (0, 10),
        "良好": (10, 15),
        "一般": (15, 20),
        "差": (20, float("inf")),
    },
    "PRD": {
        "累进 (高价高估)": (0, 0.98),
        "理想": (0.98, 1.03),
        "累退 (低价高估)": (1.03, float("inf")),
    },
}


def iaao_cod_grade(cod: float) -> str:
    """返回 COD 的 IAAO 评级"""
    for grade, (lo, hi) in IAAO_STANDARDS["COD"].items():
        if lo <= cod < hi:
            return grade
    return "差"


def iaao_prd_grade(prd: float) -> str:
    """返回 PRD 的 IAAO 评级"""
    for grade, (lo, hi) in IAAO_STANDARDS["PRD"].items():
        if lo <= prd < hi:
            return grade
    return "未知"


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════


@dataclass
class RatioStatsResult:
    """比率统计完整结果"""

    n: int
    # 集中趋势
    median: float
    mean: float
    weighted_mean: float
    # 离散度
    std_dev: float
    range_: float
    min_: float
    max_: float
    aad: float  # Average Absolute Deviation
    cod: float  # Coefficient of Dispersion (%)
    prd: float  # Price-Related Differential
    cov_median: float  # COV (以中位数为中心)
    cov_mean: float  # COV (以均值为中心)
    # CI (Bootstrap 可选)
    ci_median_95: tuple[float, float] | None = None
    ci_mean_95: tuple[float, float] | None = None
    ci_weighted_mean_95: tuple[float, float] | None = None
    # 集中度
    concentration: list[dict] = field(default_factory=list)  # [{bounds, pct_in, n_in}]
    # 按组
    by_group: list[dict] | None = None  # [{group, median, cod, prd, n}]


@dataclass
class RatioConcentration:
    """集中度指标"""

    description: str
    lower_bound: float
    upper_bound: float
    n_in: int
    pct_in: float


# ═══════════════════════════════════════════
# 核心计算
# ═══════════════════════════════════════════


def ratio_statistics(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    group: pd.Series | None = None,
    ci_level: float = 0.95,
    bootstrap_ci: bool = False,
    bootstrap_n: int = 1000,
    random_seed: int | None = 42,
) -> RatioStatsResult:
    """比率统计分析 (复刻 SPSS RATIO STATISTICS + IAAO 标准)。

    Args:
        numerator: 分子 (如评估价值)。
        denominator: 分母 (如实际交易价格)。
        group: 可选分组变量。
        ci_level: 置信区间水平。
        bootstrap_ci: 是否使用 Bootstrap 法计算 CI (推荐)。
        bootstrap_n: Bootstrap 重抽样次数。
        random_seed: 随机种子。

    Returns:
        RatioStatsResult 含全部统计量及 IAAO 合规信息。
    """
    num = numerator.astype(float)
    den = denominator.astype(float)

    # 清洗: 排除分母为 0 或缺失
    mask = (~num.isna()) & (~den.isna()) & (den != 0)
    num = num[mask]
    den = den[mask]

    n = len(num)
    if n == 0:
        raise ValueError("清洗后无有效个案 (分母为 0 或缺失)")

    ratio = num / den

    # 集中趋势
    median = float(ratio.median())
    mean = float(ratio.mean())
    weighted_mean = float(num.sum() / den.sum())

    # 离散度
    std = float(ratio.std(ddof=1)) if n > 1 else float("nan")
    min_ = float(ratio.min())
    max_ = float(ratio.max())
    range_ = max_ - min_

    # AAD & COD
    aad = float((ratio - median).abs().mean())
    cod = (aad / median) * 100 if median != 0 else float("nan")

    # PRD
    prd = mean / weighted_mean if weighted_mean != 0 else float("nan")

    # COV
    cov_median = (std / median) * 100 if median != 0 else float("nan")
    cov_mean = (std / mean) * 100 if mean != 0 else float("nan")

    # Bootstrap CI
    ci_median = ci_mean = ci_wm = None
    if bootstrap_ci and n > 1:
        rng = np.random.default_rng(random_seed)
        meds, means, wms = [], [], []
        for _ in range(bootstrap_n):
            idx = rng.choice(n, size=n, replace=True)
            bs_ratio = ratio.iloc[idx]
            meds.append(bs_ratio.median())
            means.append(bs_ratio.mean())
            wms.append(num.iloc[idx].sum() / den.iloc[idx].sum())
        alpha = 1 - ci_level
        lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
        ci_median = (float(np.percentile(meds, lo)), float(np.percentile(meds, hi)))
        ci_mean = (float(np.percentile(means, lo)), float(np.percentile(means, hi)))
        ci_wm = (float(np.percentile(wms, lo)), float(np.percentile(wms, hi)))

    # 按组
    by_group = None
    if group is not None:
        by_group = []
        df = pd.DataFrame({"num": num, "den": den, "ratio": ratio, "grp": group[mask]})
        for g_label in sorted(df["grp"].unique()):
            gdf = df[df["grp"] == g_label]
            g_med = float(gdf["ratio"].median())
            g_aad = float((gdf["ratio"] - g_med).abs().mean())
            g_cod = (g_aad / g_med) * 100 if g_med != 0 else float("nan")
            g_prd = (
                float(gdf["ratio"].mean() / (gdf["num"].sum() / gdf["den"].sum()))
                if gdf["den"].sum() != 0
                else float("nan")
            )
            by_group.append(
                {
                    "group": str(g_label),
                    "n": len(gdf),
                    "median": g_med,
                    "cod": g_cod,
                    "prd": g_prd,
                    "weighted_mean": float(gdf["num"].sum() / gdf["den"].sum()),
                }
            )

    return RatioStatsResult(
        n=n,
        median=median,
        mean=mean,
        weighted_mean=weighted_mean,
        std_dev=std,
        range_=range_,
        min_=min_,
        max_=max_,
        aad=aad,
        cod=cod,
        prd=prd,
        cov_median=cov_median,
        cov_mean=cov_mean,
        ci_median_95=ci_median,
        ci_mean_95=ci_mean,
        ci_weighted_mean_95=ci_wm,
        concentration=[],
        by_group=by_group,
    )


# ═══════════════════════════════════════════
# 集中度
# ═══════════════════════════════════════════


def concentration_between(
    ratio: pd.Series, low: float, high: float
) -> RatioConcentration:
    """计算落入 [low, high] 比率区间的个案百分比。

    Args:
        ratio: 比率序列。
        low: 下界。
        high: 上界。
    """
    n = len(ratio)
    mask = (ratio >= low) & (ratio <= high)
    n_in = int(mask.sum())
    return RatioConcentration(
        description=f"比率介于 {low:.3f} ~ {high:.3f}",
        lower_bound=low,
        upper_bound=high,
        n_in=n_in,
        pct_in=n_in / n * 100 if n > 0 else 0.0,
    )


def concentration_within_median_pct(
    ratio: pd.Series, pct: float
) -> RatioConcentration:
    """计算落入中位数 ±N% 浮动带的个案百分比。

    Args:
        ratio: 比率序列。
        pct: 浮动百分比 (如 15 → 中位数 ±15%)。
    """
    median = ratio.median()
    low = median * (1 - pct / 100)
    high = median * (1 + pct / 100)
    return concentration_between(ratio, low, high)


# ═══════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════


def ratio_report(result: RatioStatsResult) -> str:
    """生成 IAAO 标准比率统计报告文本。"""
    lines = [
        "=" * 50,
        "比率统计分析报告 (IAAO 标准)",
        "=" * 50,
        f"有效个案数: {result.n}",
        "",
        "【集中趋势】",
        f"  中位数比率     = {result.median:.4f}  (IAAO 推荐首选指标)",
        f"  加权均值       = {result.weighted_mean:.4f}  (∑分子 / ∑分母)",
        f"  算术均值       = {result.mean:.4f}",
        "",
        "【横向公平性 (水平一致性)】",
        f"  AAD           = {result.aad:.4f}",
        f"  COD           = {result.cod:.2f}%",
        f"  COV (中位数)   = {result.cov_median:.2f}%",
        f"  IAAO 评级: {iaao_cod_grade(result.cod)}",
        "",
        "【垂直公平性】",
        f"  PRD           = {result.prd:.4f}",
        f"  IAAO 评级: {iaao_prd_grade(result.prd)}",
    ]

    if "累退" in iaao_prd_grade(result.prd):
        lines.append("  ⚠ PRD > 1.03: 低价资产相对高估, 高价资产相对低估 (累退)")
    elif "累进" in iaao_prd_grade(result.prd):
        lines.append("  ⚠ PRD < 0.98: 高价资产高估 (累进)")

    if result.ci_median_95:
        lines.extend(
            [
                "",
                f"【{int(95)}% Bootstrap CI (n={1000})】",
                f"  中位数比率 CI: [{result.ci_median_95[0]:.4f}, {result.ci_median_95[1]:.4f}]",
                f"  加权均值 CI:   [{result.ci_weighted_mean_95[0]:.4f}, {result.ci_weighted_mean_95[1]:.4f}]",
            ]
        )

    if result.by_group:
        lines.extend(["", "【按组统计】", ""])
        for g in result.by_group:
            lines.append(
                f"  {g['group']}: n={g['n']}, Median={g['median']:.4f}, "
                f"COD={g['cod']:.2f}%, PRD={g['prd']:.4f}"
            )

    lines.append("=" * 50)
    return "\n".join(lines)
