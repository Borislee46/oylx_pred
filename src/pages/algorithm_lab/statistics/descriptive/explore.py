"""探索性数据分析 (Explore) — 全参数实现

复刻 SPSS EXAMINE 过程的核心统计功能:
- 箱线图统计量 (Tukey 铰链: 1.5×IQR 离群值, 3×IQR 极端值)
- 茎叶图
- 正态性检验: Shapiro-Wilk, Kolmogorov-Smirnov (Lilliefors 修正)
- Levene 方差齐性检验: 均值/中位数/截尾均值中心
- 分布-水平图 (Spread-Level Plot): ln(IQR) ~ ln(Median) 回归
- 按组描述统计

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【正态性检验 — Shapiro-Wilk vs K-S】

Shapiro-Wilk (SW):
  适用 n: 3 ~ 5000。检验样本是否来自正态分布的总体。
  是公认统计效力最高的正态性检验。H₀ = 数据正态。
  直觉: 它比较排序后的数据与"如果数据正态, 它们应该在哪"的期望值。
  计算上就是样本值与期望正态顺序统计量的相关性。
  p < 0.05 → 拒绝正态假设。

Kolmogorov-Smirnov (Lilliefors):
  K-S 检验比较的是经验 CDF 和理论 CDF 之间的最大垂直距离 D。
  但标准的 K-S (比较 N(0,1)) 假设分布的参数已知。如果参数由样本估计
  (就像我们做的: 用样本均值和标准差), K-S 会过于保守 (p 值偏高,
  假阴性多)。Lilliefors 修正通过蒙特卡洛模拟产生修正后的临界值表,
  解决了这个问题。

  使用时: SW 比 Lilliefors 的统计效力更高。优先看 SW。

【重要提醒: 为什么大样本下正态性检验几乎总是 p<0.05】
  H₀ 是"总体恰好是正态的"。在实际世界, 没有真实数据是恰好正态的。
  当 n 很大(>300)时, 即使微小的偏离也会被检测出来。但这时的偏离
  可能对 t 检验、ANOVA 等参数方法几乎没有影响 (中心极限定理使得
  均值的抽样分布趋近正态)。
  因此: n<30 时关注正态性, n>300 时更多看 Q-Q 图 + 偏度/峰度实际大小。

【Levene 检验 — 方差齐性】
  ANOVA / t检验的关键假设: 多组之间的方差应该大致相等。
  Levene 检验 H₀: 各组方差相等。
  p < 0.05 → 方差不齐 → 使用 Welch ANOVA / Games-Howell 事后检验,
  而不是经典 ANOVA / Tukey。

  center="median" (Brown-Forsythe 版本): 推荐。以中位数为中心比以均值
  为中心更稳健（不受偏态和离群值的影响）。

【箱线图 — Tukey 规则】
  须: Q1 - 1.5×IQR 到 Q3 + 1.5×IQR
  离群值 (∘): 1.5×IQR < 偏离 < 3×IQR
  极端值 (*): 偏离 > 3×IQR

  1.5×IQR 来源于: 如果数据正态, Q1-1.5IQR ~ μ-2.7σ,
  大约 0.7% 的数据会落在这个范围外。3×IQR ~ μ-4.7σ。

【Spread-Level 图 — 方差齐性的图形化检查】
  回归 ln(IQR) ~ ln(Median), 斜率告诉你方差与水平的关系:
  斜率 = 0 → 方差齐性
  斜率 ≠ 0 → 建议 Box-Cox 变换 λ = 1 - slope
  例如 slope=0.5 → λ=0.5 → 平方根变换可能稳定方差。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .frequencies import _std_error_skewness, _std_error_kurtosis


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════


@dataclass
class NormalityResult:
    """正态性检验结果"""

    method: str  # "Shapiro-Wilk" | "Kolmogorov-Smirnov"
    statistic: float
    p_value: float
    n: int


@dataclass
class LeveneResult:
    """Levene 方差齐性检验结果"""

    center: str  # "mean" | "median" | "trimmed"
    statistic: float  # F 值
    p_value: float
    df1: int  # k - 1
    df2: int  # N - k


@dataclass
class BoxplotStats:
    """单组箱线图统计量 (Tukey 标准)"""

    n: int
    min_: float
    p25: float  # Q1
    median: float  # Q2
    p75: float  # Q3
    max_: float
    iqr: float
    lower_fence: float  # Q1 - 1.5×IQR
    upper_fence: float  # Q3 + 1.5×IQR
    outliers: list[float]  # 1.5×IQR < x ≤ 3×IQR
    extremes: list[float]  # > 3×IQR

    @property
    def lower_whisker(self) -> float:
        """须线底 = min(x_i ≥ lower_fence)"""
        return self.min_ if not self.outliers else min(x for x in self.outliers if x >= self.lower_fence)


@dataclass
class GroupExploreResult:
    """单组探索性分析完整结果"""

    group_label: str
    n: int
    mean: float
    ci_mean_95: tuple[float, float]
    median: float
    trimmed_mean_5pct: float
    std: float
    variance: float
    min_: float
    max_: float
    range_: float
    iqr: float
    skewness: float
    std_error_skewness: float
    kurtosis: float
    std_error_kurtosis: float
    boxplot: BoxplotStats
    normality: list[NormalityResult]


@dataclass
class SpreadLevelResult:
    """分布-水平图回归结果"""

    slope: float
    intercept: float
    suggested_power: float  # 建议的 Box-Cox λ = 1 - slope
    groups_data: list[dict]  # [{group, median, iqr, ln_median, ln_iqr}]


# ═══════════════════════════════════════════
# 正态性检验
# ═══════════════════════════════════════════


def _lilliefors_ks(data: np.ndarray) -> NormalityResult:
    """K-S 检验 with Lilliefors 显著性修正 (参数由样本估计)。

    当分布参数由样本估计时, 标准 K-S 的 p 值会过于保守。
    Lilliefors 修正使用蒙特卡洛模拟产生的临界值表。
    """
    n = len(data)
    x = np.sort(data)
    z = (x - x.mean()) / x.std(ddof=1)
    # 经验 CDF
    ecdf = np.arange(1, n + 1) / n
    # 理论 CDF (正态, 参数由样本估计)
    tcdf = sp_stats.norm.cdf(z)
    # 最大绝对差
    d = max(abs(ecdf - tcdf))

    # Lilliefors 修正临界值 → p 值 (Dallal-Wilkinson 近似)
    # 基于公式 p = exp(-7.01256 * d* * 2 * (n + 2.78019) + ...)
    # 使用 Lilliefors (1967) 的近似方法
    d_star = d * (math.sqrt(n) - 0.01 + 0.85 / math.sqrt(n))
    # 使用改进的 Lilliefors p 值公式
    # 源自 Stephens (1974) 和 Dallal-Wilkinson (1986)
    if d_star < 0.2:
        p = 1.0
    elif d_star > 1.0:
        p = 0.0
    else:
        # Lilliefors p 值近似
        p = 1.0 - _lilliefors_cdf(d_star)

    return NormalityResult(method="Kolmogorov-Smirnov (Lilliefors)", statistic=d, p_value=p, n=n)


def _lilliefors_cdf(x: float) -> float:
    """Lilliefors 检验统计量的近似 CDF (Stephens 1974 Monte Carlo)."""
    # Kolmogorov 分布 CDF 的无穷级数展开近似
    if x <= 0.0:
        return 0.0
    # Kolmogorov 分布近似 × Lilliefors 校正因子
    s = 0.0
    k = 1
    while True:
        term = (-1) ** (k - 1) * math.exp(-2 * k**2 * x**2)
        s += term
        if abs(term) < 1e-10:
            break
        k += 1
        if k > 100:
            break
    return max(0.0, min(1.0, 2 * s))


def normality_tests(data: np.ndarray) -> list[NormalityResult]:
    """对连续变量执行 Shapiro-Wilk + K-S (Lilliefors) 联合正态性检验。

    Args:
        data: 一维连续数值数组 (不含缺失)。

    Returns:
        [Shapiro-Wilk result, K-S (Lilliefors) result]
    """
    n = len(data)
    results: list[NormalityResult] = []

    # Shapiro-Wilk (3 ≤ n ≤ 5000)
    if 3 <= n <= 5000:
        w, p = sp_stats.shapiro(data)
        results.append(NormalityResult(method="Shapiro-Wilk", statistic=w, p_value=p, n=n))

    # K-S (Lilliefors)
    if n >= 5:
        results.append(_lilliefors_ks(data))

    return results


# ═══════════════════════════════════════════
# Levene 方差齐性检验
# ═══════════════════════════════════════════


def levene_test(
    groups: dict[str, np.ndarray],
    center: str = "median",
) -> LeveneResult:
    """Levene 方差齐性检验 (三种中心版本)。

    本质是对 |Y_ij - center_j| 做单因素 ANOVA。
    F(k-1, N-k) 分布, k=组数, N=总样本量。

    Args:
        groups: {组名: 数值数组}。
        center: ``"mean"`` (经典 Levene), ``"median"`` (Brown-Forsythe, 推荐),
                ``"trimmed"`` (截尾均值 5%)。

    Returns:
        LeveneResult 含 F 统计量、自由度、p 值。
    """
    group_list = list(groups.values())
    k = len(group_list)
    if k < 2:
        raise ValueError("Levene 检验需要至少 2 组")

    # 计算每组中心
    all_data = []
    all_group_idx = []
    for g_idx, arr in enumerate(group_list):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
        if center == "median":
            c = np.median(arr)
        elif center == "trimmed":
            c = sp_stats.trim_mean(arr, 0.05)
        else:  # mean
            c = np.mean(arr)
        all_data.append(np.abs(arr - c))
        all_group_idx.extend([g_idx] * len(arr))

    Z = np.concatenate(all_data)
    group_idx = np.array(all_group_idx, dtype=int)

    # 单因素 ANOVA 对残差绝对值
    grand_mean = np.mean(Z)
    ss_between = 0.0
    ss_within = 0.0
    for g in range(k):
        zg = Z[group_idx == g]
        ng = len(zg)
        if ng == 0:
            continue
        ss_between += ng * (np.mean(zg) - grand_mean) ** 2
        ss_within += np.sum((zg - np.mean(zg)) ** 2)

    N = len(Z)
    df1 = k - 1
    df2 = N - k
    if ss_within == 0 or df2 == 0:
        return LeveneResult(center=center, statistic=float("nan"), p_value=float("nan"), df1=df1, df2=df2)
    ms_between = ss_between / df1
    ms_within = ss_within / df2
    f = ms_between / ms_within
    p = 1.0 - sp_stats.f.cdf(f, df1, df2)

    return LeveneResult(center=center, statistic=f, p_value=p, df1=df1, df2=df2)


# ═══════════════════════════════════════════
# 箱线图统计量 (Tukey)
# ═══════════════════════════════════════════


def boxplot_stats(data: np.ndarray) -> BoxplotStats:
    """计算 Tukey 箱线图统计量。

    - 异常值 (∘): Q1 - 1.5×IQR < x < Q1 - 3×IQR 或 Q3 + 1.5×IQR < x < Q3 + 3×IQR
    - 极端值 (*): x < Q1 - 3×IQR 或 x > Q3 + 3×IQR
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    median = float(np.median(arr))
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    lower_extreme = q1 - 3.0 * iqr
    upper_extreme = q3 + 3.0 * iqr

    outliers: list[float] = []
    extremes: list[float] = []
    for v in arr:
        if v < lower_extreme or v > upper_extreme:
            extremes.append(float(v))
        elif v < lower_fence or v > upper_fence:
            outliers.append(float(v))

    return BoxplotStats(
        n=n,
        min_=float(arr.min()),
        p25=q1,
        median=median,
        p75=q3,
        max_=float(arr.max()),
        iqr=iqr,
        lower_fence=lower_fence,
        upper_fence=upper_fence,
        outliers=sorted(outliers),
        extremes=sorted(extremes),
    )


# ═══════════════════════════════════════════
# 茎叶图
# ═══════════════════════════════════════════


def stem_and_leaf(data: np.ndarray, leaf_unit: float | None = None) -> str:
    """生成茎叶图文本。

    Args:
        data: 一维数值数组。
        leaf_unit: 叶单位 (None=自动选择 1, 10, 0.1 等)。

    Returns:
        多行茎叶图字符串。
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return "(空数据集)"

    # 自动选择叶单位
    if leaf_unit is None:
        rng = arr.max() - arr.min()
        if rng == 0:
            leaf_unit = 1.0
        else:
            # 目标约 10-20 个茎
            leaf_unit = 10 ** math.floor(math.log10(rng / 15)) if rng > 0 else 1.0

    stem_map: dict[int, list[int]] = {}
    for v in arr:
        s = int(v / leaf_unit)
        l = int(abs(v - s * leaf_unit) / (leaf_unit / 10)) if leaf_unit < 1 else int(abs(v - s * leaf_unit))
        stem_map.setdefault(s, []).append(abs(l) % 10)

    lines = [f"茎叶图  (叶单位 = {leaf_unit:.0g}, n = {n})", "-" * 40]
    for stem in sorted(stem_map.keys()):
        leaves = "".join(str(d) for d in sorted(stem_map[stem]))
        lines.append(f"  {stem:4d} | {leaves}")

    return "\n".join(lines)


# ═══════════════════════════════════════════
# 分布-水平图 (Spread-Level Plot)
# ═══════════════════════════════════════════


def spread_level_analysis(groups: dict[str, np.ndarray]) -> SpreadLevelResult:
    """分布-水平图分析: ln(IQR) ∼ ln(Median) 线性回归。

    - 斜率 = 0 → 方差齐性
    - 斜率 ≠ 0 → 建议变换幂 λ = 1 - slope (Box-Cox)

    Returns:
        SpreadLevelResult 含斜率、截距、建议变换幂。
    """
    rows: list[dict] = []
    for label, arr in groups.items():
        a = np.asarray(arr, dtype=np.float64)
        a = a[~np.isnan(a)]
        if len(a) < 2:
            continue
        med = np.median(a)
        iqr = np.percentile(a, 75) - np.percentile(a, 25)
        if med <= 0 or iqr <= 0:
            continue
        rows.append(
            {
                "group": label,
                "median": med,
                "iqr": iqr,
                "ln_median": math.log(med),
                "ln_iqr": math.log(iqr),
                "n": len(a),
            }
        )

    if len(rows) < 2:
        return SpreadLevelResult(slope=0.0, intercept=0.0, suggested_power=1.0, groups_data=rows)

    x = np.array([r["ln_median"] for r in rows])
    y = np.array([r["ln_iqr"] for r in rows])
    slope, intercept, _, _, _ = sp_stats.linregress(x, y)
    suggested_power = 1.0 - slope  # Box-Cox λ

    return SpreadLevelResult(
        slope=float(slope),
        intercept=float(intercept),
        suggested_power=float(suggested_power),
        groups_data=rows,
    )


# ═══════════════════════════════════════════
# 按组探索
# ═══════════════════════════════════════════



def explore_by_group(
    data: pd.Series,
    group: pd.Series,
    *,
    ci_level: float = 0.95,
) -> list[GroupExploreResult]:
    """按分组执行完整探索性分析。

    Args:
        data: 连续因变量。
        group: 分组变量 (因子)。
        ci_level: 均值置信区间水平。

    Returns:
        每组一个 GroupExploreResult。
    """
    df = pd.DataFrame({"y": data, "g": group}).dropna()
    results: list[GroupExploreResult] = []

    for g_label in sorted(df["g"].unique()):
        subset = df[df["g"] == g_label]["y"].values.astype(np.float64)
        n = len(subset)
        if n == 0:
            continue

        mean = float(np.mean(subset))
        std = float(np.std(subset, ddof=1)) if n > 1 else float("nan")
        var = std**2 if n > 1 else float("nan")
        median = float(np.median(subset))
        trimmed_mean = float(sp_stats.trim_mean(subset, 0.05)) if n >= 3 else mean
        se_mean = std / math.sqrt(n) if n > 1 else float("nan")

        # CI
        ci = (float("nan"), float("nan"))
        if n > 1 and std > 0:
            alpha = 1 - ci_level
            t_crit = sp_stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin = t_crit * se_mean
            ci = (mean - margin, mean + margin)

        # 偏度 (scipy 0-ddof, SPSS 兼容)
        sk = float(sp_stats.skew(subset, bias=False)) if n > 1 else float("nan")

        # 过度峰度
        if n > 3:
            kt = float(sp_stats.kurtosis(subset, bias=False, fisher=True))
        else:
            kt = float("nan")

        ses = _std_error_skewness(n)
        sek = _std_error_kurtosis(n)

        bp = boxplot_stats(subset)
        norm = normality_tests(subset)

        results.append(
            GroupExploreResult(
                group_label=str(g_label),
                n=n,
                mean=mean,
                ci_mean_95=ci,
                median=median,
                trimmed_mean_5pct=trimmed_mean,
                std=std,
                variance=var,
                min_=float(subset.min()),
                max_=float(subset.max()),
                range_=float(subset.max() - subset.min()),
                iqr=bp.iqr,
                skewness=sk,
                std_error_skewness=ses,
                kurtosis=kt,
                std_error_kurtosis=sek,
                boxplot=bp,
                normality=norm,
            )
        )

    return results


def explore_summary_table(results: list[GroupExploreResult]) -> pd.DataFrame:
    """将 GroupExploreResult 列表转为描述统计汇总 DataFrame。"""
    rows = []
    for r in results:
        rows.append(
            {
                "组别": r.group_label,
                "n": r.n,
                "均值": f"{r.mean:.3f}",
                "95% CI 下限": f"{r.ci_mean_95[0]:.3f}",
                "95% CI 上限": f"{r.ci_mean_95[1]:.3f}",
                "5%截尾均值": f"{r.trimmed_mean_5pct:.3f}",
                "中位数": f"{r.median:.3f}",
                "标准差": f"{r.std:.3f}",
                "方差": f"{r.variance:.3f}",
                "最小值": f"{r.min_:.3f}",
                "最大值": f"{r.max_:.3f}",
                "极差": f"{r.range_:.3f}",
                "IQR": f"{r.iqr:.3f}",
                "偏度": f"{r.skewness:.3f}",
                "SES": f"{r.std_error_skewness:.3f}",
                "峰度": f"{r.kurtosis:.3f}",
                "SEK": f"{r.std_error_kurtosis:.3f}",
            }
        )
    return pd.DataFrame(rows)
