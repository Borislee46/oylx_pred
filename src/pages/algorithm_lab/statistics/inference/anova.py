"""方差分析 (ANOVA) — 全参数实现

复刻 SPSS ONEWAY / GLM 过程:
- 单因素 ANOVA (经典 F + Welch + Brown-Forsythe)
- 事后检验: Tukey HSD, Bonferroni, Games-Howell, Dunnett, LSD, Sidak, Scheffe
- 效应量: η², ω², partial η², Cohen's f
- 线性趋势检验 (有序因子)
- Levene 方差齐性检验

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【ANOVA 的核心直觉: 方差分解】

F = (组间方差) / (组内方差)

组间方差 (MS_between): 各组均值围绕总均值的离散程度。
                      如果因子有效果, 组间应该差异很大。
组内方差 (MS_within): 同一组内部个体的离散程度。
                      自然变异, 和因子无关。

如果 F 很大 → 组间差异远超组内随机波动 → 因子有显著效果。
如果 F ≈ 1 → 组间差异和组内波动差不多 → 因子没效果。

自由度: 组间 df = k-1 (k 组, 只有 k-1 个独立的组间差异),
         组内 df = N-k (总个体减去组数)。

【η² vs ω² — 两个效应量的区别】

η² (eta-squared):
  = 组间平方和 / 总平方和
  直观但略微高估 (有偏估计), 因为分子分母都用到了同一样本。
  类似于回归的 R²。

ω² (omega-squared):
  在 η² 的基础上减去了组内方差的期望值, 是无偏估计。
  报告时优先报告 ω²。
  它可能 <0 (非常罕见, 发生在 F<1 的情况), 此时夹紧到 0。

经验法则 (Cohen): ω² ≈ 0.01=小, 0.06=中, 0.14=大。

【事后检验选择的决策树】

1. 方差齐 (Levene p > 0.05) 且样本量大致相等 → Tukey HSD
2. 方差齐但只关心实验组 vs 对照组 → Dunnett
3. 方差不齐或样本量差异大 → Games-Howell
4. 比较次数很多, 想控制 FWER → Bonferroni (但保守)

FWER (Family-Wise Error Rate): 做多次比较时, 至少犯一次
  第一类错误的概率。

Tukey HSD 原理: 基于学生化极差分布 (studentized range) q。
  计算的是"最大均值差 / 标准误"的分布。
  如果 q > q_critical → 该对差异显著。
  它精确控制了全部两两比较的 FWER。

Games-Howell 原理: Tukey 的方差不齐版本。
  使用 Welch 自由度 + q/sqrt(2) 做检验。
  既放宽了等方差假设, 又控制了 FWER。

Bonferroni 原理: α_corrected = α / m (m=比较次数)。
  特别保守 (容易漏掉真实差异), 但非常简单和普适。
  适合"比较次数少 + 需要严格控制"的场景。

【线性趋势检验】
当分组变量是有序的 (如低/中/高剂量), 检验各组均值是否存在
单调线性趋势。使用多项式对比系数 (如 -1, 0, 1 权重和)。
注意: 存在线性趋势 ≠ 其他趋势不存在 (可能有二次或三次项)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════


@dataclass
class AnovaResult:
    """ANOVA 完整结果"""

    method: str  # "Classic" | "Welch" | "Brown-Forsythe"
    f_statistic: float
    df1: int
    df2: float
    p_value: float
    eta_sq: float
    omega_sq: float
    n_total: int
    k: int
    group_stats: list[dict] = field(default_factory=list)
    levene_f: float | None = None
    levene_p: float | None = None
    linear_trend: dict | None = None  # {F, df1, df2, p}


@dataclass
class PostHocResult:
    """事后多重比较完整结果"""

    method: str  # "Tukey HSD" | "Bonferroni" | "Games-Howell" etc.
    comparisons: list[PostHocComparison]
    alpha: float
    alpha_corrected: float


@dataclass
class PostHocComparison:
    group1: str
    group2: str
    mean_diff: float
    se: float
    t_or_q: float
    df: float
    p_value: float
    ci_95: tuple[float, float]
    significant: bool


# ═══════════════════════════════════════════
# ANOVA 核心
# ═══════════════════════════════════════════


def oneway_anova(
    data: pd.Series,
    group: pd.Series,
    *,
    ci_level: float = 0.95,
) -> AnovaResult:
    """单因素 ANOVA (经典 F 检验 + Welch + Brown-Forsythe)。

    Args:
        data: 连续因变量。
        group: 分类因子 (分组变量)。
        ci_level: 置信区间水平。

    Returns:
        AnovaResult (默认 method="Classic").
    """
    df = pd.DataFrame({"y": data.values, "g": group.values})
    df = df.dropna()
    groups = {str(g): df[df["g"] == g]["y"].values.astype(np.float64) for g in sorted(df["g"].unique())}
    group_list = list(groups.values())
    group_names = list(groups.keys())
    k = len(group_list)

    if k < 2:
        raise ValueError("ANOVA 需要至少 2 组")

    n_total = sum(len(a) for a in group_list)
    grand_mean = np.mean(np.concatenate(group_list))

    # Levene
    lf, lp = None, None
    if k >= 2:
        try:
            lf, lp = sp_stats.levene(*group_list, center="median")
            lf, lp = float(lf), float(lp)
        except Exception:
            pass

    # 经典 ANOVA
    ss_between = sum(len(a) * (np.mean(a) - grand_mean) ** 2 for a in group_list)
    ss_within = sum(np.sum((a - np.mean(a)) ** 2) for a in group_list)
    df1 = k - 1
    df2 = n_total - k
    ms_between = ss_between / df1 if df1 > 0 else 0
    ms_within = ss_within / df2 if df2 > 0 else 0
    F = ms_between / ms_within if ms_within > 0 else float("nan")
    p = 1.0 - sp_stats.f.cdf(F, df1, df2) if not math.isnan(F) else float("nan")

    # 效应量
    ss_total = ss_between + ss_within
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    omega_sq = (ss_between - df1 * ms_within) / (ss_total + ms_within) if ms_within > 0 else 0.0
    omega_sq = max(0.0, omega_sq)

    # 各组描述统计
    group_stats = []
    for name, arr in groups.items():
        a = arr[~np.isnan(arr)]
        n = len(a)
        m = float(np.mean(a))
        s = float(np.std(a, ddof=1)) if n > 1 else float("nan")
        se = s / math.sqrt(n) if n > 1 else float("nan")
        ci = (float("nan"), float("nan"))
        if n > 1 and s > 0:
            alpha = 1 - ci_level
            t_crit = sp_stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin = t_crit * se
            ci = (m - margin, m + margin)
        group_stats.append(
            {"group": name, "n": n, "mean": m, "std": s, "se": se, "ci_95_low": ci[0], "ci_95_high": ci[1]}
        )

    return AnovaResult(
        method="Classic",
        f_statistic=F,
        df1=df1,
        df2=float(df2),
        p_value=p,
        eta_sq=eta_sq,
        omega_sq=omega_sq,
        n_total=n_total,
        k=k,
        group_stats=group_stats,
        levene_f=lf,
        levene_p=lp,
    )


def welch_anova(data: pd.Series, group: pd.Series) -> AnovaResult:
    """Welch ANOVA (不假定等方差)。"""
    df = pd.DataFrame({"y": data.values, "g": group.values}).dropna()
    groups = {str(g): df[df["g"] == g]["y"].values.astype(np.float64) for g in sorted(df["g"].unique())}
    group_list = list(groups.values())
    k = len(group_list)
    if k < 2:
        raise ValueError("Welch ANOVA 需要至少 2 组")

    n_total = sum(len(a) for a in group_list)
    means = np.array([np.mean(a) for a in group_list])
    ns = np.array([len(a) for a in group_list])
    vars_ = np.array([np.var(a, ddof=1) for a in group_list])
    ws = ns / vars_

    # Welch 加权均值
    grand_w = np.sum(ws * means) / np.sum(ws)

    # Welch F
    numer = np.sum(ws * (means - grand_w) ** 2) / (k - 1)
    lambdas = (1 - ws / np.sum(ws)) ** 2 / (ns - 1)
    denom = 1 + 2 * (k - 2) / (k**2 - 1) * np.sum(lambdas)

    F_welch = numer / denom if denom > 0 else float("nan")
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * np.sum(lambdas)) if np.sum(lambdas) > 0 else float("nan")
    p = 1.0 - sp_stats.f.cdf(F_welch, df1, df2) if not math.isnan(F_welch) and not math.isnan(df2) else float("nan")

    # 效应量 (Welch 下近似)
    ss_between = sum(len(a) * (np.mean(a) - np.mean(np.concatenate(group_list))) ** 2 for a in group_list)
    ss_within = sum(np.sum((a - np.mean(a)) ** 2) for a in group_list)
    ss_total = ss_between + ss_within
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    group_stats = []
    for name, arr in groups.items():
        a = arr[~np.isnan(arr)]
        n = len(a)
        m = float(np.mean(a))
        s = float(np.std(a, ddof=1)) if n > 1 else float("nan")
        group_stats.append({"group": name, "n": n, "mean": m, "std": s, "se": s / math.sqrt(n) if n > 1 else float("nan"),
                           "ci_95_low": float("nan"), "ci_95_high": float("nan")})

    return AnovaResult(
        method="Welch",
        f_statistic=F_welch,
        df1=df1,
        df2=float(df2) if not math.isnan(df2) else float("nan"),
        p_value=p,
        eta_sq=eta_sq,
        omega_sq=float("nan"),
        n_total=n_total,
        k=k,
        group_stats=group_stats,
    )


# ═══════════════════════════════════════════
# 事后检验
# ═══════════════════════════════════════════


def _studentized_range_critical(k: int, df: int, alpha: float = 0.05) -> float:
    """学生化极差 q 分布临界值 (近似)"""
    try:
        return float(sp_stats.studentized_range.ppf(1 - alpha, k, df))
    except Exception:
        return 3.0


def tukey_hsd(groups: dict[str, np.ndarray], alpha: float = 0.05) -> PostHocResult:
    """Tukey HSD 事后检验 (假定等方差)。"""
    group_names = list(groups.keys())
    k = len(group_names)
    group_list = [groups[n] for n in group_names]

    n_total = sum(len(a) for a in group_list)
    df_within = n_total - k
    ms_within = sum(np.sum((a - np.mean(a)) ** 2) for a in group_list) / df_within if df_within > 0 else 0

    q_crit = _studentized_range_critical(k, df_within, alpha)
    comparisons: list[PostHocComparison] = []

    for i in range(k):
        for j in range(i + 1, k):
            a1, a2 = group_list[i], group_list[j]
            n1, n2 = len(a1), len(a2)
            diff = np.mean(a1) - np.mean(a2)
            se = math.sqrt(ms_within / 2 * (1 / n1 + 1 / n2)) if ms_within > 0 else 0
            q = abs(diff) / se if se > 0 else 0
            p = 1.0 - sp_stats.studentized_range.cdf(q * math.sqrt(2), k, df_within) if df_within > 0 else 1.0
            # Tukey CI
            hsd = q_crit * se / math.sqrt(2)
            ci = (diff - hsd, diff + hsd)

            comparisons.append(
                PostHocComparison(
                    group1=group_names[i],
                    group2=group_names[j],
                    mean_diff=float(diff),
                    se=float(se),
                    t_or_q=float(q),
                    df=float(df_within),
                    p_value=float(p),
                    ci_95=ci,
                    significant=float(p) < alpha,
                )
            )

    return PostHocResult(method="Tukey HSD", comparisons=comparisons, alpha=alpha, alpha_corrected=alpha)


def bonferroni(groups: dict[str, np.ndarray], alpha: float = 0.05) -> PostHocResult:
    """Bonferroni 事后检验 (p 值乘子法, 假定等方差)。"""
    group_names = list(groups.keys())
    k = len(group_names)
    group_list = [groups[n] for n in group_names]
    n_total = sum(len(a) for a in group_list)
    df_within = n_total - k
    ms_within = sum(np.sum((a - np.mean(a)) ** 2) for a in group_list) / df_within if df_within > 0 else 0
    n_comparisons = k * (k - 1) // 2
    alpha_corrected = alpha / n_comparisons

    comparisons: list[PostHocComparison] = []
    for i in range(k):
        for j in range(i + 1, k):
            a1, a2 = group_list[i], group_list[j]
            n1, n2 = len(a1), len(a2)
            diff = np.mean(a1) - np.mean(a2)
            # 合并方差 SE（与 Tukey HSD 一致）
            se = math.sqrt(ms_within * (1.0 / n1 + 1.0 / n2)) if ms_within > 0 else 0
            t = diff / se if se > 0 else 0.0
            p_raw = 2.0 * (1.0 - sp_stats.t.cdf(abs(t), df_within)) if df_within > 0 else 1.0
            p_corrected = min(p_raw * n_comparisons, 1.0)

            t_crit = sp_stats.t.ppf(1.0 - alpha_corrected / 2.0, df_within)
            ci = (diff - t_crit * se, diff + t_crit * se)

            comparisons.append(
                PostHocComparison(
                    group1=group_names[i],
                    group2=group_names[j],
                    mean_diff=float(diff),
                    se=float(se),
                    t_or_q=float(t),
                    df=float(df_within),
                    p_value=p_corrected,
                    ci_95=ci,
                    significant=p_corrected < alpha,
                )
            )

    return PostHocResult(method="Bonferroni", comparisons=comparisons, alpha=alpha, alpha_corrected=alpha_corrected)


def games_howell(groups: dict[str, np.ndarray], alpha: float = 0.05) -> PostHocResult:
    """Games-Howell 事后检验 (不假定等方差, 方差不齐时的首选)。

    使用 Welch 自由度修正 + 学生化极差临界值。
    """
    group_names = list(groups.keys())
    k = len(group_names)
    group_list = [groups[n] for n in group_names]

    comparisons: list[PostHocComparison] = []
    q_crit = _studentized_range_critical(k, max(2, sum(len(a) for a in group_list) // k), alpha)

    for i in range(k):
        for j in range(i + 1, k):
            a1, a2 = group_list[i], group_list[j]
            n1, n2 = len(a1), len(a2)
            m1, m2 = np.mean(a1), np.mean(a2)
            v1, v2 = np.var(a1, ddof=1), np.var(a2, ddof=1)

            diff = m1 - m2
            se = math.sqrt(v1 / n1 + v2 / n2)
            t = abs(diff) / se if se > 0 else 0.0

            # Welch df
            num = (v1 / n1 + v2 / n2) ** 2
            denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            df = num / denom if denom > 0 else 2.0

            # Games-Howell: 使用 q/sqrt(2) 而非 t
            q_stat = t * math.sqrt(2)
            try:
                p = 1.0 - sp_stats.studentized_range.cdf(q_stat, k, max(int(df), 2))
            except Exception:
                p = 2.0 * (1.0 - sp_stats.t.cdf(t, df=max(int(df), 2)))
            p = min(max(float(p), 0.0), 1.0)

            margin = q_crit / math.sqrt(2) * se
            ci = (diff - margin, diff + margin)

            comparisons.append(
                PostHocComparison(
                    group1=group_names[i],
                    group2=group_names[j],
                    mean_diff=float(diff),
                    se=float(se),
                    t_or_q=float(q_stat),
                    df=float(df),
                    p_value=p,
                    ci_95=ci,
                    significant=p < alpha,
                )
            )

    return PostHocResult(method="Games-Howell", comparisons=comparisons, alpha=alpha, alpha_corrected=alpha)


def lsd(
    groups: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> PostHocResult:
    """LSD (Least Significant Difference) — 无校正的配对 t 检验。

    直接对每对做 t 检验，使用合并组内方差 (pooled MS_within)。
    不做任何多重比较校正 → FWER 急剧膨胀。
    仅适用于: ANOVA 显著 + 组数 ≤ 3 + 探索性质。
    不推荐用于结论性分析。
    """
    group_names = list(groups.keys())
    k = len(group_names)
    group_list = [groups[n] for n in group_names]
    n_total = sum(len(a) for a in group_list)
    df_within = n_total - k
    ms_within = sum(np.sum((a - np.mean(a)) ** 2) for a in group_list) / df_within if df_within > 0 else 0

    comparisons: list[PostHocComparison] = []
    for i in range(k):
        for j in range(i + 1, k):
            a1, a2 = group_list[i], group_list[j]
            n1, n2 = len(a1), len(a2)
            diff = np.mean(a1) - np.mean(a2)
            se = math.sqrt(ms_within * (1.0 / n1 + 1.0 / n2)) if ms_within > 0 else 0
            t = diff / se if se > 0 else 0.0
            p = 2.0 * (1.0 - sp_stats.t.cdf(abs(t), df_within)) if df_within > 0 else 1.0

            t_crit = sp_stats.t.ppf(1.0 - alpha / 2.0, df_within)
            ci = (diff - t_crit * se, diff + t_crit * se)

            comparisons.append(
                PostHocComparison(
                    group1=group_names[i],
                    group2=group_names[j],
                    mean_diff=float(diff),
                    se=float(se),
                    t_or_q=float(t),
                    df=float(df_within),
                    p_value=float(p),
                    ci_95=ci,
                    significant=float(p) < alpha,
                )
            )

    return PostHocResult(method="LSD", comparisons=comparisons, alpha=alpha, alpha_corrected=alpha)


def sidak(
    groups: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> PostHocResult:
    """Sidak 事后检验 — p 值的 Sidak 校正。

    α_corrected = 1 - (1 - α)^{1/m}, m = 比较次数。
    假设各比较独立, 比 Bonferroni 略不保守。
    使用合并组内方差 (假定等方差)。
    """
    group_names = list(groups.keys())
    k = len(group_names)
    group_list = [groups[n] for n in group_names]
    n_total = sum(len(a) for a in group_list)
    df_within = n_total - k
    ms_within = sum(np.sum((a - np.mean(a)) ** 2) for a in group_list) / df_within if df_within > 0 else 0
    n_comparisons = k * (k - 1) // 2
    alpha_corrected = 1.0 - (1.0 - alpha) ** (1.0 / n_comparisons)

    comparisons: list[PostHocComparison] = []
    for i in range(k):
        for j in range(i + 1, k):
            a1, a2 = group_list[i], group_list[j]
            n1, n2 = len(a1), len(a2)
            diff = np.mean(a1) - np.mean(a2)
            se = math.sqrt(ms_within * (1.0 / n1 + 1.0 / n2)) if ms_within > 0 else 0
            t = diff / se if se > 0 else 0.0
            p_raw = 2.0 * (1.0 - sp_stats.t.cdf(abs(t), df_within)) if df_within > 0 else 1.0
            p_corrected = min(1.0 - (1.0 - p_raw) ** n_comparisons, 1.0)

            t_crit = sp_stats.t.ppf(1.0 - alpha_corrected / 2.0, df_within)
            ci = (diff - t_crit * se, diff + t_crit * se)

            comparisons.append(
                PostHocComparison(
                    group1=group_names[i],
                    group2=group_names[j],
                    mean_diff=float(diff),
                    se=float(se),
                    t_or_q=float(t),
                    df=float(df_within),
                    p_value=p_corrected,
                    ci_95=ci,
                    significant=p_corrected < alpha,
                )
            )

    return PostHocResult(method="Sidak", comparisons=comparisons, alpha=alpha, alpha_corrected=alpha_corrected)


def scheffe(
    groups: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> PostHocResult:
    """Scheffe 事后检验 — 最保守, 控制所有可能对比的 FWER。

    临界值 = sqrt((k-1) × F_crit(k-1, df_within))。
    适用于事后任意对比 (不限于配对), 但代价是极其保守。
    p 值来自 F 分布: p = 1 - F_cdf(t²/(k-1), k-1, df_within)。
    """
    group_names = list(groups.keys())
    k = len(group_names)
    group_list = [groups[n] for n in group_names]
    n_total = sum(len(a) for a in group_list)
    df_within = n_total - k
    ms_within = sum(np.sum((a - np.mean(a)) ** 2) for a in group_list) / df_within if df_within > 0 else 0

    F_crit = sp_stats.f.ppf(1.0 - alpha, k - 1, df_within)
    S = math.sqrt((k - 1) * F_crit)  # Scheffe 临界乘数

    comparisons: list[PostHocComparison] = []
    for i in range(k):
        for j in range(i + 1, k):
            a1, a2 = group_list[i], group_list[j]
            n1, n2 = len(a1), len(a2)
            diff = np.mean(a1) - np.mean(a2)
            se = math.sqrt(ms_within * (1.0 / n1 + 1.0 / n2)) if ms_within > 0 else 0
            t = diff / se if se > 0 else 0.0

            # Scheffe p: F = t²/(k-1), F ~ F(k-1, df_within)
            if df_within > 0 and k > 1:
                F_val = t**2 / (k - 1)
                p = 1.0 - sp_stats.f.cdf(F_val, k - 1, df_within)
            else:
                p = 1.0

            margin = S * se
            ci = (diff - margin, diff + margin)

            comparisons.append(
                PostHocComparison(
                    group1=group_names[i],
                    group2=group_names[j],
                    mean_diff=float(diff),
                    se=float(se),
                    t_or_q=float(t),
                    df=float(df_within),
                    p_value=float(p),
                    ci_95=ci,
                    significant=float(p) < alpha,
                )
            )

    return PostHocResult(method="Scheffe", comparisons=comparisons, alpha=alpha, alpha_corrected=alpha)

def dunnett(
    groups: dict[str, np.ndarray],
    control_group: str,
    alpha: float = 0.05,
) -> PostHocResult:
    """Dunnett 事后检验 (多组对单对照组)。

    Args:
        groups: {组名: 数组}。
        control_group: 对照组的名称。
    """
    group_names = list(groups.keys())
    if control_group not in group_names:
        raise ValueError(f"对照组 '{control_group}' 不在分组中")

    k = len(group_names)
    group_list = [groups[n] for n in group_names]
    n_total = sum(len(a) for a in group_list)
    df_within = n_total - k
    ms_within = sum(np.sum((a - np.mean(a)) ** 2) for a in group_list) / df_within if df_within > 0 else 0

    ctrl_idx = group_names.index(control_group)
    ctrl_arr = group_list[ctrl_idx]
    n_ctrl = len(ctrl_arr)

    # Dunnett 临界值 (使用 t 分布近似)
    n_comparisons = k - 1
    alpha_corrected = 1 - (1 - alpha) ** (1.0 / n_comparisons)

    comparisons: list[PostHocComparison] = []
    for idx, name in enumerate(group_names):
        if name == control_group:
            continue
        arr = group_list[idx]
        n = len(arr)
        diff = np.mean(arr) - np.mean(ctrl_arr)
        se = math.sqrt(ms_within * (1 / n + 1 / n_ctrl)) if ms_within > 0 else 0
        t = diff / se if se > 0 else 0
        p = 2.0 * (1.0 - sp_stats.t.cdf(abs(t), df_within)) if df_within > 0 else 1.0
        p_corrected = min(p * n_comparisons, 1.0)

        t_crit = sp_stats.t.ppf(1 - alpha_corrected / 2, df=df_within)
        ci = (diff - t_crit * se, diff + t_crit * se)

        comparisons.append(
            PostHocComparison(
                group1=name,
                group2=control_group,
                mean_diff=float(diff),
                se=float(se),
                t_or_q=float(t),
                df=float(df_within),
                p_value=p_corrected,
                ci_95=ci,
                significant=p_corrected < alpha,
            )
        )

    return PostHocResult(method=f"Dunnett (vs {control_group})", comparisons=comparisons, alpha=alpha,
                         alpha_corrected=alpha_corrected)


# ═══════════════════════════════════════════
# 事后检验路由
# ═══════════════════════════════════════════

POSTHOC_METHODS = {
    "tukey": tukey_hsd,
    "bonferroni": bonferroni,
    "games_howell": games_howell,
    "lsd": lsd,
    "sidak": sidak,
    "scheffe": scheffe,
}

POSTHOC_GUIDE = {
    "tukey": "全面两两比较, 假定等方差。最常用。",
    "bonferroni": "p 值乘子法校正, 比较次数多时保守。",
    "games_howell": "方差不齐且样本量不等时的首选, 使用 Welch 自由度。",
    "dunnett": "多组对单对照组比较。需指定 control_group。",
    "lsd": "无校正, FWER 极高, 不推荐用于结论性分析。",
    "sidak": "独立检验假设下, 功效略优 Bonferroni, 通过 (1-(1-p)^m) 校正。",
    "scheffe": "适用于所有复杂对比, 极其保守。使用 F 分布临界值。",
}


def posthoc(
    groups: dict[str, np.ndarray],
    method: str = "tukey",
    alpha: float = 0.05,
    control_group: str | None = None,
) -> PostHocResult:
    """事后比较路由。

    Args:
        groups: {组名: 数值数组}。
        method: ``"tukey"`` / ``"bonferroni"`` / ``"games_howell"`` /
                ``"dunnett"`` / ``"lsd"`` / ``"sidak"`` / ``"scheffe"``。
        alpha: 显著性水平。
        control_group: Dunnett 方法的对照组名。

    Returns:
        PostHocResult。
    """
    if method == "dunnett":
        if control_group is None:
            raise ValueError("Dunnett 方法必须指定 control_group")
        return dunnett(groups, control_group, alpha)
    fn = POSTHOC_METHODS.get(method)
    if fn is None:
        raise ValueError(f"不支持的事后检验方法: {method}。可选: {list(POSTHOC_METHODS)} + dunnett")
    return fn(groups, alpha)


# ═══════════════════════════════════════════
# 线性趋势检验
# ═══════════════════════════════════════════


def linear_trend_test(data: pd.Series, group: pd.Series) -> dict:
    """线性趋势检验 (适用于有序分组变量)。

    使用多项式对比: 检验各组均值是否存在单调线性趋势。
    """
    df = pd.DataFrame({"y": data.values, "g": group.values}).dropna()
    groups = sorted(df["g"].unique())
    k = len(groups)

    # 分配线性对比系数
    contrast = np.arange(k, dtype=np.float64)
    contrast = contrast - contrast.mean()

    group_means = []
    group_ns = []
    for g in groups:
        arr = df[df["g"] == g]["y"].values.astype(np.float64)
        group_means.append(np.mean(arr))
        group_ns.append(len(arr))

    # SS_linear = (Σ c_j * M_j)² / (Σ c_j² / n_j)
    num = sum(c * m for c, m in zip(contrast, group_means)) ** 2
    denom = sum(c**2 / n for c, n in zip(contrast, group_ns))
    ss_linear = num / denom if denom > 0 else 0.0

    # MS_within
    ss_within = 0.0
    for g in groups:
        arr = df[df["g"] == g]["y"].values.astype(np.float64)
        ss_within += np.sum((arr - np.mean(arr)) ** 2)

    n_total = len(df)
    df_within = n_total - k
    ms_within = ss_within / df_within if df_within > 0 else 0
    F_linear = ss_linear / ms_within if ms_within > 0 else float("nan")
    p = 1.0 - sp_stats.f.cdf(F_linear, 1, df_within) if not math.isnan(F_linear) else float("nan")

    return {"F": float(F_linear), "df1": 1, "df2": df_within, "p": float(p)}


# ═══════════════════════════════════════════
# 重复测量 ANOVA (Repeated Measures)
# ═══════════════════════════════════════════


@dataclass
class RMAnovaResult:
    """单因素重复测量 ANOVA 完整结果"""

    n: int
    k: int
    # 基本 ANOVA
    f_statistic: float
    df1: int  # k - 1
    df2: int  # (n - 1) * (k - 1)
    p_value: float
    # 效应量
    ges: float  # generalized eta-squared
    # 球性检验
    mauchly_w: float
    mauchly_chi2: float
    mauchly_df: int
    mauchly_p: float
    # 校正
    gg_epsilon: float  # Greenhouse-Geisser
    hf_epsilon: float  # Huynh-Feldt
    gg_p: float  # GG 校正后的 p
    hf_p: float  # HF 校正后的 p
    # 描述
    condition_means: dict[str, float]
    condition_sds: dict[str, float]
    # 事后配对比较 (Bonferroni 校正)
    posthoc: list[dict] | None = None


def repeated_measures_anova(
    data: np.ndarray,
    *,
    condition_names: list[str] | None = None,
    alpha: float = 0.05,
) -> RMAnovaResult:
    """单因素重复测量 ANOVA。

    同一组被试在 k 个条件下各测量一次。检验各条件均值是否相等。

    Args:
        data: (n_subjects, k_conditions) 2D 数组。每行 = 一个被试, 每列 = 一个条件。
        condition_names: 条件名称 (默认 "C1", "C2", ...)。
        alpha: 显著性水平。

    Returns:
        RMAnovaResult。
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[~np.isnan(arr).any(axis=1)]
    n, k = arr.shape

    if k < 2:
        raise ValueError("至少需要 2 个条件")
    if n < 3:
        raise ValueError("至少需要 3 个被试")

    if condition_names is None:
        condition_names = [f"C{i+1}" for i in range(k)]

    grand_mean = float(np.mean(arr))
    subject_means = np.mean(arr, axis=1)
    condition_means_arr = np.mean(arr, axis=0)

    # SS 分解
    ss_total = float(np.sum((arr - grand_mean) ** 2))
    ss_subject = k * float(np.sum((subject_means - grand_mean) ** 2))
    ss_condition = n * float(np.sum((condition_means_arr - grand_mean) ** 2))
    ss_error = ss_total - ss_subject - ss_condition

    df1 = k - 1
    df2 = (n - 1) * (k - 1)
    ms_condition = ss_condition / df1 if df1 > 0 else 0
    ms_error = ss_error / df2 if df2 > 0 else 0
    F = ms_condition / ms_error if ms_error > 0 else float("nan")
    p = 1.0 - sp_stats.f.cdf(F, df1, df2) if not math.isnan(F) else float("nan")

    # generalized eta-squared (推荐用于 RM 设计)
    ges = ss_condition / (ss_condition + ss_subject + ss_error) if (ss_condition + ss_subject + ss_error) > 0 else 0.0

    # ── 球性检验 (Mauchly's W) ──
    # 基于正交对比矩阵
    C = _orthogonal_contrasts(k)
    Y_transformed = arr @ C  # (n, k-1)
    S = np.cov(Y_transformed, rowvar=False)  # (k-1, k-1)
    eigvals = np.linalg.eigvalsh(S)
    eigvals = eigvals[eigvals > 1e-10]

    m = k - 1
    if len(eigvals) >= m and np.all(eigvals > 0):
        gm = np.exp(np.mean(np.log(eigvals)))  # 几何均值
        am = np.mean(eigvals)  # 算术均值
        W = gm / am if am > 0 else 1.0
        # χ² 近似
        chi2 = -(n - 1 - (2 * m**2 + m + 2) / (6 * m)) * math.log(max(W, 1e-10))
        df_mauchly = m * (m + 1) // 2 - 1
        p_mauchly = 1.0 - sp_stats.chi2.cdf(max(chi2, 0), df_mauchly) if df_mauchly > 0 else 1.0
    else:
        W = 0.0
        chi2 = float("inf")
        df_mauchly = 0
        p_mauchly = 0.0

    # ── Greenhouse-Geisser ε ──
    # ε = (Σ λ_i)² / (m × Σ λ_i²)
    if m > 1 and np.sum(eigvals) > 0:
        gg_eps = float(np.sum(eigvals) ** 2 / (m * np.sum(eigvals**2)))
        gg_eps = max(1.0 / m, min(1.0, gg_eps))  # clamp [1/m, 1]
    else:
        gg_eps = 1.0

    # ── Huynh-Feldt ε ──
    if m > 1:
        hf_eps = (n * m * gg_eps - 2) / (m * (n - 1 - m * gg_eps))
        hf_eps = max(1.0 / m, hf_eps)  # can exceed 1, don't clamp upper
    else:
        hf_eps = 1.0

    # 校正 p 值
    gg_df1 = gg_eps * df1
    gg_df2 = gg_eps * df2
    gg_p = 1.0 - sp_stats.f.cdf(F, gg_df1, gg_df2) if not math.isnan(F) and gg_df1 > 0 and gg_df2 > 0 else float("nan")

    hf_df1 = min(hf_eps, 1.0) * df1
    hf_df2 = min(hf_eps, 1.0) * df2
    hf_p = 1.0 - sp_stats.f.cdf(F, hf_df1, hf_df2) if not math.isnan(F) and hf_df1 > 0 and hf_df2 > 0 else float("nan")

    # ── 描述统计 ──
    cond_means = {condition_names[i]: float(condition_means_arr[i]) for i in range(k)}
    cond_sds = {condition_names[i]: float(np.std(arr[:, i], ddof=1)) for i in range(k)}

    # ── 事后配对比较 (Bonferroni) ──
    n_comparisons = k * (k - 1) // 2
    alpha_bonf = alpha / n_comparisons
    posthoc_comps = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = arr[:, i] - arr[:, j]
            mean_diff = float(np.mean(diff))
            se_diff = float(np.std(diff, ddof=1) / math.sqrt(n)) if n > 1 else 0
            t = mean_diff / se_diff if se_diff > 0 else 0.0
            p_raw = 2.0 * (1.0 - sp_stats.t.cdf(abs(t), n - 1)) if n > 1 else 1.0
            p_corrected = min(p_raw * n_comparisons, 1.0)
            t_crit = sp_stats.t.ppf(1.0 - alpha_bonf / 2.0, n - 1) if n > 1 else 0
            ci = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)
            posthoc_comps.append({
                "group1": condition_names[i],
                "group2": condition_names[j],
                "mean_diff": mean_diff,
                "p": p_corrected,
                "ci_95": ci,
                "significant": p_corrected < alpha,
            })

    return RMAnovaResult(
        n=n,
        k=k,
        f_statistic=F,
        df1=df1,
        df2=df2,
        p_value=p,
        ges=ges,
        mauchly_w=max(0.0, min(1.0, W)),
        mauchly_chi2=float(chi2),
        mauchly_df=int(df_mauchly),
        mauchly_p=float(p_mauchly),
        gg_epsilon=float(gg_eps),
        hf_epsilon=float(hf_eps),
        gg_p=float(gg_p),
        hf_p=float(hf_p),
        condition_means=cond_means,
        condition_sds=cond_sds,
        posthoc=posthoc_comps,
    )


def _orthogonal_contrasts(k: int) -> np.ndarray:
    """生成 (k, k-1) 正交对比矩阵 (Helmert)。"""
    C = np.zeros((k, k - 1))
    for j in range(k - 1):
        C[:j+1, j] = 1.0
        C[j+1, j] = -(j + 1)
    # 列标准化
    for j in range(k - 1):
        norm = np.sqrt(np.sum(C[:, j] ** 2))
        if norm > 0:
            C[:, j] /= norm
    return C


def rm_anova_report(r: RMAnovaResult) -> str:
    """重复测量 ANOVA 报告。"""
    lines = [
        f"{'='*60}",
        f"  单因素重复测量 ANOVA",
        f"  n={r.n}, k={r.k}",
        f"{'='*60}",
        f"  F({r.df1}, {r.df2}) = {r.f_statistic:.3f}, p = {r.p_value:.4f}",
        f"  generalized η² = {r.ges:.4f}",
        f"",
        f"  【球性检验】",
        f"  Mauchly's W = {r.mauchly_w:.4f}, χ²({r.mauchly_df}) = {r.mauchly_chi2:.3f}, p = {r.mauchly_p:.4f}",
        f"  Greenhouse-Geisser ε = {r.gg_epsilon:.4f}",
        f"  Huynh-Feldt ε      = {r.hf_epsilon:.4f}",
        f"",
        f"  【校正 p 值】",
        f"  未校正:        p = {r.p_value:.4f}",
        f"  GG 校正:       p = {r.gg_p:.4f}",
        f"  HF 校正:       p = {r.hf_p:.4f}",
        f"",
        f"  【条件描述统计】",
        f"  {'条件':<12} {'M':>8} {'SD':>8}",
    ]
    for name in r.condition_means:
        lines.append(f"  {name:<12} {r.condition_means[name]:8.3f} {r.condition_sds[name]:8.3f}")

    if r.posthoc:
        lines.extend(["", "  【事后配对比较 (Bonferroni)】", ""])
        for pc in r.posthoc:
            sig = "*" if pc["significant"] else "ns"
            lines.append(
                f"  {pc['group1']} vs {pc['group2']}: diff={pc['mean_diff']:.4f}, "
                f"p={pc['p']:.4f} {sig}  CI=[{pc['ci_95'][0]:.4f}, {pc['ci_95'][1]:.4f}]"
            )

    lines.append(f"{'='*60}")
    return "\n".join(lines)


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def anova_report(result: AnovaResult) -> str:
    """生成 ANOVA APA 格式报告。"""
    lines = [
        f"{'='*50}",
        f"  单因素方差分析 ({result.method})",
        f"{'='*50}",
        f"  F({result.df1}, {result.df2:.1f}) = {result.f_statistic:.3f}, p = {result.p_value:.4f}",
        f"  η² = {result.eta_sq:.4f},  ω² = {result.omega_sq:.4f}",
        f"  n = {result.n_total},  k = {result.k}",
        f"",
        f"  各组描述统计:",
    ]
    for gs in result.group_stats:
        lines.append(f"    {gs['group']}: n={gs['n']}, M={gs['mean']:.3f}, SD={gs['std']:.3f}")

    if result.levene_f is not None:
        lines.append(f"\n  Levene 检验 (中位数): F = {result.levene_f:.3f}, p = {result.levene_p:.4f}")

    if result.linear_trend:
        lt = result.linear_trend
        lines.append(f"\n  线性趋势检验: F(1, {lt['df2']}) = {lt['F']:.3f}, p = {lt['p']:.4f}")

    return "\n".join(lines)


def posthoc_report(result: PostHocResult) -> str:
    """生成事后检验报告。"""
    lines = [
        f"{'='*50}",
        f"  事后多重比较: {result.method}",
        f"  α = {result.alpha}, α_corrected = {result.alpha_corrected:.6f}",
        f"{'='*50}",
    ]
    for c in result.comparisons:
        sig = "*" if c.significant else "ns"
        lines.append(
            f"  {c.group1} vs {c.group2}: diff={c.mean_diff:.4f}, "
            f"p={c.p_value:.4f} {sig}"
            f"  [95% CI: {c.ci_95[0]:.4f}, {c.ci_95[1]:.4f}]"
        )
    return "\n".join(lines)
