"""非参数检验 (Nonparametric Tests) — 全参数实现

复刻 SPSS NPTESTS / NPAR TESTS 过程:
- 两独立样本: Mann-Whitney U, Kolmogorov-Smirnov 两样本
- k 独立样本: Kruskal-Wallis, Jonckheere-Terpstra 趋势检验
- 两相关样本: Wilcoxon signed-rank, Sign test, McNemar
- k 相关样本: Friedman, Cochran's Q
- 效应量: Rank-biserial r, η²_H, Kendall's W, Cliff's Delta, Hodges-Lehmann

══════════════════════════════════════════════════════════════════════
统计概念速查 — 非参数检验的哲学
══════════════════════════════════════════════════════════════════════

【为什么需要非参数检验】

参数检验 (t, ANOVA) 假定:
1. 数据来自正态分布 (至少近似)
2. 多组间方差相等

当这些假设不成立, 或者你有离群值, 或者数据是顺序尺度的 (如李克特量表
1-5分), 非参数方法给出了"放弃分布假设"之后的备选方案。

核心思路: 不看原始值, 看"秩"(rank)。把最小的标为 1, 第二小的标为 2...
然后对"秩"做推断。

【方法与参数等效对照】

| 参数检验          | 非参数等效               | 检验什么                    |
|------------------|-------------------------|-----------------------------|
| 独立 t 检验       | Mann-Whitney U           | 两组的分布是否相同           |
| 配对 t 检验       | Wilcoxon Signed-Rank     | 两组差值的分布是否对称于0    |
| 单因素 ANOVA      | Kruskal-Wallis H         | 多组的中位数是否来自同一分布  |
| 重复测量 ANOVA    | Friedman                 | 多个处理的秩和是否相同       |
| Pearson 相关      | Spearman ρ               | 单调相关 (不要求线性)        |
| 配对 χ²           | McNemar                  | 配对二分类的边际齐性         |

【Mann-Whitney U 到底在检验什么】

很多人说 M-W "比较两个中位数"。这不完全对。
M-W 检验的是"随机从 A 组抽一个值, 比从 B 组随机抽的值大的概率是否为 0.5"。
这被称为"随机优势" (stochastic dominance)。

换句话说: 如果两组分布形状相同只是位置不同, M-W 确实是比较中位数。
但如果两组分布形状不同 (如一方差性), M-W 可能显著但中位数相同。
解读时始终检查两组的中位数 + IQR, 不能只看 p 值。

效应量: Rank-biserial r = 1 - 2U/(n1×n2)
  这是 U 统计量与最大值之比的转换。r=0 → 无差异, |r|=1 → 完全分离。
  另提供 Cliff's Delta (非参数版的 Cohen's d), 两者在分布形状相同时等价。

【Hodges-Lehmann 估计】

两独立样本: 所有可能的 X_i - Y_j 配对差值的中位数。
配对样本: 所有 (d_i + d_j)/2 (Walsh averages) 的中位数。

这是非参数版的"均值差"点估计。它估计的是如果两组分布完全相同
只是平移了, 这个平移量是多少。

【Wilcoxon Signed-Rank 的假设】
  H₀: 差值的分布关于 0 对称。
  如果差值不对称但中位数=0, Wilcoxon 可能给出错误结果。
  此时用 Sign test (没有对称假设, 但效力更低, 只看正负号)。

【Friedman vs Kruskal-Wallis】
  K-W 用于独立设计的 k 组 (like 学生来自 k 个不同班级)
  Friedman 用于重复测量/区组设计 (like 同一个学生在 k 种条件下各测一次)
  两者都基于秩, Friedman 在"被试内部"排秩, K-W 在"总数据"排秩。

【Jonckheere-Terpstra 趋势检验】
  当分组有自然顺序 (低/中/高) 且你预期有单调趋势时使用。
  比 K-W 更专门化, 统计效力更高 (因为用了"有序"信息)。
  J 统计量 = 累加所有成对比较中"低组值 < 高组值"的次数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════


@dataclass
class NonparametricResult:
    """非参数检验结果"""

    method: str
    statistic: float
    p_value: float
    n: int
    # 描述统计
    medians: dict[str, float] = field(default_factory=dict)
    iqrs: dict[str, tuple[float, float]] = field(default_factory=dict)
    # 效应量
    effect_size: float = float("nan")
    effect_label: str = ""
    effect_method: str = ""
    cliffs_delta: float = float("nan")  # Cliff's Delta (仅 Mann-Whitney)
    # Hodges-Lehmann
    hl_estimate: float = float("nan")
    hl_ci_95: tuple[float, float] | None = None
    # 事后比较 (K-W / Friedman)
    posthoc: list[dict] | None = None


# ═══════════════════════════════════════════
# Mann-Whitney U
# ═══════════════════════════════════════════


def mann_whitney(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    alternative: str = "two-sided",
    method: str = "auto",
) -> NonparametricResult:
    """Mann-Whitney U 检验 (两独立样本)。

    Args:
        x1, x2: 两组独立样本。
        alternative: ``"two-sided"`` / ``"greater"`` / ``"less"``。
        method: ``"auto"`` (n<50 exact) / ``"exact"`` / ``"asymptotic"``。

    Returns:
        NonparametricResult。
    """
    a1 = np.asarray(x1, dtype=np.float64)
    a2 = np.asarray(x2, dtype=np.float64)
    a1, a2 = a1[~np.isnan(a1)], a2[~np.isnan(a2)]

    n1, n2 = len(a1), len(a2)
    n = n1 + n2

    res = sp_stats.mannwhitneyu(a1, a2, alternative=alternative, method=method)
    U = float(res.statistic)
    p = float(res.pvalue)

    # 效应量: rank-biserial correlation r
    r = 1 - 2 * U / (n1 * n2) if n1 > 0 and n2 > 0 else float("nan")

    # Cliff's Delta
    # 计算 A 组每个值 > B 组每个值的比例
    cliff = _cliffs_delta(a1, a2)

    # Hodges-Lehmann 估计 (位置漂移)
    hl = _hodges_lehmann_2sample(a1, a2)

    med1 = float(np.median(a1))
    med2 = float(np.median(a2))
    q1_1, q3_1 = float(np.percentile(a1, 25)), float(np.percentile(a1, 75))
    q1_2, q3_2 = float(np.percentile(a2, 25)), float(np.percentile(a2, 75))

    r_label = "小" if abs(r) < 0.3 else ("中" if abs(r) < 0.5 else "大")

    return NonparametricResult(
        method="Mann-Whitney U",
        statistic=U,
        p_value=p,
        n=n,
        medians={"组1": med1, "组2": med2},
        iqrs={"组1": (q1_1, q3_1), "组2": (q1_2, q3_2)},
        effect_size=r,
        effect_label=f"{r_label} (rank-biserial r)",
        effect_method="rank-biserial r",
        cliffs_delta=cliff,
        hl_estimate=hl,
    )


def _cliffs_delta(x1: np.ndarray, x2: np.ndarray) -> float:
    """Cliff's Delta 效应量 (非参数版的 Cohen's d)。"""
    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    # 计算支配矩阵
    greater = sum(1 for a in x1 for b in x2 if a > b)
    less = sum(1 for a in x1 for b in x2 if a < b)
    return (greater - less) / (n1 * n2)


def _hodges_lehmann_2sample(x1: np.ndarray, x2: np.ndarray) -> float:
    """Hodges-Lehmann 两样本位置漂移估计 (所有配对差值的中位数)。"""
    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    diffs = []
    for a in x1:
        for b in x2:
            diffs.append(a - b)
    return float(np.median(diffs))


# ═══════════════════════════════════════════
# Kruskal-Wallis
# ═══════════════════════════════════════════


def kruskal_wallis(
    groups: dict[str, np.ndarray],
) -> NonparametricResult:
    """Kruskal-Wallis H 检验 (k 独立样本)。

    Args:
        groups: {组名: 数组}。

    Returns:
        NonparametricResult 含效应量 η²_H 和成对事后比较。
    """
    group_list = [np.asarray(a, dtype=np.float64)[~np.isnan(a)] for a in groups.values()]
    names = list(groups.keys())
    k = len(group_list)

    H, p = sp_stats.kruskal(*group_list)
    H, p = float(H), float(p)

    # 效应量 η²_H
    n_total = sum(len(a) for a in group_list)
    eta_sq_H = (H - k + 1) / (n_total - k) if n_total > k else float("nan")

    # 描述统计
    medians = {}
    iqrs = {}
    for name, arr in groups.items():
        a = arr[~np.isnan(arr)]
        medians[name] = float(np.median(a))
        iqrs[name] = (float(np.percentile(a, 25)), float(np.percentile(a, 75)))

    # 事后成对比较 (Dunn 检验 with Bonferroni)
    posthoc_comps = _dunn_posthoc(groups)

    return NonparametricResult(
        method="Kruskal-Wallis H",
        statistic=H,
        p_value=p,
        n=n_total,
        medians=medians,
        iqrs=iqrs,
        effect_size=eta_sq_H,
        effect_label=f"η²_H = {eta_sq_H:.4f}",
        effect_method="η²_H",
        posthoc=posthoc_comps,
    )


def _dunn_posthoc(groups: dict[str, np.ndarray]) -> list[dict]:
    """Dunn 事后检验 (K-W 后的成对比较, 带 Bonferroni 校正)。"""
    names = list(groups.keys())
    k = len(names)
    if k < 2:
        return []

    # 合并数据 + 编秩
    all_data = []
    all_groups = []
    for i, (name, arr) in enumerate(groups.items()):
        a = np.asarray(arr, dtype=np.float64)
        a = a[~np.isnan(a)]
        all_data.append(a)
        all_groups.extend([i] * len(a))

    combined = np.concatenate(all_data)
    ranks = sp_stats.rankdata(combined)

    # 每组秩和
    rank_sums = {}
    ns = {}
    for i, name in enumerate(names):
        mask = np.array(all_groups) == i
        rank_sums[name] = ranks[mask].sum()
        ns[name] = mask.sum()

    N = len(combined)
    n_comparisons = k * (k - 1) / 2

    comparisons = []
    for i in range(k):
        for j in range(i + 1, k):
            ni, nj = ns[names[i]], ns[names[j]]
            ri_bar = rank_sums[names[i]] / ni if ni > 0 else 0
            rj_bar = rank_sums[names[j]] / nj if nj > 0 else 0

            # Dunn's z
            se = math.sqrt((N * (N + 1) / 12) * (1.0 / ni + 1.0 / nj))
            z = (ri_bar - rj_bar) / se if se > 0 else 0.0

            # Bonferroni 校正
            p_raw = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
            p_corrected = min(p_raw * n_comparisons, 1.0)

            comparisons.append(
                {
                    "group1": names[i],
                    "group2": names[j],
                    "z": float(z),
                    "p_raw": float(p_raw),
                    "p_corrected": float(p_corrected),
                    "significant": p_corrected < 0.05,
                }
            )

    return comparisons


# ═══════════════════════════════════════════
# Wilcoxon Signed-Rank
# ═══════════════════════════════════════════


def wilcoxon_signed_rank(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    alternative: str = "two-sided",
    method: str = "auto",
) -> NonparametricResult:
    """Wilcoxon 符号秩检验 (两相关/配对样本)。

    Args:
        x1, x2: 配对数据 (等长)。
    """
    a1 = np.asarray(x1, dtype=np.float64)
    a2 = np.asarray(x2, dtype=np.float64)
    mask = (~np.isnan(a1)) & (~np.isnan(a2))
    a1, a2 = a1[mask], a2[mask]
    n = len(a1)

    res = sp_stats.wilcoxon(a1, a2, alternative=alternative, method=method, correction=True)
    W = float(res.statistic) if hasattr(res, "statistic") else float("nan")
    p = float(res.pvalue)

    # 效应量: r = Z / sqrt(n)
    # 先计算 Z 近似
    diffs = a1 - a2
    non_zero = diffs[diffs != 0]
    n_nz = len(non_zero)
    if n_nz > 0:
        abs_ranks = sp_stats.rankdata(np.abs(non_zero))
        signed_ranks = np.sign(non_zero) * abs_ranks
        W_val = signed_ranks[signed_ranks > 0].sum()
        mean_W = n_nz * (n_nz + 1) / 4
        se_W = math.sqrt(n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24)
        z = (W_val - mean_W) / se_W if se_W > 0 else 0.0
    else:
        z = 0.0

    r = abs(z) / math.sqrt(n) if n > 0 else float("nan")

    # Hodges-Lehmann
    hl = _hodges_lehmann_paired(a1, a2)

    med1 = float(np.median(a1))
    med2 = float(np.median(a2))

    return NonparametricResult(
        method="Wilcoxon Signed-Rank",
        statistic=float(W),
        p_value=p,
        n=n,
        medians={"前测": med1, "后测": med2},
        iqrs={
            "前测": (float(np.percentile(a1, 25)), float(np.percentile(a1, 75))),
            "后测": (float(np.percentile(a2, 25)), float(np.percentile(a2, 75))),
        },
        effect_size=r,
        effect_label=f"{'小' if r < 0.3 else ('中' if r < 0.5 else '大')} (|Z|/√n)",
        effect_method="|Z|/√n",
        hl_estimate=hl,
    )


def _hodges_lehmann_paired(x1: np.ndarray, x2: np.ndarray) -> float:
    """配对 Hodges-Lehmann: Walsh averages of differences 的中位数。"""
    diffs = x1 - x2
    n = len(diffs)
    if n == 0:
        return float("nan")
    # Walsh averages
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((diffs[i] + diffs[j]) / 2.0)
    return float(np.median(walsh))


# ═══════════════════════════════════════════
# Sign Test
# ═══════════════════════════════════════════


def sign_test(
    x1: np.ndarray,
    x2: np.ndarray,
) -> NonparametricResult:
    """符号检验 (配对的 nonparametric, 不需要差值对称)。"""
    a1 = np.asarray(x1, dtype=np.float64)
    a2 = np.asarray(x2, dtype=np.float64)
    mask = (~np.isnan(a1)) & (~np.isnan(a2))
    a1, a2 = a1[mask], a2[mask]

    diffs = a1 - a2
    diffs = diffs[diffs != 0]
    n = len(diffs)

    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))

    # 二项检验
    from scipy.stats import binomtest

    res = binomtest(min(n_pos, n_neg) if n_pos != n_neg else n_pos, n=n_pos + n_neg, p=0.5, alternative="two-sided")

    return NonparametricResult(
        method="Sign Test",
        statistic=float(max(n_pos, n_neg)),
        p_value=float(res.pvalue),
        n=n,
        medians={
            "正差值": float(n_pos),
            "负差值": float(n_neg),
        },
        iqrs={},
        effect_size=n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else float("nan"),
        effect_label="正差值比例",
        effect_method="prop_positive",
    )


# ═══════════════════════════════════════════
# Friedman
# ═══════════════════════════════════════════


def friedman(
    data: np.ndarray,
) -> NonparametricResult:
    """Friedman 检验 (k 相关样本)。

    Args:
        data: (n_subjects, k_treatments) 形状的 2D 数组, 每行一个被试, 每列一个处理。

    Returns:
        NonparametricResult。
    """
    arr = np.asarray(data, dtype=np.float64)
    # 移除含 NaN 的行
    arr = arr[~np.isnan(arr).any(axis=1)]
    n_subjects, k = arr.shape

    chi2, p = sp_stats.friedmanchisquare(*[arr[:, j] for j in range(k)])
    chi2, p = float(chi2), float(p)

    # Kendall's W
    # W = chi2 / (n * (k - 1))
    W = chi2 / (n_subjects * (k - 1)) if n_subjects > 0 and k > 1 else float("nan")

    # 各处理中位数
    medians = {f"处理{j+1}": float(np.median(arr[:, j])) for j in range(k)}
    iqrs = {
        f"处理{j+1}": (float(np.percentile(arr[:, j], 25)), float(np.percentile(arr[:, j], 75)))
        for j in range(k)
    }

    return NonparametricResult(
        method="Friedman",
        statistic=chi2,
        p_value=p,
        n=n_subjects,
        medians=medians,
        iqrs=iqrs,
        effect_size=W,
        effect_label=f"Kendall's W = {W:.4f}",
        effect_method="Kendall's W",
    )


# ═══════════════════════════════════════════
# Jonckheere-Terpstra 趋势检验
# ═══════════════════════════════════════════


def jonckheere_terpstra(
    groups: dict[str, np.ndarray],
    *,
    alternative: str = "increasing",
) -> NonparametricResult:
    """Jonckheere-Terpstra 趋势检验 (有序分组下的单调趋势检验)。

    Args:
        groups: 有序的 {组名: 数组} (按组别顺序)。
        alternative: ``"increasing"`` (递增趋势) / ``"decreasing"`` / ``"two-sided"``。

    Returns:
        NonparametricResult。
    """
    names = list(groups.keys())
    group_list = [np.asarray(groups[n], dtype=np.float64) for n in names]
    group_list = [g[~np.isnan(g)] for g in group_list]
    k = len(group_list)

    # 计算 J-T 统计量
    # J = sum_{i<j} U_{ij}, where U_{ij} = count(x_i < x_j) + 0.5 * count(x_i == x_j)
    J = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            for xi in group_list[i]:
                for xj in group_list[j]:
                    if xi < xj:
                        J += 1.0
                    elif xi == xj:
                        J += 0.5

    # 渐近正态检验
    ns = [len(g) for g in group_list]
    N = sum(ns)
    E_J = (N**2 - sum(n**2 for n in ns)) / 4
    # 方差 (Terpstra 1988)
    A = sum(n * (n - 1) * (2 * n + 5) for n in ns)
    B = sum(n * (n - 1) * (n - 2) for n in ns)
    C = sum(n * (n - 1) for n in ns)
    D = N * (N - 1) * (2 * N + 5)
    var_J = (D - A - B) / 72 + C * (N - 2) / 36

    z = (J - E_J) / math.sqrt(var_J) if var_J > 0 else 0.0

    if alternative == "increasing":
        p = 1.0 - sp_stats.norm.cdf(z)
    elif alternative == "decreasing":
        p = sp_stats.norm.cdf(z)
    else:
        p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))

    medians = {name: float(np.median(g)) for name, g in zip(names, group_list)}

    return NonparametricResult(
        method="Jonckheere-Terpstra",
        statistic=float(J),
        p_value=float(p),
        n=N,
        medians=medians,
        effect_size=float(z),
        effect_label=f"Z = {z:.4f}",
        effect_method="标准正态 Z",
    )


# ═══════════════════════════════════════════
# Kolmogorov-Smirnov 两样本
# ═══════════════════════════════════════════


def ks_two_sample(
    x1: np.ndarray,
    x2: np.ndarray,
) -> NonparametricResult:
    """K-S 两样本检验 (对位置、尺度和形状的任何差异都敏感)。"""
    a1 = np.asarray(x1, dtype=np.float64)
    a2 = np.asarray(x2, dtype=np.float64)
    a1, a2 = a1[~np.isnan(a1)], a2[~np.isnan(a2)]

    res = sp_stats.ks_2samp(a1, a2)
    D = float(res.statistic)
    p = float(res.pvalue)

    return NonparametricResult(
        method="Kolmogorov-Smirnov (两样本)",
        statistic=D,
        p_value=p,
        n=len(a1) + len(a2),
        medians={"组1": float(np.median(a1)), "组2": float(np.median(a2))},
        effect_size=D,
        effect_label=f"最大垂直距离 D = {D:.4f}",
        effect_method="D (KS distance)",
    )


# ═══════════════════════════════════════════
# McNemar
# ═══════════════════════════════════════════


def mcnemar(
    before: np.ndarray,
    after: np.ndarray,
    *,
    correction: bool = True,
) -> NonparametricResult:
    """McNemar 检验 (配对二分类数据的边际齐性检验)。

    Args:
        before, after: 二元配对数据 (0/1 或 True/False)。
        correction: 是否使用连续性校正 (Edward 校正)。
    """
    a1 = np.asarray(before, dtype=bool)
    a2 = np.asarray(after, dtype=bool)
    mask = (~np.isnan(before.astype(float))) & (~np.isnan(after.astype(float)))
    a1, a2 = a1[mask], a2[mask]

    # 2x2 列联表
    b = int(np.sum(a1 & ~a2))  # before=1, after=0
    c = int(np.sum(~a1 & a2))  # before=0, after=1

    # McNemar χ² = (b - c)² / (b + c)
    if correction:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    else:
        chi2 = (b - c) ** 2 / (b + c) if (b + c) > 0 else 0.0

    p = 1.0 - sp_stats.chi2.cdf(chi2, 1) if (b + c) > 0 else 1.0

    return NonparametricResult(
        method="McNemar" + (" (连续性校正)" if correction else ""),
        statistic=float(chi2),
        p_value=float(p),
        n=len(a1),
        medians={"b (1→0)": float(b), "c (0→1)": float(c)},
        effect_size=float(b) / (b + c) if (b + c) > 0 else float("nan"),
        effect_label=f"不匹配比例: b/(b+c) = {b/(b+c):.3f}" if (b + c) > 0 else "",
        effect_method="边际不对称比",
    )


# ═══════════════════════════════════════════
# Cochran's Q
# ═══════════════════════════════════════════


def cochran_q(
    data: np.ndarray,
) -> NonparametricResult:
    """Cochran's Q 检验 (k 轮重复测量的二元变量, McNemar 的 k 轮推广)。

    Args:
        data: (n_subjects, k_measures) 2D 数组, 每行为一个被试的 k 次 0/1 测量。
    """
    arr = np.asarray(data, dtype=bool)
    arr = arr[~np.isnan(arr.astype(float)).any(axis=1)]
    n, k = arr.shape

    # Q = (k-1) * [k * sum(C_j²) - (sum C_j)²] / [k * sum(R_i) - sum(R_i²)]
    C = arr.sum(axis=0)  # 每列 (处理) 的 1 的个数
    R = arr.sum(axis=1)  # 每行 (被试) 的 1 的个数

    sum_C = C.sum()
    sum_C2 = (C**2).sum()
    sum_R = R.sum()
    sum_R2 = (R**2).sum()

    if k * sum_R - sum_R2 == 0:
        Q = 0.0
    else:
        Q = (k - 1) * (k * sum_C2 - sum_C**2) / (k * sum_R - sum_R2)

    p = 1.0 - sp_stats.chi2.cdf(Q, k - 1) if (k - 1) > 0 else 1.0

    proportions = {f"处理{j+1}": float(C[j] / n) for j in range(k)}

    return NonparametricResult(
        method="Cochran's Q",
        statistic=float(Q),
        p_value=float(p),
        n=n,
        medians=proportions,
        effect_size=float(Q) / n if n > 0 else float("nan"),
        effect_label=f"Q/n = {Q/n:.4f}" if n > 0 else "",
        effect_method="Q/n",
    )
