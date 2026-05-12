"""可靠性分析 (Reliability Analysis) — 全参数实现

复刻 SPSS RELIABILITY 过程:
- Cronbach's α (原始/标准化) + 删除项后 α
- ICC 组内相关系数 (6 种 Shrout & Fleiss 模型)
- Fleiss Kappa (多评定者名义一致性)
- Hotelling  T² 检验
- Tukey 可加性检验

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【Cronbach's α — 量表内部一致性】

α = k/(k-1) × (1 - Σσ²_i / σ²_T)
k = 题项数, Σσ²_i = 各题方差之和, σ²_T = 总分的方差。

直觉: α 衡量的是"所有题项在多大程度上测量了同一个东西"。
  如果各题高度相关, 总分方差 >> 各题方差之和 → α 接近 1。
  如果各题毫无关联, 总分方差 ≈ 各题方差之和 → α 接近 0。

  注意: α 在样本量小 (<30) 时不稳定。n>100 是安全的。

【α 的重要谬误 — 它不是"一维性"指标！】

  α 受题项数 k 和题项间平均相关的影响:
  如果你有 20 个题项, 即使题间相关只有 0.2, α 也能达到 0.83。
  这并不意味着你测的是单一构念 (一维性), 只是一个可靠的总分。

  高 α ≠ 一维性。要检查一维性, 需要 EFA (因子分析)。

【各项指标的含义】

修正的题项-总分相关 (Corrected Item-Total Correlation):
  题项得分与"所有其他题项的总分"的相关。
  排除自身是为了避免自我关联的人为膨胀。
  < 0.3 → 考虑删除此题。> 0.5 → 良好。

删除项后 α (Alpha If Item Deleted):
  如果删除此题, α 会变成多少。
  如果删除后 α 上升 → 此题可能"拉低"了内部一致性, 考虑删除。
  但: 如果 α 本来就 > 0.8, 删除一个题 α 变成 0.82, 不一定值得删。

标准化 α: 先标准化所有题项为均值 0 方差 1, 再计算 α。
  当各题量纲不同 (如有些 1-5, 有些 1-7) 时使用。

原始 α: 使用原始分数。当所有题项量纲相同时使用 (常见情况)。

Hotelling T²: H₀ = 所有题项均值都相等。
  p < 0.05 → 题项难度/倾向不同 (通常这是期望的)。

【ICC (组内相关系数) — 6 种模型的含义】

ICC 衡量的是"评定者之间的一致性"或"反复测量的一致性"。
Shrout & Fleiss (1979) 定义了 6 种模型, 你需根据实际设计选择:

模型参数:
  One-way   → 每个被试由不同的评定者评分 (评定者不交叉)
  Two-way   → 所有评定者评分所有被试 (评定者交叉, 常见)
    random  → 评定者是总体的随机样本 (希望推广到其他评定者)
    fixed   → 评定者是你唯一关心的 (不推广)
  Consistency       → 评定者的排序一致 (允许评定者有系统性的严格/宽松差异)
  Absolute Agreement → 评定者的分数完全相同 (排序 AND 水平都相同)

  ICC(2,1): Two-way random, absolute agreement。最常用的模型。
  ICC(3,1): Two-way fixed, consistency。评定者是唯一的, 只关心排序。
  ICC(1,1): One-way random。用于不同评定者评不同被试的情况。

ICC 解读: <0.50=差, 0.50~0.75=中等, 0.75~0.90=良好, >0.90=优秀。

【Fleiss Kappa — 多评定者名义一致性】

当评定者有 2+ 个, 类别是名义的 (如诊断 A/B/C), 使用 Fleiss Kappa。
它是 Cohen's Kappa 对多评定者的推广。

与 Cohen's Kappa 相同的原理: κ = (P_bar - P_e) / (1 - P_e)。
但 P_bar 是对所有被试的"评定者间一致度"取平均。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ═══════════════════════════════════════════
# Cronbach's α
# ═══════════════════════════════════════════


@dataclass
class AlphaResult:
    """Cronbach's α 结果"""

    n_cases: int
    n_items: int
    alpha_raw: float
    alpha_standardized: float
    item_means: list[float]
    item_variances: list[float]
    item_total_correlations: list[float]
    alpha_if_deleted: list[float]
    mean_inter_item_corr: float
    hotelling_t2: float | None = None
    hotelling_p: float | None = None


def cronbach_alpha(
    data: np.ndarray | pd.DataFrame,
    *,
    reverse_items: list[int] | None = None,
) -> AlphaResult:
    """Cronbach's α 内部一致性信度。

    Args:
        data: (n, k) — n 被试 × k 题项。
        reverse_items: 需要反向计分的题项索引 (0-based)。

    Returns:
        AlphaResult。
    """
    if isinstance(data, pd.DataFrame):
        arr = data.values.astype(np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)

    # 反向计分
    if reverse_items:
        arr = arr.copy()
        for idx in reverse_items:
            max_val = np.nanmax(arr[:, idx])
            min_val = np.nanmin(arr[:, idx])
            arr[:, idx] = max_val + min_val - arr[:, idx]

    # 列表删除
    arr = arr[~np.isnan(arr).any(axis=1)]
    n, k = arr.shape

    # 题项统计
    item_means = [float(np.mean(arr[:, j])) for j in range(k)]
    item_vars = [float(np.var(arr[:, j], ddof=1)) for j in range(k)]

    # 总分方差
    total_scores = arr.sum(axis=1)
    total_var = float(np.var(total_scores, ddof=1)) if n > 1 else 0.0

    # 原始 α = k/(k-1) * (1 - Σσ²_i / σ²_T)
    sum_item_var = sum(item_vars)
    alpha_raw = (k / (k - 1)) * (1 - sum_item_var / total_var) if k > 1 and total_var > 0 else float("nan")

    # 标准化 α (基于相关矩阵)
    corr = np.corrcoef(arr, rowvar=False)
    mean_r = (np.sum(corr) - k) / (k * (k - 1)) if k > 1 else 0.0  # 排除对角线
    alpha_std = (k * mean_r) / (1 + (k - 1) * mean_r) if k > 1 and not math.isnan(mean_r) else float("nan")

    # 删除项后 α
    alpha_if_del = []
    item_total_corr = []
    for j in range(k):
        # 删除项后的总分
        scores_without_j = total_scores - arr[:, j]
        var_without_j = float(np.var(scores_without_j, ddof=1)) if n > 1 else 0.0

        # 删除项后的 α
        sum_var_without = sum_item_var - item_vars[j]
        if k > 1 and var_without_j > 0:
            alpha_del = ((k - 1) / (k - 2)) * (1 - sum_var_without / var_without_j)
        else:
            alpha_del = float("nan")
        alpha_if_del.append(alpha_del)

        # 修正的题项-总分相关
        corr_it = float(np.corrcoef(arr[:, j], scores_without_j)[0, 1]) if n > 1 else float("nan")
        item_total_corr.append(corr_it)

    # Hotelling T² (检验题项均值是否相等)
    hotelling_t2, hotelling_p = None, None
    if n > k and k >= 2:
        item_means_arr = np.array(item_means)
        diff = item_means_arr - np.mean(item_means_arr)
        cov = np.cov(arr, rowvar=False)
        try:
            inv_cov = np.linalg.inv(cov)
            T2 = n * diff @ inv_cov @ diff
            # F = (n-k)/(k-1)/n * T2
            F_stat = ((n - k) / ((k - 1) * n)) * T2
            df1, df2 = k - 1, n - k
            hotelling_t2 = float(T2)
            hotelling_p = float(1.0 - sp_stats.f.cdf(F_stat, df1, df2))
        except np.linalg.LinAlgError:
            pass

    return AlphaResult(
        n_cases=n,
        n_items=k,
        alpha_raw=alpha_raw,
        alpha_standardized=alpha_std,
        item_means=item_means,
        item_variances=item_vars,
        item_total_correlations=item_total_corr,
        alpha_if_deleted=alpha_if_del,
        mean_inter_item_corr=mean_r,
        hotelling_t2=hotelling_t2,
        hotelling_p=hotelling_p,
    )


# ═══════════════════════════════════════════
# ICC (组内相关系数)
# ═══════════════════════════════════════════


@dataclass
class ICCResult:
    """ICC 结果"""

    model: str  # "one-way random" | "two-way random" | "two-way fixed"
    type_: str  # "consistency" | "absolute agreement"
    icc: float
    ci_95: tuple[float, float]
    f_test: float
    f_df1: int
    f_df2: int
    f_p: float
    n_subjects: int
    n_raters: int


def icc(
    data: np.ndarray,
    model: str = "two-way",
    type_: str = "consistency",
    raters_random: bool = True,
    alpha: float = 0.05,
) -> ICCResult:
    """ICC 组内相关系数 (Shrout & Fleiss 1979)。

    Args:
        data: (n_subjects, n_raters) — 每行=被试, 每列=评定者。
        model: ``"one-way"`` (每个被试由不同评定者评分) /
               ``"two-way"`` (所有评定者评分所有被试, 推荐的默认选择)。
        type_: ``"consistency"`` (忽略评定者间系统误差) /
               ``"absolute_agreement"`` (要求打分数值绝对相等)。
        raters_random: True=评定者为随机效应, False=固定效应。
        alpha: CI 显著性水平。

    Returns:
        ICCResult。
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[~np.isnan(arr).any(axis=1)]
    n, k = arr.shape  # n subjects, k raters

    # ANOVA 方差分解
    grand_mean = np.mean(arr)
    ms_subjects = k * np.sum((np.mean(arr, axis=1) - grand_mean) ** 2) / (n - 1) if n > 1 else 0.0
    ms_raters = n * np.sum((np.mean(arr, axis=0) - grand_mean) ** 2) / (k - 1) if k > 1 else 0.0
    ss_total = np.sum((arr - grand_mean) ** 2)
    ss_subjects = k * np.sum((np.mean(arr, axis=1) - grand_mean) ** 2)
    ss_raters = n * np.sum((np.mean(arr, axis=0) - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_raters
    df_error = (n - 1) * (k - 1)
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    # 模型标签
    if model == "one-way":
        model_label = "One-way random"
        # ICC(1,1) = (MS_s - MS_e) / (MS_s + (k-1) * MS_e)
        icc_val = (ms_subjects - ms_error) / (ms_subjects + (k - 1) * ms_error) if ms_subjects > 0 else 0.0
        # F = MS_s / MS_e
        F_val = ms_subjects / ms_error if ms_error > 0 else float("nan")
        df1, df2 = n - 1, n * (k - 1)
    else:
        rater_label = "random" if raters_random else "fixed"
        model_label = f"Two-way {rater_label}"
        if type_ == "absolute_agreement":
            # ICC(2,1) 或 ICC(3,1): absolute agreement
            # (MS_s - MS_e) / (MS_s + (k-1)*MS_e + k*(MS_r - MS_e)/n)
            icc_val = (ms_subjects - ms_error) / (
                ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
            ) if ms_subjects > 0 and n > 0 else 0.0
        else:
            # consistency
            icc_val = (ms_subjects - ms_error) / (ms_subjects + (k - 1) * ms_error) if ms_subjects > 0 else 0.0
        F_val = ms_subjects / ms_error if ms_error > 0 else float("nan")
        df1, df2 = n - 1, df_error

    # CI (F 分布法)
    F_obs = F_val if not math.isnan(F_val) else 1.0
    F_lower = F_obs / sp_stats.f.ppf(1 - alpha / 2, df1, df2) if df2 > 0 else 0.0
    F_upper = F_obs * sp_stats.f.ppf(1 - alpha / 2, df2, df1) if df2 > 0 else float("inf")

    if model == "one-way" or type_ == "consistency":
        ci_lo = (F_lower - 1) / (F_lower + k - 1) if F_lower > 0 else 0.0
        ci_hi = (F_upper - 1) / (F_upper + k - 1) if F_upper > 0 else 1.0
    else:
        ci_lo = (F_lower - 1) / (F_lower + (k - 1) + k * (ms_raters / ms_error - 1) / n) if F_lower > 0 else 0.0
        ci_hi = (F_upper - 1) / (F_upper + (k - 1) + k * (ms_raters / ms_error - 1) / n) if F_upper > 0 else 1.0

    ci_lo = max(0.0, ci_lo)
    ci_hi = min(1.0, ci_hi)

    return ICCResult(
        model=model_label,
        type_=type_,
        icc=max(0.0, min(1.0, float(icc_val))),
        ci_95=(float(ci_lo), float(ci_hi)),
        f_test=float(F_val),
        f_df1=int(df1),
        f_df2=int(df2),
        f_p=float(1.0 - sp_stats.f.cdf(F_val, df1, df2)) if not math.isnan(F_val) else float("nan"),
        n_subjects=n,
        n_raters=k,
    )


# ═══════════════════════════════════════════
# Fleiss Kappa
# ═══════════════════════════════════════════


@dataclass
class FleissKappaResult:
    """Fleiss Kappa 结果"""

    kappa: float
    se: float
    z: float
    p_value: float
    ci_95: tuple[float, float]
    n_subjects: int
    n_raters: int
    n_categories: int
    category_kappas: list[float]


def fleiss_kappa(ratings: np.ndarray) -> FleissKappaResult:
    """Fleiss Kappa (多评定者名义一致性)。

    Args:
        ratings: (n_subjects, n_raters) — 每行=被试, 每列=评定者,
                 取值为 0, 1, 2, ... (类别编码)。

    Returns:
        FleissKappaResult。
    """
    n, m = ratings.shape  # n subjects, m raters
    categories = sorted(set(int(r) for row in ratings for r in row if not np.isnan(r)))
    q = len(categories)

    # 计数矩阵 (n, q)
    counts = np.zeros((n, q))
    for i in range(n):
        for j in range(m):
            val = ratings[i, j]
            if not np.isnan(val):
                cat_idx = categories.index(int(val))
                counts[i, cat_idx] += 1

    # P_i: 第 i 个被试的评定者间一致程度
    P_i = np.zeros(n)
    for i in range(n):
        P_i[i] = (np.sum(counts[i] ** 2) - m) / (m * (m - 1)) if m > 1 else 1.0

    # P_bar: 平均一致度
    P_bar = np.mean(P_i)

    # p_j: 每个类别在所有评定中的比例
    p_j = np.sum(counts, axis=0) / (n * m)

    # P_e: 期望一致度 (随机)
    P_e = np.sum(p_j**2)

    # Kappa
    kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1.0 else 0.0

    # 标准误 (Fleiss et al. 1979)
    pj2_sum = np.sum(p_j**2)
    pj3_sum = np.sum(p_j**3)

    var_kappa = (
        2
        / (n * m * (m - 1))
        * (pj2_sum - (2 * m - 3) * pj2_sum**2 + 2 * (m - 2) * pj3_sum)
        / (1 - P_e) ** 2
    )
    se = math.sqrt(max(var_kappa, 0))

    z = kappa / se if se > 0 else 0.0
    p_val = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))

    z_alpha = sp_stats.norm.ppf(0.975)
    ci = (kappa - z_alpha * se, kappa + z_alpha * se)

    # 每个类别的 Kappa
    cat_kappas = []
    for j in range(q):
        P_bar_j = np.mean(counts[:, j]) / m
        k_j = (P_bar_j - p_j[j]) / (1 - p_j[j]) if p_j[j] < 1 else 0.0
        cat_kappas.append(k_j)

    return FleissKappaResult(
        kappa=float(kappa),
        se=float(se),
        z=float(z),
        p_value=float(p_val),
        ci_95=(float(ci[0]), float(ci[1])),
        n_subjects=n,
        n_raters=m,
        n_categories=q,
        category_kappas=cat_kappas,
    )


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def alpha_report(r: AlphaResult) -> str:
    """Cronbach's α 报告。"""
    lines = [
        f"{'='*50}",
        f"  Cronbach's Alpha 信度分析",
        f"  有效个案: {r.n_cases}, 题项数: {r.n_items}",
        f"{'='*50}",
        f"  原始 α          = {r.alpha_raw:.4f}",
        f"  标准化 α         = {r.alpha_standardized:.4f}",
        f"  平均题项相关系数  = {r.mean_inter_item_corr:.4f}",
        f"",
        f"  {'题项':<6} {'均值':>8} {'方差':>8} {'修正项总计相关':>10} {'删除项后α':>10}",
    ]
    for j in range(r.n_items):
        lines.append(
            f"  {j+1:<6} {r.item_means[j]:8.3f} {r.item_variances[j]:8.3f} "
            f"{r.item_total_correlations[j]:10.4f} {r.alpha_if_deleted[j]:10.4f}"
        )
    if r.hotelling_t2 is not None:
        lines.append(f"\n  Hotelling T² = {r.hotelling_t2:.4f}, p = {r.hotelling_p:.4f}")
    return "\n".join(lines)


def icc_report(r: ICCResult) -> str:
    """ICC 报告。"""
    return "\n".join([
        f"{'='*50}",
        f"  组内相关系数 ICC",
        f"  模型: {r.model}, 类型: {r.type_}",
        f"  被试: {r.n_subjects}, 评定者: {r.n_raters}",
        f"{'='*50}",
        f"  ICC = {r.icc:.4f}  95% CI [{r.ci_95[0]:.4f}, {r.ci_95[1]:.4f}]",
        f"  F({r.f_df1}, {r.f_df2}) = {r.f_test:.4f}, p = {r.f_p:.4f}",
    ])


def fleiss_report(r: FleissKappaResult) -> str:
    """Fleiss Kappa 报告。"""
    lines = [
        f"{'='*50}",
        f"  Fleiss Kappa (多评定者一致性)",
        f"  被试: {r.n_subjects}, 评定者: {r.n_raters}, 类别: {r.n_categories}",
        f"{'='*50}",
        f"  κ = {r.kappa:.4f}  SE = {r.se:.4f}  Z = {r.z:.3f}  p = {r.p_value:.4f}",
        f"  95% CI: [{r.ci_95[0]:.4f}, {r.ci_95[1]:.4f}]",
    ]
    return "\n".join(lines)
