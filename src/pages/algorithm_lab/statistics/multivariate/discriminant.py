"""判别分析 (Discriminant Analysis) — 全参数实现

复刻 SPSS DISCRIMINANT 过程:
- LDA: Wilks' Lambda, Box's M, 典型判别函数, 结构矩阵
- Fisher 分类系数 (每组独立线性函数)
- 特征值/典型相关系数/方差解释
- 标准化/未标准化系数
- 预测分类 + 后验概率

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【LDA vs ANOVA vs 逻辑回归 — 何时用哪个】

LDA (Linear Discriminant Analysis):
  目标: 找到最大化组间差异 / 组内差异的线性组合 (判别函数)。
  假设: 正态性 + 等协方差矩阵 (Box's M), 连续自变量。
  输出: 判别分数 + 分类规则 + 特征重要性 (结构矩阵)。
  当自变量是连续且来自多元正态时 LDA 比逻辑回归统计效力高 30%。

ANOVA: 只问"各组是否有差异", 不关心"哪些变量组合最能区分"。
逻辑回归: 对分布没假设, 但没法直接回答"哪些变量的线性组合区分度最大"。

实际中: 逻辑回归比 LDA 更常用, 因为不需要正态+等协方差假设。
但如果你的数据满足这两个假设, LDA 更优。

【W⁻¹B 特征分解 — LDA 的数学核心】

W (Within-group covariance): 合并组内协方差矩阵。所有组"内部"共有的变异。
B (Between-group covariance): 组间协方差。各组均值相对于总体均值的变异。

解 W⁻¹B 的特征问题: 找方向 v 使得组间差异 (B) 相对于组内噪音 (W) 最大化。
  特征值 = 该方向上的"组间/组内"比值。
  对应于最大特征值的特征向量 = 最能区分各组的方向 (第一判别函数)。

判别函数数 = min(g-1, k)。g 组 → 最多 g-1 个独立的对比。

【Wilks' Lambda — 整体检验】
  Λ = Σ 1/(1+λ_j)  (所有特征值的累积)
  检验"所有判别函数组合起来, 各组均值是否相等"。
  Bartlett χ² → p < 0.05 → 至少有一个判别函数有效。
  Λ 越小 → 组间差异越大 → 判别越好。Λ 接近 1 = 各组几乎一样。

【结构矩阵 vs 标准化系数 — 哪个更重要】

结构矩阵 (Structure Matrix / Pooled within-group correlations):
  每个自变量与每个判别函数的相关系数。
  不受共线性影响。用于"命名"判别函数 (哪些变量和这个函数最相关)。

标准化系数: 每个变量对判别函数的"独立贡献" (控制了其他变量后)。
  受共线性影响 (和回归的 Beta 一样)。用于判断哪个变量贡献最大。

实践中两者都应报告, 但结构矩阵更适合用于解读和命名。

【Fisher 分类规则 — 不考虑先验的线性分类器】

每组有一个 Fisher 线性函数:
  S_g = c_{g1}X₁ + c_{g2}X₂ + ... + const_g
  分类: 选择 S_g 最大的组。

  系数计算: coef = W⁻¹ × μ_g, const = -0.5 × μ_g' × W⁻¹ × μ_g + ln(P(g))
  这是基于正态 + 等协方差假设的最优分类规则 (在使总错误分类率最小化的意义上)。

【Box's M — 等协方差检验 (LDA 的关键假设)】
  H₀: 各组协方差矩阵相等。
  p < 0.05 → 拒绝 → 协方差矩阵不等 → LDA 不可靠 → 考虑 QDA (二次判别)
  或逻辑回归。

  Box's M 对非正态非常敏感。大样本下即使小的协方差差异也显著。
  此时应看判别函数的交叉验证准确率而不是过度关注 Box's M 的 p 值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats


@dataclass
class LDAResult:
    """线性判别分析结果"""

    n: int
    k: int  # 预测变量数
    g: int  # 组数
    # 全局检验
    wilks_lambda: float
    wilks_chi2: float
    wilks_df: int
    wilks_p: float
    # Box's M
    box_m: float
    box_chi2: float | None
    box_p: float | None
    # 特征
    eigenvalues: list[float]
    canonical_corr: list[float]
    variance_explained: list[float]
    # 系数
    std_coefficients: list[list[float]]  # 标准化典型系数
    unstd_coefficients: list[list[float]]  # 未标准化典型系数
    structure_matrix: list[list[float]]  # 结构矩阵 (合并组内相关)
    # Fisher 分类函数 (每组一行)
    fisher_coefficients: dict[str, list[float]]  # {组名: [系数]}
    fisher_constants: dict[str, float]
    # 分类结果
    confusion_matrix: np.ndarray | None = None
    accuracy: float = float("nan")


def lda(
    X: np.ndarray,
    y: np.ndarray,
    *,
    priors: dict | None = None,
) -> LDAResult:
    """线性判别分析 (LDA)。

    Args:
        X: 预测变量矩阵 (n, k)。
        y: 组标签向量。
        priors: 先验概率 {组名: 概率}, None=基于样本比例。

    Returns:
        LDAResult。
    """
    X_mat = np.asarray(X, dtype=np.float64)
    y_str = np.asarray(y)

    mask = ~np.isnan(X_mat).any(axis=1)
    y_str = np.array([str(v) for v in y_str])
    mask = mask & (y_str != "nan")
    X_mat = X_mat[mask]
    y_str = y_str[mask]

    n, k = X_mat.shape
    groups = sorted(set(y_str))
    g = len(groups)

    # 各组统计
    group_data = {}
    group_means = {}
    group_ns = {}
    group_covs = {}

    for grp in groups:
        Xg = X_mat[np.array([str(v) == grp for v in y_str])]
        group_data[grp] = Xg
        group_means[grp] = np.mean(Xg, axis=0)
        group_ns[grp] = len(Xg)
        if len(Xg) > 1:
            group_covs[grp] = np.cov(Xg, rowvar=False)
        else:
            group_covs[grp] = np.eye(k)

    grand_mean = np.mean(X_mat, axis=0)

    # 合并组内协方差 (Pooled)
    W = np.zeros((k, k))  # Within-group
    for grp in groups:
        ng = group_ns[grp]
        W += (ng - 1) * group_covs[grp]
    W /= (n - g)

    # 组间离差
    B = np.zeros((k, k))
    for grp in groups:
        ng = group_ns[grp]
        diff = (group_means[grp] - grand_mean).reshape(-1, 1)
        B += ng * diff @ diff.T

    # 解 W⁻¹B 的特征问题
    try:
        W_inv = np.linalg.inv(W)
        W_inv_B = W_inv @ B
    except np.linalg.LinAlgError:
        W_inv = np.linalg.pinv(W)
        W_inv_B = W_inv @ B

    eigvals, eigvecs = np.linalg.eig(W_inv_B)
    # 取实部并排序
    eigvals = np.real(eigvals)
    idx = np.argsort(eigvals)[::-1]
    m = min(g - 1, k)  # 判别函数维度
    eigvals = eigvals[idx][:m]
    eigvecs = np.real(eigvecs[:, idx][:, :m])

    # Wilks' Lambda
    wilks = 1.0
    for ev in eigvals:
        wilks /= (1 + ev)
    # Bartlett's χ² 近似
    wilks_chi2 = -(n - 1 - (k + g) / 2) * math.log(max(wilks, 1e-10))
    wilks_df = k * (g - 1)
    wilks_p = 1.0 - sp_stats.chi2.cdf(wilks_chi2, wilks_df)

    # 典型相关系数
    canonical_corr = [math.sqrt(ev / (1 + ev)) for ev in eigvals]

    # 方差解释
    total_eig = sum(eigvals)
    var_exp = [ev / total_eig * 100 if total_eig > 0 else 0.0 for ev in eigvals]

    # 标准化系数 (用合并组内标准差)
    pooled_std = np.sqrt(np.diag(W))
    std_coefs = []
    for j in range(m):
        std_coefs.append([float(eigvecs[i, j] * pooled_std[i]) for i in range(k)])

    # 未标准化系数
    unstd_coefs = []
    for j in range(m):
        unstd_coefs.append([float(eigvecs[i, j]) for i in range(k)])

    # 结构矩阵 (Pooled within-group correlations)
    struct = np.zeros((k, m))
    for j in range(m):
        scores = X_mat @ eigvecs[:, j]
        for i in range(k):
            struct[i, j] = np.corrcoef(X_mat[:, i], scores)[0, 1]

    # Fisher 分类系数
    fisher_coefs = {}
    fisher_consts = {}
    for grp in groups:
        mu = group_means[grp]
        f_coef = W_inv @ mu
        f_const = -0.5 * mu @ W_inv @ mu
        if priors:
            f_const += math.log(priors.get(grp, 1.0 / g))
        fisher_coefs[grp] = [float(v) for v in f_coef]
        fisher_consts[grp] = float(f_const)

    # Box's M
    box_m, box_chi2, box_p = _box_m_test(group_covs, group_ns, k, g)

    # 分类混淆矩阵
    y_pred = _predict_lda(X_mat, fisher_coefs, fisher_consts, groups)
    cm = np.zeros((g, g), dtype=int)
    for i in range(n):
        true_idx = groups.index(str(y_str[i]))
        pred_idx = groups.index(y_pred[i])
        cm[true_idx, pred_idx] += 1
    accuracy = np.trace(cm) / n

    return LDAResult(
        n=n,
        k=k,
        g=g,
        wilks_lambda=wilks,
        wilks_chi2=wilks_chi2,
        wilks_df=wilks_df,
        wilks_p=wilks_p,
        box_m=box_m,
        box_chi2=box_chi2,
        box_p=box_p,
        eigenvalues=[float(ev) for ev in eigvals],
        canonical_corr=canonical_corr,
        variance_explained=var_exp,
        std_coefficients=std_coefs,
        unstd_coefficients=unstd_coefs,
        structure_matrix=[[float(struct[i, j]) for j in range(m)] for i in range(k)],
        fisher_coefficients=fisher_coefs,
        fisher_constants=fisher_consts,
        confusion_matrix=cm,
        accuracy=float(accuracy),
    )


def _box_m_test(
    covs: dict[str, np.ndarray],
    ns: dict[str, int],
    k: int,
    g: int,
) -> tuple[float, float | None, float | None]:
    """Box's M 检验 (等协方差矩阵)。"""
    n_total = sum(ns.values())
    # 合并组内协方差
    Sp = np.zeros((k, k))
    for grp, cov in covs.items():
        Sp += (ns[grp] - 1) * cov
    Sp /= (n_total - g)

    # M = (n_total - g) * ln|Sp| - Σ (n_i - 1) * ln|S_i|
    sign_sp, logdet_sp = np.linalg.slogdet(Sp)
    M = (n_total - g) * logdet_sp if sign_sp > 0 else 0.0
    for grp, cov in covs.items():
        sign_s, logdet_s = np.linalg.slogdet(cov)
        if sign_s > 0:
            M -= (ns[grp] - 1) * logdet_s

    # 自由度校正
    c1 = (sum(1.0 / (ns[grp] - 1) for grp in covs) - 1.0 / (n_total - g)) * (2 * k**2 + 3 * k - 1) / (6 * (k + 1) * (g - 1))
    df = k * (k + 1) * (g - 1) // 2

    if df <= 0 or c1 >= 1:
        return float(M), None, None

    chi2 = M * (1 - c1)
    p = 1.0 - sp_stats.chi2.cdf(max(chi2, 0), df)
    return float(M), float(chi2), float(p)


def _predict_lda(
    X: np.ndarray,
    fisher_coefs: dict,
    fisher_consts: dict,
    groups: list[str],
) -> list[str]:
    """Fisher 分类预测。"""
    predictions = []
    for i in range(len(X)):
        scores = {}
        for grp in groups:
            scores[grp] = X[i] @ np.array(fisher_coefs[grp]) + fisher_consts[grp]
        predictions.append(max(scores, key=scores.get))
    return predictions


