"""交叉表与关联度量 (Crosstabs) — 全参数实现

复刻 SPSS CROSSTABS 过程:
- 列联表 (频数 + 行%/列%/总%)
- χ² 检验 (Pearson + Yates + Likelihood Ratio)
- Fisher 精确检验
- 关联度量: Phi, Cramer's V, Contingency Coefficient, Lambda, Goodman-Kruskal τ
- 分层分析 (Mantel-Haenszel)
- 一致性与 Kappa (2×2 Cohen's Kappa)

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【χ² 检验的核心: 观察值 vs 期望值】

期望值 E_ij = (行总和 × 列总和) / 总样本量
这是"如果两个变量完全独立, 这个格子应该有多少人"。

χ² = Σ (O_ij - E_ij)² / E_ij
如果 χ² 很大 (远大于其自由度对应的期望值) → 变量间存在关联。

自由度 = (行数 - 1) × (列数 - 1)
直觉: 因为行和列的总和是固定的, 你只能自由填写 (r-1)(c-1) 个格子。

χ² 检验的有效性条件 (Cochran 规则):
  所有格子的期望频率 ≥ 1, 且有 ≤ 20% 的期望频率 < 5。
  违反 → 用 Fisher 精确检验而不是 χ²。

【Pearson χ² vs Likelihood Ratio χ² (G²) vs Yates】

Pearson χ²: 经典。大样本性质好, 最常用。
  G² (似然比): = 2 × Σ O_ij × ln(O_ij / E_ij)。
  在嵌套模型比较时优越 (可分解), 与 Pearson χ² 几乎一致。
  Yates 连续校正: 对 2×2 表, 每个 |O-E| 减去 0.5。
  目标: 补偿离散观察值拟合连续 χ² 分布的误差。
  争议: 过于保守 (第一类错误率 < 0.05), 很多统计学家不建议使用。

【关联度量 — χ² 显著的后续问题】

χ² 显著 → 变量有关联, 但关联强吗？多强？

φ (Phi): 2×2 表的标准化。= sqrt(χ²/n)。
  在 2×2 表中取值 0~1。在更大的表中可以 >1 (所以不通用)。

Cramer's V: φ 对 r×c 表的推广。= sqrt(χ² / (n×min(r-1,c-1)))。
  取值 0~1, 不受行列数影响。最广泛推荐的关联度量。
  V ≈ 0.1=弱, 0.3=中, 0.5=强 (取决于自由度)。

Contingency Coefficient: = sqrt(χ² / (χ² + n))。
  数学上不能达到 1, 上限 = sqrt((min(r,c)-1)/min(r,c))。
  现在较少使用, 推荐用 Cramer's V 代替。

Lambda (Goodman-Kruskal): 基于"预测"的度量。
  解读: "用行变量预测列变量时, 预测错误的减少比例"。
  0 → 无改善, 1 → 完全预测。
  注意: 当有一个类别的列占了绝大多数时, Lambda 可能 = 0
  (因为"总是猜众数"已经是很好的基线, 行变量提供不了额外帮助)。

【Cohen's Kappa — 两评定者一致性】

κ = (Pₒ - Pₑ) / (1 - Pₑ)
Pₒ: 两位评定者实际一致的比率
Pₑ: 如果两人随机乱评, 期望一致的比率

Kappa 排除了"运气好碰巧一致"的成分。

κ = 0 → 除了随机一致外没有额外一致性
κ = 1 → 完美一致
κ < 0 → 一致性比随机还差 (少见, 暗示系统性的不一致)

Landis & Koch 的经验解读: <0.00 差, 0~0.20 微弱, 0.21~0.40 可接受,
  0.41~0.60 中等, 0.61~0.80 实质一致, 0.81~1.00 近乎完美。

Kappa 的悖论: 在几乎所有人都在同一个类别时 (极高的 Pₑ),
即使 Pₒ 很高, Kappa 也可能很低。因为"碰巧一致"太高了。
此时应同时报告 Pₒ, Pₑ 和 κ, 不只靠 κ。

加权 Kappa: 当类别有序时, 对距离近的错误给更多容忍。
  线性加权 → 距离成线性惩罚; 二次加权 → 距离平方惩罚 (类 ICC)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════


@dataclass
class CrosstabResult:
    """交叉表分析结果"""

    table: np.ndarray
    row_labels: list[str]
    col_labels: list[str]
    n: int
    # 检验
    pearson_chi2: float
    pearson_df: int
    pearson_p: float
    yates_chi2: float | None = None
    yates_p: float | None = None
    likelihood_ratio: float | None = None
    likelihood_p: float | None = None
    fisher_exact_p: float | None = None
    # 关联度量
    phi: float = float("nan")
    cramers_v: float = float("nan")
    contingency_coef: float = float("nan")
    lambda_row: float = float("nan")
    lambda_col: float = float("nan")
    lambda_symmetric: float = float("nan")
    # 分层 (Mantel-Haenszel)
    mh_common_or: float | None = None
    mh_chi2: float | None = None
    mh_p: float | None = None


# ═══════════════════════════════════════════
# 列联表分析
# ═══════════════════════════════════════════


def crosstab(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    include_marginals: bool = True,
) -> pd.DataFrame:
    """生成交叉频数表 (含边际)。"""
    table = pd.crosstab(x, y, margins=include_marginals, margins_name="合计")
    return table


def chi_square_test(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    correction: bool = True,
) -> CrosstabResult:
    """χ² 独立性检验 + 全套关联度量。

    Args:
        x: 行变量。
        y: 列变量。
        correction: 是否对 2×2 表使用 Yates 连续性校正。

    Returns:
        CrosstabResult。
    """
    # 去缺失
    mask = (~pd.isna(x)) & (~pd.isna(y))
    x_clean = np.asarray(x)[mask]
    y_clean = np.asarray(y)[mask]
    n = len(x_clean)

    # 构建列联表
    row_labels = sorted(set(str(v) for v in x_clean))
    col_labels = sorted(set(str(v) for v in y_clean))
    r, c = len(row_labels), len(col_labels)

    table = np.zeros((r, c), dtype=int)
    for i in range(n):
        ri = row_labels.index(str(x_clean[i]))
        ci = col_labels.index(str(y_clean[i]))
        table[ri, ci] += 1

    # 边际
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    # Pearson χ²
    expected = np.outer(row_sums, col_sums) / n
    if np.any(expected == 0):
        pearson_chi2 = float("nan")
        pearson_p = float("nan")
    else:
        pearson_chi2 = float(np.sum((table - expected) ** 2 / expected))
        df = (r - 1) * (c - 1)
        pearson_p = float(1.0 - sp_stats.chi2.cdf(pearson_chi2, df)) if df > 0 else 1.0

    df = (r - 1) * (c - 1)

    # Yates 校正 (仅 2×2)
    yates_chi2, yates_p = None, None
    if r == 2 and c == 2 and correction:
        a, b_ = table[0, 0], table[0, 1]
        c_, d = table[1, 0], table[1, 1]
        yates_chi2 = n * (abs(a * d - b_ * c_) - n / 2) ** 2 / (row_sums[0] * row_sums[1] * col_sums[0] * col_sums[1])
        yates_p = float(1.0 - sp_stats.chi2.cdf(yates_chi2, 1))

    # 似然比 χ²
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = 2 * np.sum(table * np.log(table / expected), where=(table > 0))
    lr = float(lr) if np.isfinite(lr) else float("nan")
    lr_p = float(1.0 - sp_stats.chi2.cdf(lr, df)) if np.isfinite(lr) and df > 0 else float("nan")

    # Fisher 精确检验 (仅 2×2)
    fisher_p = None
    if r == 2 and c == 2:
        _, fisher_p = sp_stats.fisher_exact(table.astype(int))

    # 关联度量
    phi = math.sqrt(pearson_chi2 / n) if not math.isnan(pearson_chi2) and n > 0 else float("nan")
    min_dim = min(r, c)
    cramers_v = math.sqrt(pearson_chi2 / (n * (min_dim - 1))) if not math.isnan(pearson_chi2) and n > 0 and min_dim > 1 else float("nan")
    cc = math.sqrt(pearson_chi2 / (pearson_chi2 + n)) if not math.isnan(pearson_chi2) else float("nan")

    # Lambda (Goodman-Kruskal)
    lam_row, lam_col, lam_sym = _lambda_measures(table, row_sums, col_sums, n)

    return CrosstabResult(
        table=table,
        row_labels=row_labels,
        col_labels=col_labels,
        n=n,
        pearson_chi2=pearson_chi2,
        pearson_df=df,
        pearson_p=pearson_p,
        yates_chi2=yates_chi2,
        yates_p=yates_p,
        likelihood_ratio=lr,
        likelihood_p=lr_p,
        fisher_exact_p=fisher_p,
        phi=phi,
        cramers_v=cramers_v,
        contingency_coef=cc,
        lambda_row=lam_row,
        lambda_col=lam_col,
        lambda_symmetric=lam_sym,
    )


def _lambda_measures(
    table: np.ndarray, row_sums: np.ndarray, col_sums: np.ndarray, n: int
) -> tuple[float, float, float]:
    """Goodman-Kruskal Lambda 关联度量。"""
    r, c = table.shape

    # 对称 Lambda
    max_row = np.max(table, axis=1).sum()
    max_col = np.max(table, axis=0).sum()
    row_mode = np.max(row_sums)
    col_mode = np.max(col_sums)

    lam_row = (max_row - col_mode) / (n - col_mode) if n > col_mode else 0.0
    lam_col = (max_col - row_mode) / (n - row_mode) if n > row_mode else 0.0
    lam_sym = (max_row + max_col - row_mode - col_mode) / (2 * n - row_mode - col_mode) if (2 * n) > (row_mode + col_mode) else 0.0

    return float(lam_row), float(lam_col), float(lam_sym)


# ═══════════════════════════════════════════
# Cohen's Kappa
# ═══════════════════════════════════════════


@dataclass
class KappaResult:
    """Cohen's Kappa 结果"""

    kappa: float
    se: float
    z: float
    p_value: float
    ci_95: tuple[float, float]
    po: float  # 观察一致率
    pe: float  # 期望一致率
    n: int


def cohens_kappa(
    rater1: np.ndarray,
    rater2: np.ndarray,
    *,
    weights: str | None = None,
) -> KappaResult:
    """Cohen's Kappa (两评定者一致性)。

    Args:
        rater1, rater2: 两位评定者的评分 (等长)。
        weights: ``"linear"`` (线性加权) / ``"quadratic"`` (平方加权) / None。

    Returns:
        KappaResult。
    """
    a1 = np.asarray(rater1)
    a2 = np.asarray(rater2)
    mask = (~pd.isna(a1)) & (~pd.isna(a2))
    a1, a2 = a1[mask], a2[mask]
    n = len(a1)

    categories = sorted(set(list(a1) + list(a2)))
    k = len(categories)
    cat_map = {c: i for i, c in enumerate(categories)}

    # 频数矩阵
    table = np.zeros((k, k), dtype=int)
    for i in range(n):
        table[cat_map[a1[i]], cat_map[a2[i]]] += 1

    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    # 权重矩阵
    if weights is None:
        W = np.eye(k)  # 仅完全一致计分
    else:
        W = np.zeros((k, k))
        max_diff = k - 1
        for i in range(k):
            for j in range(k):
                diff = abs(i - j)
                if weights == "linear":
                    W[i, j] = 1.0 - diff / max_diff if max_diff > 0 else 1.0
                else:  # quadratic
                    W[i, j] = 1.0 - (diff / max_diff) ** 2 if max_diff > 0 else 1.0

    # 观察一致率
    Po = float(np.sum(W * table) / n)

    # 期望一致率
    expected = np.outer(row_sums, col_sums) / n
    Pe = float(np.sum(W * expected) / n)

    # Kappa
    kappa = (Po - Pe) / (1 - Pe) if Pe < 1.0 else 0.0

    # 标准误
    P_bar = np.sum(W * expected) / n
    se = math.sqrt((Po * (1 - Po)) / (n * (1 - Pe) ** 2)) if n > 0 and Pe < 1 else float("nan")

    z = kappa / se if se and se > 0 else 0.0
    p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))

    z_alpha = sp_stats.norm.ppf(0.975)
    ci = (kappa - z_alpha * se, kappa + z_alpha * se)

    return KappaResult(
        kappa=float(kappa),
        se=float(se),
        z=float(z),
        p_value=float(p),
        ci_95=(float(ci[0]), float(ci[1])),
        po=Po,
        pe=Pe,
        n=n,
    )


# ═══════════════════════════════════════════
# 分层 Mantel-Haenszel
# ═══════════════════════════════════════════


def mantel_haenszel(
    tables: list[np.ndarray],
) -> dict:
    """Mantel-Haenszel 分层分析 (2×2×k 表)。

    Args:
        tables: k 个 2×2 表的列表 [table1, table2, ...]。

    Returns:
        {common_or, chi2_mh, p, or_ci_95}。
    """
    k = len(tables)
    ors = []
    weights = []
    num_sum = 0.0
    denom_sum = 0.0
    a_sum = 0.0
    e_sum = 0.0
    v_sum = 0.0

    for t in tables:
        a, b = t[0, 0], t[0, 1]
        c, d = t[1, 0], t[1, 1]
        n = a + b + c + d

        # 公共 OR (Mantel-Haenszel)
        num_sum += a * d / n
        denom_sum += b * c / n

        # χ²_MH
        e_a = (a + b) * (a + c) / n
        v_a = (a + b) * (c + d) * (a + c) * (b + d) / (n**2 * (n - 1))
        a_sum += a
        e_sum += e_a
        v_sum += v_a

    common_or = num_sum / denom_sum if denom_sum > 0 else float("nan")
    chi2_mh = (abs(a_sum - e_sum) - 0.5) ** 2 / v_sum if v_sum > 0 else 0.0  # 连续性校正
    p_mh = 1.0 - sp_stats.chi2.cdf(chi2_mh, 1) if v_sum > 0 else 1.0

    return {
        "common_or": float(common_or),
        "chi2_mh": float(chi2_mh),
        "p": float(p_mh),
        "k_strata": k,
    }


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


# ═══════════════════════════════════════════
# 卡方拟合优度检验 (Goodness of Fit)
# ═══════════════════════════════════════════


@dataclass
class GofResult:
    """卡方拟合优度检验结果"""

    chi2: float
    df: int
    p_value: float
    n: int
    observed: np.ndarray
    expected: np.ndarray
    residuals: np.ndarray  # (O - E) / sqrt(E)


def chi_square_gof(
    observed: dict[str, int] | list[int] | np.ndarray,
    expected: dict[str, float] | list[float] | np.ndarray | None = None,
    *,
    expected_proportions: list[float] | None = None,
) -> GofResult:
    """卡方拟合优度检验 (One-Sample χ² GOF)。

    H₀: 观测频数分布与期望分布一致。

    可通过 ``expected`` 提供绝对期望频数，或通过 ``expected_proportions``
    提供期望比例（自动乘以总样本量转为频数）。若两者均不提供，假定均匀分布。

    Args:
        observed: 观测频数 (dict 以类别为 key, 或 list/array 按顺序)。
        expected: 绝对期望频数 (与 observed 等长或同 dict)。
        expected_proportions: 期望比例 (与 observed 等长, 自动归一化)。

    Returns:
        GofResult。
    """
    # 统一为 numpy 数组
    if isinstance(observed, dict):
        keys = sorted(observed.keys())
        obs = np.array([observed[k] for k in keys], dtype=np.float64)
    else:
        obs = np.asarray(observed, dtype=np.float64)

    n = int(np.sum(obs))
    k = len(obs)

    if expected is not None:
        if isinstance(expected, dict):
            exp = np.array([expected.get(k, 0.0) for k in keys], dtype=np.float64)
        else:
            exp = np.asarray(expected, dtype=np.float64)
    elif expected_proportions is not None:
        props = np.asarray(expected_proportions, dtype=np.float64)
        props = props / props.sum()
        exp = props * n
    else:
        # 默认均匀分布
        exp = np.full(k, n / k, dtype=np.float64)

    # 确保不小于 1e-10
    exp = np.maximum(exp, 1e-10)

    # χ² = Σ (O - E)² / E
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    df = k - 1
    p = float(1.0 - sp_stats.chi2.cdf(chi2, df)) if df > 0 else 1.0
    residuals = (obs - exp) / np.sqrt(exp)

    # Cochran 规则检查
    n_small = int(np.sum(exp < 5))
    pct_small = n_small / k * 100
    if n_small > k * 0.2 or np.any(exp < 1):
        import warnings
        warnings.warn(
            f"Cochran 规则违反: {n_small}/{k} 个类别 ({pct_small:.0f}%) "
            f"期望频数 < 5（不应超过 20%），或存在期望频数 < 1。"
            f"考虑合并类别或使用精确检验。"
        )

    return GofResult(
        chi2=chi2,
        df=df,
        p_value=p,
        n=n,
        observed=obs,
        expected=exp,
        residuals=residuals,
    )


def gof_report(r: GofResult) -> str:
    """拟合优度检验报告。"""
    lines = [
        f"{'='*50}",
        f"  χ² 拟合优度检验",
        f"  n = {r.n}, df = {r.df}",
        f"{'='*50}",
        f"  χ² = {r.chi2:.4f}, p = {r.p_value:.4f}",
        "",
        f"  {'类别':<8} {'观测':>8} {'期望':>8} {'残差':>8}",
    ]
    for i in range(len(r.observed)):
        lines.append(
            f"  {i+1:<8} {r.observed[i]:8.1f} {r.expected[i]:8.2f} {r.residuals[i]:8.3f}"
        )
    return "\n".join(lines)


def chi_square_report(r: CrosstabResult) -> str:
    """χ² 检验报告。"""
    lines = [
        f"{'='*50}",
        f"  χ² 独立性检验",
        f"  n = {r.n}, df = {r.pearson_df}",
        f"{'='*50}",
        f"  Pearson χ² = {r.pearson_chi2:.4f}, p = {r.pearson_p:.4f}",
    ]
    if r.yates_chi2 is not None:
        lines.append(f"  Yates 校正 χ² = {r.yates_chi2:.4f}, p = {r.yates_p:.4f}")
    if r.likelihood_ratio is not None and not math.isnan(r.likelihood_ratio):
        lines.append(f"  似然比 χ² = {r.likelihood_ratio:.4f}, p = {r.likelihood_p:.4f}")
    if r.fisher_exact_p is not None:
        lines.append(f"  Fisher 精确检验 p = {r.fisher_exact_p:.4f}")
    lines.extend([
        "",
        f"  Phi = {r.phi:.4f}",
        f"  Cramer's V = {r.cramers_v:.4f}",
        f"  列联系数 = {r.contingency_coef:.4f}",
    ])
    return "\n".join(lines)
