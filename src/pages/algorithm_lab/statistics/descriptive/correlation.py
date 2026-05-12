"""相关分析 — 全参数实现

复刻 SPSS 相关分析 (Bivariate Correlations) + 多重比较校正:
- Pearson r（线性相关）
- Spearman ρ（秩相关，单调但不要求线性）
- Kendall τ-b（基于 concordant/discordant 对，处理 ties 最优）
- Fisher z 变换构建置信区间
- 多重比较校正：Bonferroni（最保守）、Holm（仍控制 FWER）、Benjamini-Hochberg FDR（推荐探索分析）
- 成对完整案例删除（最大化可用数据）

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【Pearson r — 线性关系的度量】

    r = Σ[(x_i - x̄)(y_i - ȳ)] / [√Σ(x_i - x̄)² · √Σ(y_i - ȳ)²]
    r ∈ [-1, 1]，0 = 无线性关系，1 = 完美正线性，-1 = 完美负线性

假定：
- 连续变量（不是序数/分类）
- 线性关系（不检测非线性关联）
- 二元正态分布（推论用）
- 无极端离群值（r 对离群值敏感）

经验法则（Cohen）：
    |r| = 0.1 → 弱（小效应）
    |r| = 0.3 → 中等
    |r| = 0.5 → 强（大效应）

何时用 Pearson：变量连续、分布大致正态、关心线性关系强度。

【Spearman ρ — 秩相关（单调关系）】

    ρ = 皮尔逊公式应用于秩（rank）而非原始值。

Spearman 不要求线性！只要 Y 随 X 一直增大（或一直减小），
无论关系是直线还是曲线，ρ ≈ ±1。

    例：Y = X^3，Pearson r ≈ 0.92（不是 1，因为不是直线）
            Spearman ρ = 1.00（完美单调）

优势：
- 不假定正态分布
- 对离群值稳健（秩限制了极端值的影响）
- 适用于序数变量

何时用 Spearman：数据偏态、有离群值、序数变量、怀疑关系是单调但非线性的。

【Kendall τ-b — 基于 concordance 的关系】

    对于每一对观测 (i, j)：
    - Concordant：x_i< x_j 且 y_i< y_j（方向一致）
    - Discordant：x_i< x_j 且 y_i> y_j（方向相反）
    τ = (n_concordant - n_discordant) / √((n₀-n₁)(n₀-n₂))

    其中 n₀ = n(n-1)/2，n₁/n₂ 分别是对 x 和 y 的 ties 的修正。

优势：
- 对 ties 处理最好（τ-b 专门修正了 ties）
- 小样本 + 多 ties 时最稳健
- 直观解释：τ = 0.5 表示"对于随机取出的两对观测，concordant 比 discordant 多 50%"

劣势：
- 计算最慢（O(n²) 对数比较）
- 绝对值通常比 Pearson/Spearman 小（不是相同尺度）

何时用 Kendall：小样本且有很多并列值（ties）、需要最稳健的估计。

【Fisher z 变换 — 构建 r 的置信区间】

Pearson r 的抽样分布不是正态的：在接近 ±1 时严重偏斜。
Fisher z = arctanh(r) 的抽样分布近似正态：
    z ~ N( arctanh(ρ), 1/(n-3) )

在 z 空间构建 CI → 变换回 r 空间：
    CI_low  = tanh( z ± z_crit × 1/√(n-3) )

    注意：n-3（不是 n-1）是因为估计 r 消耗了 3 个自由度
    （两个均值 + 一个协方差）。

Spearman ρ 也可以用 Fisher z 近似（n 不太小时）。

【多重比较问题 — 为什么需要校正】

假设你对 10 个变量计算所有两两相关：m = 10×9/2 = 45 对。
每个检验 α=0.05，如果所有 45 对都是独立不相关的，
你期望有 45 × 0.05 ≈ 2.25 个"显著"结果——纯由随机产生。

不做校正 → 你的报告里平均会有 2 个假阳性。
变量越多，假阳性越多。这就是"多重比较膨胀"。

三种校正策略：

1. Bonferroni 校正（控制 FWER — Familywise Error Rate）
   p_adj = min(p × m, 1.0)
   把 α 除以 m。最保守 → 不容易检出真实效应，但出错概率极低。
   适用于：必须有极高把握才能宣称"存在相关"的场景。

2. Holm 校正（也是 FWER，但比 Bonferroni 更不保守）
   步骤：
   (a) 把所有 p 值从小到大排序
   (b) 第 i 小的 p 值：p_adj = p_i × (m - i + 1)
   (c) 强制单调性：第 i+1 个的调整 p 值不能小于第 i 个
   Holm 的统计效力（power）始终 ≥ Bonferroni。

3. Benjamini-Hochberg FDR（控制 False Discovery Rate）
   控制"显著结果中被错误拒绝的比例的期望值"。
   步骤：
   (a) 把所有 p 值从小到大排序：p₁ ≤ p₂ ≤ ... ≤ p_m
   (b) 第 i 小的 p 值：p_adj = p_i × m / i
   (c) 强制单调性
   比 FWER 方法更不保守 → 更多真实效应被检出。
   适用于：探索性分析，允许一定比例的假阳性，更关心"不遗漏"。
   推荐作为默认选择。

【成对 (pairwise) vs 列表删除 (listwise) 缺失值处理】

成对删除：每对 (X_i, X_j) 只用两者都非缺失的观测来算相关。
   优势：最大化可用数据（不因为某个变量的缺失丢失整行）。
   劣势：不同变量对的相关基于不同样本量，可能产生不一致的相关矩阵。

列表删除：只使用所有变量都非缺失的行。
   优势：所有相关基于同一个样本。
   劣势：一个缺失变量就让整行作废，在大变量集时浪费严重。
   当缺失完全随机 (MCAR) 时两者一致。

默认使用成对删除（SPSS 默认行为）。
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
class CorrelationPairResult:
    """单对相关分析结果"""

    method: str  # "pearson" | "spearman" | "kendall"
    r: float  # 相关系数
    p_value: float  # 双侧 p 值
    n: int  # 有效样本量
    ci_95: tuple[float, float]  # Fisher z 变换 CI
    label_x: str = ""  # 变量名 X
    label_y: str = ""  # 变量名 Y


@dataclass
class CorrelationResult:
    """相关矩阵完整结果"""

    method: str  # "pearson" | "spearman" | "kendall"
    n_vars: int  # 变量数
    n_samples: list[list[int]]  # 每对的有效样本量 (下三角)
    corr_matrix: np.ndarray  # 相关矩阵 (n_vars × n_vars)
    p_matrix: np.ndarray  # 未校正 p 值矩阵
    ci_lower: np.ndarray  # CI 下界
    ci_upper: np.ndarray  # CI 上界
    significant_matrix: np.ndarray  # 校正后显著标记矩阵 (bool)
    correction_method: str  # "bonferroni" | "holm" | "fdr_bh" | "none"
    var_names: list[str] = field(default_factory=list)  # 变量名


# ═══════════════════════════════════════════
# Fisher z 变换
# ═══════════════════════════════════════════


def _fisher_z(r: float) -> float:
    """Fisher z 变换 = arctanh(r)。

    r → z 空间的变换使 z 近似服从正态分布，用于构建 CI。

    Args:
        r: 相关系数（-1 到 1）。

    Returns:
        z 值。r=0 时 z=0，r=0.9 时 z≈1.47。
    """
    if r >= 1.0:
        return float("inf")
    if r <= -1.0:
        return float("-inf")
    return float(0.5 * math.log((1.0 + r) / (1.0 - r)))


def _fisher_z_inv(z: float) -> float:
    """Fisher z 逆变换 = tanh(z)。

    把 z 空间的值变换回 r 空间。

    Args:
        z: z 值。

    Returns:
        相关系数 r，范围 (-1, 1)。
    """
    if z == float("inf"):
        return 1.0
    if z == float("-inf"):
        return -1.0
    return float(math.tanh(z))


# ═══════════════════════════════════════════
# 多重比较校正
# ═══════════════════════════════════════════


def _flatten_upper_triangle(mat: np.ndarray) -> np.ndarray:
    """提取上三角（不含对角线）元素变平，用于计算 p 值校正。

    Args:
        mat: 方阵。

    Returns:
        一维数组，包含所有上三角元素。
    """
    n = mat.shape[0]
    idx = np.triu_indices(n, k=1)
    return mat[idx]


def _rebuild_symmetric(flat: np.ndarray, n_vars: int) -> np.ndarray:
    """从压缩上三角重建完整对称矩阵（对角线为 nan）。

    Args:
        flat: 上三角压缩值。
        n_vars: 变量数。

    Returns:
        n_vars × n_vars 对称矩阵。
    """
    mat = np.full((n_vars, n_vars), np.nan)
    idx = np.triu_indices(n_vars, k=1)
    mat[idx] = flat
    mat[(idx[1], idx[0])] = flat  # 下三角 = 上三角转置
    return mat


def _correction_bonferroni(p_flat: np.ndarray, m: int) -> np.ndarray:
    """Bonferroni 校正：p_adj = min(p × m, 1.0)。

    最严格的 FWER 控制。适用于"不能有任何假阳性"的场景。

    Args:
        p_flat: 未校正的 p 值（上三角压缩）。
        m: 总比较次数。

    Returns:
        校正后的 p 值。
    """
    return np.minimum(p_flat * m, 1.0)


def _correction_holm(p_flat: np.ndarray, m: int) -> np.ndarray:
    """Holm 校正（step-down 过程）。

    步骤：
    1. p 从小到大的排序化：p₁ ≤ p₂ ≤ ... ≤ p_m
    2. 调整：p_adj_i = min(1, max( p_i × (m - i + 1), 前一个调整 p 值 ))
    3. 强制单调性（后一个不能小于前一个）

    控制 FWER，但比 Bonferroni 更有统计效力。

    Args:
        p_flat: 未校正的 p 值。
        m: 总比较次数。

    Returns:
        校正后的 p 值。
    """
    n_tests = len(p_flat)
    order = np.argsort(p_flat)
    p_adj = np.zeros(n_tests)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = p_flat[idx] * (m - rank)
        adj = max(adj, prev)  # 单调性约束
        adj = min(adj, 1.0)
        p_adj[idx] = adj
        prev = adj
    return p_adj


def _correction_fdr_bh(p_flat: np.ndarray, m: int) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正。

    控制 False Discovery Rate（显著结果中假阳性的期望比例）。

    步骤：
    1. p 从大到小的排序化
    2. 第 k 个最小的 p 值：p_adj = p_k × m / k
    3. 强制单调性

    是三者中最不保守的，推荐作为探索性分析的默认校正。

    Args:
        p_flat: 未校正的 p 值。
        m: 总比较次数。

    Returns:
        校正后的 p 值。
    """
    n_tests = len(p_flat)
    order = np.argsort(p_flat)
    p_adj = np.zeros(n_tests)
    prev = float("inf")  # 从最大 p 值开始向后
    # 从大到小遍历（step-up 过程）
    for rank_1 in range(n_tests, 0, -1):  # rank in 1..n
        idx = order[rank_1 - 1]
        adj = p_flat[idx] * m / rank_1
        adj = min(adj, prev)  # 单调性：当前不能大于前一个（更大的 p）
        adj = min(adj, 1.0)
        p_adj[idx] = adj
        prev = adj
    return p_adj


# ═══════════════════════════════════════════
# 核心相关矩阵
# ═══════════════════════════════════════════


def correlation_matrix(
    data: np.ndarray | pd.DataFrame,
    *,
    method: str = "pearson",
    ci_level: float = 0.95,
    correction: str = "fdr_bh",
    pairwise: bool = True,
) -> CorrelationResult:
    """相关矩阵 — 所有两两相关 + p 值 + CI + 多重比较校正。

    对 k 个变量计算 k×k 相关矩阵，含：

    - 相关系数矩阵
    - 未校正 p 值矩阵
    - Fisher z 变换 95% CI
    - 多重比较校正（Bonferroni / Holm / FDR-BH）
    - 每对有效样本量（成对删除时各对不同）

    Args:
        data: n × k 数据矩阵或 DataFrame（行=观测，列=变量）。
        method: 相关方法。
            ``"pearson"`` — 积矩相关，要求连续变量，对离群值敏感。
            ``"spearman"`` — 秩相关，检测单调关系（不一定是线性）。
            ``"kendall"`` — τ-b 相关，基于 concordant/discordant 对，处理 ties 最佳。
        ci_level: Fisher z 变换置信区间的置信水平（默认 0.95）。
        correction: 多重比较校正方法。
            ``"none"`` — 不做校正（假阳性风险随变量数增加）。
            ``"bonferroni"`` — FWER 控制，最保守。
            ``"holm"`` — FWER 控制，比 Bonferroni 更有统计效力。
            ``"fdr_bh"`` — FDR 控制（推荐探索分析的默认选择）。
        pairwise: 缺失值处理。
            ``True`` — 成对删除（每对用两者都非缺失的观测计算，最大化可用数据）。
            ``False`` — 列表删除（删除任何包含 NaN 的整行，所有相关基于同一组样本）。

    Returns:
        CorrelationResult。

    使用示例:
        >>> data = np.random.randn(100, 5)
        >>> result = correlation_matrix(data, method="spearman", correction="fdr_bh")
        >>> result.corr_matrix  # 5×5 相关矩阵
        >>> result.significant_matrix  # 校正后显著的对为 True
    """
    # ── 数据清洗 ──
    if isinstance(data, pd.DataFrame):
        var_names = list(data.columns)
        arr = data.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
        k = arr.shape[1]
        var_names = [f"X{i+1}" for i in range(k)]

    if arr.ndim != 2:
        raise ValueError(f"data 必须是 2 维数组/DataFrame，但得到 {arr.ndim} 维。")
    n, k = arr.shape
    if k < 2:
        raise ValueError(f"至少需要 2 个变量才能计算相关矩阵（当前 k={k}）。")

    # ── 选择相关函数 ──
    if method == "pearson":
        corr_func = _pairwise_pearson
    elif method == "spearman":
        corr_func = _pairwise_spearman
    elif method == "kendall":
        corr_func = _pairwise_kendall
    else:
        raise ValueError(f"不支持的相关方法: '{method}'。可选: pearson, spearman, kendall。")

    # ── 逐对计算 ──
    corr_mat = np.full((k, k), np.nan)
    p_mat = np.full((k, k), np.nan)
    n_mat = [[0] * k for _ in range(k)]

    for i in range(k):
        corr_mat[i, i] = 1.0
        p_mat[i, i] = 0.0
        n_mat[i][i] = n
        for j in range(i + 1, k):
            xi = arr[:, i]
            xj = arr[:, j]
            if pairwise:
                keep = ~np.isnan(xi) & ~np.isnan(xj)
                # 成对完整案例
            else:
                keep = ~np.isnan(arr).any(axis=1)
                # 列表删除
            xi_clean = xi[keep]
            xj_clean = xj[keep]
            n_ij = int(np.sum(keep))

            if n_ij < 3:
                # 样本量太小，无法计算有意义的 CI
                corr_mat[i, j] = corr_mat[j, i] = float("nan")
                p_mat[i, j] = p_mat[j, i] = float("nan")
                n_mat[i][j] = n_mat[j][i] = n_ij
                continue

            r_val, p_val = corr_func(xi_clean, xj_clean)
            corr_mat[i, j] = corr_mat[j, i] = r_val
            p_mat[i, j] = p_mat[j, i] = p_val
            n_mat[i][j] = n_mat[j][i] = n_ij

    # ── Fisher z 变换 CI ──
    z_crit = sp_stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0)
    ci_low = np.full((k, k), np.nan)
    ci_up = np.full((k, k), np.nan)
    for i in range(k):
        ci_low[i, i] = 1.0
        ci_up[i, i] = 1.0
        for j in range(i + 1, k):
            r_ij = corr_mat[i, j]
            n_ij = n_mat[i][j]
            if np.isnan(r_ij) or n_ij < 3:
                continue
            z_val = _fisher_z(r_ij)
            se = 1.0 / math.sqrt(max(n_ij - 3, 1))
            ci_low[i, j] = ci_low[j, i] = _fisher_z_inv(z_val - z_crit * se)
            ci_up[i, j] = ci_up[j, i] = _fisher_z_inv(z_val + z_crit * se)

    # ── 多重比较校正 ──
    m_comparisons = int(k * (k - 1) / 2)  # 比较次数
    p_flat = _flatten_upper_triangle(p_mat)
    valid_mask = ~np.isnan(p_flat)

    if correction == "none":
        p_adj_flat = p_flat.copy()
    elif correction == "bonferroni":
        p_adj_flat = p_flat.copy()
        p_adj_flat[valid_mask] = _correction_bonferroni(p_flat[valid_mask], m_comparisons)
    elif correction == "holm":
        p_adj_flat = p_flat.copy()
        p_adj_flat[valid_mask] = _correction_holm(p_flat[valid_mask], m_comparisons)
    elif correction == "fdr_bh":
        p_adj_flat = p_flat.copy()
        p_adj_flat[valid_mask] = _correction_fdr_bh(p_flat[valid_mask], m_comparisons)
    else:
        raise ValueError(f"不支持的校正方法: '{correction}'。可选: none, bonferroni, holm, fdr_bh。")

    # ── 显著性矩阵 ──
    alpha = 1.0 - ci_level  # 默认 0.05
    sig_flat = p_adj_flat < alpha
    sig_flat[~valid_mask] = False
    sig_mat = _rebuild_symmetric(sig_flat.astype(float), k)
    sig_mat = sig_mat.astype(bool)

    return CorrelationResult(
        method=method,
        n_vars=k,
        n_samples=n_mat,
        corr_matrix=corr_mat,
        p_matrix=p_mat,
        ci_lower=ci_low,
        ci_upper=ci_up,
        significant_matrix=sig_mat,
        correction_method=correction,
        var_names=var_names,
    )


# ═══════════════════════════════════════════
# 单对相关便捷函数
# ═══════════════════════════════════════════


def correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "pearson",
    ci_level: float = 0.95,
) -> CorrelationPairResult:
    """单对相关分析 — 含 Fisher z 置信区间。

    便捷函数，用于只需要两个变量间关联的情况。

    Args:
        x: 一维数组（变量 X）。
        y: 一维数组（变量 Y），需与 x 等长。
        method: 相关方法。``"pearson"`` / ``"spearman"`` / ``"kendall"``。
        ci_level: 置信区间水平（默认 0.95）。

    Returns:
        CorrelationPairResult。
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    keep = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_arr, y_arr = x_arr[keep], y_arr[keep]
    n = len(x_arr)

    if n < 3:
        return CorrelationPairResult(
            method=method, r=float("nan"), p_value=float("nan"), n=n,
            ci_95=(float("nan"), float("nan")),
        )

    if method == "pearson":
        r_val, p_val = _pairwise_pearson(x_arr, y_arr)
    elif method == "spearman":
        r_val, p_val = _pairwise_spearman(x_arr, y_arr)
    elif method == "kendall":
        r_val, p_val = _pairwise_kendall(x_arr, y_arr)
    else:
        raise ValueError(f"不支持的相关方法: '{method}'。可选: pearson, spearman, kendall。")

    z_crit = sp_stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0)
    z_val = _fisher_z(r_val)
    se = 1.0 / math.sqrt(max(n - 3, 1))
    ci_low = _fisher_z_inv(z_val - z_crit * se)
    ci_high = _fisher_z_inv(z_val + z_crit * se)

    return CorrelationPairResult(
        method=method,
        r=r_val,
        p_value=float(p_val),
        n=n,
        ci_95=(ci_low, ci_high),
    )


# ═══════════════════════════════════════════
# 逐对相关计算（内部）
# ═══════════════════════════════════════════


def _pairwise_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r + t 检验 p 值。

    对两个一维清洁数组计算 Pearson 积矩相关。
    """
    r, p = sp_stats.pearsonr(x, y)
    return float(r), float(p)


def _pairwise_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman ρ + 秩相关 p 值。

    scipy 的 spearmanr 对无 ties 的数据使用精确分布，
    对大规模或有 ties 的数据使用 t 近似。
    """
    result = sp_stats.spearmanr(x, y)
    # scipy 1.9+ 返回 SignificanceResult，之前返回 tuple
    if hasattr(result, "correlation"):
        return float(result.correlation), float(result.pvalue)
    return float(result[0]), float(result[1])


def _pairwise_kendall(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Kendall τ-b + 双边 p 值。

    τ-b 专门修正了 ties 造成的不完美 concordance 的可能性。
    对于小样本 + 多 ties 的场景是最稳健的选择。
    """
    result = sp_stats.kendalltau(x, y)
    if hasattr(result, "correlation"):
        return float(result.correlation), float(result.pvalue)
    return float(result[0]), float(result[1])


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def correlation_report(r: CorrelationResult, alpha: float = 0.05) -> str:
    """相关矩阵报告文本。

    Args:
        r: CorrelationResult。
        alpha: 显著性水平（默认 0.05）。

    Returns:
        格式化报告字符串。
    """
    k = r.n_vars
    method_label = {"pearson": "Pearson r", "spearman": "Spearman ρ", "kendall": "Kendall τ-b"}.get(r.method, r.method)
    corr_label = {"bonferroni": "Bonferroni", "holm": "Holm", "fdr_bh": "Benjamini-Hochberg FDR", "none": "无"}.get(
        r.correction_method, r.correction_method
    )

    lines = [
        f"{'='*60}",
        f"  相关矩阵: {method_label}",
        f"  变量数={k}, 校正方法={corr_label}",
        f"{'='*60}",
        "",
        f"  {''.join(f'{v:>10}' for v in r.var_names)}",
        f"  {'-'*(10*k)}",
    ]

    for i in range(k):
        row_str = f"  {r.var_names[i]:<10}"
        for j in range(k):
            val = r.corr_matrix[i, j]
            sig = r.significant_matrix[i, j] if i != j else False
            if i == j:
                row_str += f"{'1.00':>10}"
            elif np.isnan(val):
                row_str += f"{'NA':>10}"
            else:
                mark = "*" if sig else " "
                row_str += f"{mark}{val:>9.3f}"
        lines.append(row_str)

    lines.append(f"  {'-'*(10*k)}")
    lines.append(f"  * p < {alpha} ({corr_label} 校正后)")

    # 有效样本量矩阵（如各对不同则显示范围）
    n_vals = []
    for i in range(k):
        for j in range(i + 1, k):
            v = r.n_samples[i][j]
            if not np.isnan(v):
                n_vals.append(v)
    if n_vals:
        n_min, n_max = int(min(n_vals)), int(max(n_vals))
        if n_min == n_max:
            lines.append(f"  每对有效 N={n_min}")
        else:
            lines.append(f"  每对有效 N={n_min}~{n_max}")

    lines.append(f"{'='*60}")
    return "\n".join(lines)


def correlation_pair_report(r: CorrelationPairResult) -> str:
    """单对相关报告文本。

    Args:
        r: CorrelationPairResult。

    Returns:
        格式化报告字符串。
    """
    method_label = {"pearson": "Pearson r", "spearman": "Spearman ρ", "kendall": "Kendall τ-b"}.get(r.method, r.method)
    return (
        f"{'='*50}\n"
        f"  {method_label} 相关\n"
        f"  n={r.n}, r={r.r:.4f}, p={r.p_value:.4f}\n"
        f"  95% CI: [{r.ci_95[0]:.4f}, {r.ci_95[1]:.4f}]\n"
        f"{'='*50}"
    )
