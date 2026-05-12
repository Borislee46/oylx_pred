"""探索性因子分析 (EFA) — 全参数实现

复刻 SPSS FACTOR 过程:
- 适用性: KMO (整体+单变量), Bartlett 球形检验
- 提取: PAF (主轴因子法), ML (最大似然法), PCA
- 旋转: Varimax (正交), Promax (斜交)
- 辅助: 平行分析 (Parallel Analysis), 碎石图数据
- 载荷矩阵: Pattern Matrix + Structure Matrix (斜交)

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【EFA vs PCA — 本质区别】

PCA (主成分分析):
  目标: 最大化方差解释。把所有方差 (共同方差 + 独特方差 + 误差) 都算进去。
  数学: 没有"误差项"概念, 就是把原始变量线性组合成新变量。
  使用场景: 数据降维 (减少变量个数), 而非寻找潜变量。

EFA (探索性因子分析):
  目标: 寻找潜在的不可观测的"因子"来解释观测到的相关性模式。
  数学: X = Λ × F + ε  (观测值 = 载荷 × 因子得分 + 独特方差)
  使用场景: 量表开发、心理测量、寻找理论构念。

SPSS 的 FACTOR 菜单默认提供 PCA, 但 PCA ≠ 因子分析。
论文中如果想要"降维"用 PCA, 想要"发现潜在构念"用 EFA。

【KMO 抽样适切性 — 数据能做因子分析吗】

KMO = Σ(r²) / [Σ(r²) + Σ(p²)]
r = 相关系数, p = 偏相关系数 (控制了所有其他变量的相关性)

直觉: 如果 KMO 高, 说明相关主要由"变量之间的共享关系"驱动,
而不是"别的变量没控制好"。如果偏相关很大 (控制其他变量后仍有强相关),
暗示数据不适合因子分析。

∞≥0.9=优秀, 0.8=良好, 0.7=中等, 0.6=勉强, <0.5=不要做因子分析。

单变量 KMO (对角线): 每个变量单独看。如果某个变量 KMO 极低 (<0.5),
可能要删除。

Bartlett 球形检验: H₀ = 相关矩阵是单位矩阵 (变量间完全独立)。
  p<0.05 → 至少有一些相关, 数据可以做因子分析。

【因子提取方法】

PAF (Principal Axis Factoring, 主轴因子法): 关注共同方差, 推荐默认。
  迭代更新共同度, 直到收敛。
  优势: 不做正态假设。即使数据严重偏态也稳健。
  这是大多数社会科学论文的首选方法。

ML (Maximum Likelihood, 最大似然法): 假设数据正态。
  优势: 提供拟合优度检验 (χ² 检验, H₀: k 个因子足够),
  允许计算标准误和比较嵌套模型。
  劣势: 正态假设强, 非正态时可能不收敛或 Heywood 情况 (共同度 > 1)。

PCA: 技术上不是因子分析。共同度=1 (假设所有方差都是共同方差)。
  更容易收敛, 但理论上不严谨。

【旋转 — 正交 vs 斜交】

旋转的目的是让载荷矩阵"简单" (每行只在少数因子上有高载荷),
使因子更容易命名和解读。

Varimax (正交): 因子间不相关。
  ═约束: 因子都互相垂直。
  问题: 这个假设在心理学/社会科学里几乎从不成立。
  大部分真实世界的心理构念 (如焦虑和抑郁) 是相关的。

Promax (斜交, 推荐): 允许因子相关。
  两步: 先做 Varimax, 再"倾斜"载荷矩阵使得高载荷更高, 低载荷更低。
  你需要报告两个矩阵:
    Pattern Matrix (载荷): "每个变量独立对每个因子的贡献"。
      类似回归的"偏相关系数"。
    Structure Matrix: Pattern Matrix × 因子相关矩阵。
      类似"简单相关系数"。变量和因子的整体关联。
  通常报告Pattern Matrix, 因为"独立贡献"更容易解读。
  并报告因子相关矩阵 (Φ), 看因子之间是否确实相关。

  如果因子间相关性很低 (<0.2), 报告 Varimax 也可以。

Kappa 参数 (默认 4): 越高 → 越极端 → 因子区分度越强, 但可能失真。

【平行分析 — 决定因子数】

Horn (1965) 的方法: 生成随机数据 (n × k, 每个变量都是独立标准正态),
计算其特征值。重复多次, 取每个分位点的 95 分位特征值。
真正的因子数 = 数据特征值 > 随机数据 95 分位特征值的个数。

这是现今学术界公认的确定因子数的最佳方法之一。
优于 Kaiser 标准 (特征值 > 1, 是粗略的直觉法则)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.linalg import eigh, svd


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════


@dataclass
class FactorResult:
    """EFA 完整结果"""

    n: int
    k: int  # 变量数
    n_factors: int
    # 适用性
    kmo_overall: float
    kmo_per_variable: list[float]
    bartlett_chi2: float
    bartlett_df: int
    bartlett_p: float
    # 提取
    extraction_method: str
    eigenvalues: np.ndarray
    variance_explained: np.ndarray  # 各因子方差解释比例
    cumulative_variance: np.ndarray
    communalities: np.ndarray  # 提取后共同度
    # 载荷
    loadings: np.ndarray  # Pattern Matrix (斜交) 或 旋转后的载荷矩阵
    structure_matrix: np.ndarray | None = None  # Structure Matrix (仅斜交)
    factor_correlations: np.ndarray | None = None  # 斜交旋转后的因子相关矩阵
    rotation: str = "none"
    # 平行分析
    parallel_eigenvalues: np.ndarray | None = None
    # 变量名
    variable_names: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════
# KMO & Bartlett
# ═══════════════════════════════════════════


def kmo_test(corr: np.ndarray) -> tuple[float, np.ndarray]:
    """KMO 抽样适切性检验。

    Returns:
        (overall_kmo, per_variable_kmo)
    """
    k = corr.shape[0]
    # 偏相关矩阵
    inv_corr = np.linalg.inv(corr)
    diag_inv = np.diag(inv_corr)
    # 反影像相关矩阵
    aic = np.zeros_like(corr)
    for i in range(k):
        for j in range(k):
            aic[i, j] = -inv_corr[i, j] / math.sqrt(diag_inv[i] * diag_inv[j])
    np.fill_diagonal(aic, 1.0)

    # 配对相关的平方和
    r2_sum = 0.0
    p2_sum = 0.0
    per_kmo = np.zeros(k)
    for i in range(k):
        r2_i = 0.0
        p2_i = 0.0
        for j in range(k):
            if i != j:
                r2_i += corr[i, j] ** 2
                p2_i += aic[i, j] ** 2
        r2_sum += r2_i
        p2_sum += p2_i
        per_kmo[i] = r2_i / (r2_i + p2_i) if (r2_i + p2_i) > 0 else 0.0

    overall = r2_sum / (r2_sum + p2_sum) if (r2_sum + p2_sum) > 0 else 0.0
    return float(overall), per_kmo


def bartlett_test(corr: np.ndarray, n: int) -> tuple[float, int, float]:
    """Bartlett 球形检验。

    H₀: 总体相关矩阵 = I (变量间独立)
    """
    k = corr.shape[0]
    # χ² = -(n - 1 - (2k + 5) / 6) * ln|R|
    det = np.linalg.det(corr)
    if det <= 0:
        return float("inf"), k * (k - 1) // 2, 0.0
    chi2 = -(n - 1 - (2 * k + 5) / 6) * math.log(det)
    df = k * (k - 1) // 2
    p = 1.0 - sp_stats.chi2.cdf(chi2, df) if df > 0 else 1.0
    return float(chi2), int(df), float(p)


# ═══════════════════════════════════════════
# 因子提取
# ═══════════════════════════════════════════


def _eigen_decomp(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """特征值分解"""
    k = corr.shape[0]
    eigvals, eigvecs = eigh(corr)
    # 降序
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]


def _extract_paf(corr: np.ndarray, n_factors: int, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """主轴因子法 (Principal Axis Factoring)。

    迭代更新共同度: h² = 1 - diag(R - ΛΛ')
    """
    k = corr.shape[0]
    # 初始共同度: SMC (squared multiple correlation)
    communalities = np.zeros(k)
    for i in range(k):
        others = [j for j in range(k) if j != i]
        b = np.linalg.inv(corr[np.ix_(others, others)]) @ corr[others, i]
        communalities[i] = max(0.01, min(0.99, corr[others, i] @ b))

    for _ in range(max_iter):
        # R* = R with diag replaced by communalities
        R_star = corr.copy()
        np.fill_diagonal(R_star, communalities)

        eigvals, eigvecs = _eigen_decomp(R_star)
        eigvals = np.maximum(eigvals[:n_factors], 1e-10)
        L = eigvecs[:, :n_factors] * np.sqrt(eigvals[:n_factors])

        # 新共同度
        new_comm = np.sum(L**2, axis=1)
        if np.max(np.abs(new_comm - communalities)) < 1e-4:
            break
        communalities = new_comm

    return L, eigvals[:n_factors], communalities


def _extract_ml(corr: np.ndarray, n_factors: int, n: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """最大似然因子提取 (Lawley-Maxwell 迭代)。

    最小化: F = tr(R⁻¹ S) - log|R⁻¹ S| - k
    """
    k = corr.shape[0]
    # 初始化: 使用 PAF 结果
    L, evals, comm = _extract_paf(corr, n_factors)

    R = corr.copy()
    for _ in range(max_iter):
        # 唯一方差 Psi = I - diag(LL')
        psi = np.maximum(1.0 - np.sum(L**2, axis=1), 1e-6)
        psi_sqrt = np.sqrt(psi)
        psi_inv_sqrt = 1.0 / psi_sqrt

        # 缩放相关矩阵 R* = Psi⁻½ R Psi⁻½
        R_psi = np.diag(psi_inv_sqrt) @ R @ np.diag(psi_inv_sqrt)

        eigvals, eigvecs = _eigen_decomp(R_psi)
        eigvals = np.maximum(eigvals[:n_factors] - 1.0, 1e-10)

        # 新载荷 L_new = Psi½ * V * sqrt(Λ - I)
        L_new = np.diag(psi_sqrt) @ eigvecs[:, :n_factors] * np.sqrt(eigvals[:n_factors])

        diff = np.max(np.abs(L_new - L))
        L = L_new
        if diff < 1e-5:
            break

    communalities = np.sum(L**2, axis=1)
    return L, eigvals[:n_factors], communalities


# ═══════════════════════════════════════════
# 旋转
# ═══════════════════════════════════════════


def _varimax_rotation(L: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Varimax 正交旋转 (Kaiser 1958)。"""
    k, m = L.shape
    if m < 2:
        return L.copy()

    L_rot = L.copy()
    for _ in range(max_iter):
        h2 = np.sum(L_rot**2, axis=1, keepdims=True)
        U = L_rot / np.sqrt(np.maximum(h2, 1e-10))

        for i in range(m - 1):
            for j in range(i + 1, m):
                ui, uj = U[:, i], U[:, j]
                a = ui**2 - uj**2
                b = 2 * ui * uj
                num = 4 * np.sum(a * b) - 4 * np.sum(a) * np.sum(b) / k
                denom = np.sum((a + b) * (a - b)) - (np.sum(a)**2 - np.sum(b)**2) / k
                phi = math.atan2(num, max(denom, 1e-10)) / 4

                cos_phi, sin_phi = math.cos(phi), math.sin(phi)
                rot = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
                L_rot[:, [i, j]] = L_rot[:, [i, j]] @ rot.T

    return L_rot


def _promax_rotation(L: np.ndarray, kappa: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Promax 斜交旋转 (Hendrickson & White 1964)。

    Returns:
        (pattern_matrix, structure_matrix, factor_correlations)
    """
    m = L.shape[1]
    if m < 2:
        return L, L, np.eye(m)

    # Step 1: Varimax 旋转
    L_varimax = _varimax_rotation(L)

    # Step 2: 构造目标矩阵 P = |L_varimax|^{kappa+1} / L_varimax
    P = np.abs(L_varimax) ** (kappa + 1) / np.maximum(np.abs(L_varimax), 1e-10)
    P = np.clip(P, -100, 100)

    # Step 3: 最小二乘拟合 P = L_varimax @ T
    # T = (L'L)⁻¹ L' P
    try:
        T = np.linalg.inv(L_varimax.T @ L_varimax) @ L_varimax.T @ P
    except np.linalg.LinAlgError:
        T = np.eye(m)

    # 列归一化 T
    for j in range(m):
        norm = np.sqrt(np.sum(T[:, j] ** 2))
        if norm > 1e-10:
            T[:, j] /= norm

    # Pattern matrix
    pattern = L_varimax @ T

    # Factor correlations: Φ = (T'T)⁻¹
    try:
        phi = np.linalg.inv(T.T @ T)
    except np.linalg.LinAlgError:
        phi = np.eye(m)

    # Structure matrix: S = Pattern @ Φ
    structure = pattern @ phi

    return pattern, structure, phi


# ═══════════════════════════════════════════
# 平行分析
# ═══════════════════════════════════════════


def parallel_analysis(
    corr: np.ndarray, n: int, n_simulations: int = 100, random_seed: int | None = 42
) -> np.ndarray:
    """平行分析 (Horn 1965): 随机数据的特征值 95 分位点。

    Returns:
        每个分位点的平均/95分位特征值。
    """
    k = corr.shape[0]
    rng = np.random.default_rng(random_seed)
    sim_evals = np.zeros((n_simulations, k))

    for s in range(n_simulations):
        sim_data = rng.normal(0, 1, (n, k))
        sim_corr = np.corrcoef(sim_data, rowvar=False)
        evals, _ = _eigen_decomp(sim_corr)
        sim_evals[s, :] = evals[:k]

    # 95 分位
    return np.percentile(sim_evals, 95, axis=0)


# ═══════════════════════════════════════════
# EFA 主接口
# ═══════════════════════════════════════════


def factor_analysis(
    data: np.ndarray | pd.DataFrame,
    n_factors: int,
    *,
    extraction: str = "paf",
    rotation: str = "promax",
    promax_kappa: float = 4.0,
    parallel_n_sim: int = 0,
    random_seed: int | None = 42,
) -> FactorResult:
    """探索性因子分析 (EFA)。

    Args:
        data: (n, k) 数组或 DataFrame。
        n_factors: 提取因子数。
        extraction: ``"paf"`` (主轴因子法) / ``"ml"`` (最大似然) / ``"pca"``。
        rotation: ``"varimax"`` (正交) / ``"promax"`` (斜交, 推荐) / ``"none"``。
        promax_kappa: Promax 的 kappa 参数 (默认 4)。
        parallel_n_sim: 平行分析模拟数 (0=不执行)。
        random_seed: 随机种子。

    Returns:
        FactorResult。
    """
    if isinstance(data, pd.DataFrame):
        var_names = list(data.columns)
        arr = data.values.astype(np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
        var_names = [f"V{i+1}" for i in range(arr.shape[1])]

    # 移除含 NaN 的行 (listwise)
    arr = arr[~np.isnan(arr).any(axis=1)]
    n, k = arr.shape

    # 相关矩阵
    corr = np.corrcoef(arr, rowvar=False)

    # KMO
    kmo_overall, kmo_per = kmo_test(corr)

    # Bartlett
    bart_chi2, bart_df, bart_p = bartlett_test(corr, n)

    # 提取
    if extraction == "pca":
        eigvals, eigvecs = _eigen_decomp(corr)
        m = min(n_factors, k)
        L = eigvecs[:, :m] * np.sqrt(np.maximum(eigvals[:m], 0))
        communalities = np.sum(L**2, axis=1)
        extraction_label = "PCA"
    elif extraction == "ml":
        L, evals, communalities = _extract_ml(corr, n_factors, n)
        m = n_factors
        eigvals = np.zeros(k)
        eigvals[:m] = evals
        extraction_label = "Maximum Likelihood"
    else:  # paf
        L, evals, communalities = _extract_paf(corr, n_factors)
        m = n_factors
        eigvals = np.zeros(k)
        eigvals[:m] = evals
        extraction_label = "Principal Axis Factoring"

    # 方差解释
    var_exp = eigvals[:k] / k * 100
    cum_var = np.cumsum(var_exp)

    # 旋转
    pattern = L.copy()
    structure = None
    factor_corr = None
    rotation_label = "none"

    if rotation == "varimax" and m >= 2:
        pattern = _varimax_rotation(L)
        rotation_label = "Varimax (正交)"
    elif rotation == "promax" and m >= 2:
        pattern, structure, factor_corr = _promax_rotation(L, promax_kappa)
        rotation_label = f"Promax (斜交, κ={promax_kappa})"

    # 平行分析
    parallel_evals = None
    if parallel_n_sim > 0:
        parallel_evals = parallel_analysis(corr, n, parallel_n_sim, random_seed)

    return FactorResult(
        n=n,
        k=k,
        n_factors=m,
        kmo_overall=kmo_overall,
        kmo_per_variable=kmo_per.tolist() if isinstance(kmo_per, np.ndarray) else list(kmo_per),
        bartlett_chi2=bart_chi2,
        bartlett_df=bart_df,
        bartlett_p=bart_p,
        extraction_method=extraction_label,
        eigenvalues=eigvals[:k],
        variance_explained=var_exp[:k],
        cumulative_variance=cum_var[:k],
        communalities=communalities,
        loadings=pattern,
        structure_matrix=structure,
        factor_correlations=factor_corr,
        rotation=rotation_label,
        parallel_eigenvalues=parallel_evals,
        variable_names=var_names,
    )


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def factor_report(result: FactorResult) -> str:
    """EFA 报告文本。"""
    lines = [
        f"{'='*60}",
        f"  探索性因子分析 (EFA)",
        f"  n={result.n}, k={result.k}, factors={result.n_factors}",
        f"  提取: {result.extraction_method}, 旋转: {result.rotation}",
        f"{'='*60}",
        "",
        f"  KMO = {result.kmo_overall:.4f}  ({'优秀' if result.kmo_overall > 0.8 else ('良好' if result.kmo_overall > 0.7 else ('一般' if result.kmo_overall > 0.6 else '不足'))})",
        f"  Bartlett χ²({result.bartlett_df}) = {result.bartlett_chi2:.4f}, p = {result.bartlett_p:.4f}",
        "",
        f"  {'因子':<10} {'特征值':>8} {'方差%':>8} {'累积%':>8}",
    ]
    for j in range(len(result.eigenvalues)):
        lines.append(
            f"  {j+1:<10} {result.eigenvalues[j]:8.3f} {result.variance_explained[j]:8.2f} {result.cumulative_variance[j]:8.2f}"
        )

    lines.extend(["", f"  【载荷矩阵 ({result.rotation})】", ""])
    header = f"  {'变量':<12}"
    for j in range(result.n_factors):
        header += f" {'F'+str(j+1):>8}"
    header += f" {'共同度':>8}"
    lines.append(header)
    lines.append(f"  {'-'*50}")

    loadings = result.loadings
    for i in range(result.k):
        row = f"  {result.variable_names[i]:<12}"
        for j in range(result.n_factors):
            row += f" {loadings[i, j]:8.3f}"
        row += f" {result.communalities[i]:8.3f}"
        lines.append(row)

    if result.factor_correlations is not None:
        lines.extend(["", "  【因子相关矩阵 (Φ)】", ""])
        fc = result.factor_correlations
        for i in range(result.n_factors):
            row = f"  F{i+1}: "
            for j in range(result.n_factors):
                row += f" {fc[i, j]:7.3f}"
            lines.append(row)

    if result.parallel_eigenvalues is not None:
        lines.extend(["", "  【平行分析 (95 分位特征值)】", ""])
        for j in range(min(len(result.parallel_eigenvalues), len(result.eigenvalues))):
            lines.append(
                f"  F{j+1}: 数据={result.eigenvalues[j]:.3f}, 随机95%={result.parallel_eigenvalues[j]:.3f} "
                f"{'✓' if result.eigenvalues[j] > result.parallel_eigenvalues[j] else ''}"
            )

    return "\n".join(lines)
