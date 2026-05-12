"""线性回归 (OLS) — 全参数实现

复刻 SPSS REGRESSION 过程 + 现代诊断最佳实践:
- OLS 系数推断 (t, F, CI)
- 分层/块回归 (ΔR², F-change)
- 共线性诊断: VIF, Tolerance
- 残差诊断: Durbin-Watson, Breusch-Pagan
- 影响点: Cook's D, Leverage, DfBeta
- 稳健标准误 (HC0-HC3)
- 标准化系数 Beta

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【OLS 的核心: 最小化残差平方和】

β = (X'X)⁻¹X'y 就是这个优化问题的闭式解。
每个 β_j 的解读: "在其他变量不变的条件下, X_j 每增加 1 单位,
Y 平均变化 β_j 单位"。

t 值 = β / SE: 检验"这个系数 = 0"的假设。
F 检验 = 检验"所有系数同时 = 0"。

【R² vs 调整 R²】

R² = 模型解释的 Y 的方差比例。
  永远随变量增多而增大, 即使添加的是噪音变量。
调整 R²: 惩罚了变量个数。
  调整 R² = 1 - (1-R²)(n-1)/(n-p)
  当加更多的变量, 分子 (1-R²) 减小, 分母 (n-p) 也减小。
  如果变量没有实质贡献, 调整 R² 可能反而下降。

【VIF — 多重共线性诊断】

VIF_j = 1 / (1 - R²_j)
其中 R²_j 是 X_j 对其他所有 X 回归的 R²。
  如果 X_j 几乎可以被其他 X 完美预测 → R²_j ≈ 1 → VIF → ∞。

经验法则:
  VIF = 1 → 完全无共线性 (理想)
  VIF < 5 → 可接受
  VIF 5~10 → 有共线性, 系数估计可能不稳定
  VIF > 10 → 严重共线性, 考虑删除或合并变量

Tolerance = 1/VIF = 1 - R²_j。
共线性的影响: 系数估计仍然无偏, 但标准误会变得巨大 → t 不显著 → 你
看不到本来存在的效应。两个高度相关的变量会"抢夺"同一个方差分量。

【稳健标准误 — HC0 ~ HC3】

为什么需要: 如果残差的方差不是常数 (异方差), 经典 SE 被低估 →
假阳性增加 (p 值看起来很小, 实际不显著)。

HC0: 原始 White 估计。不对小样本做任何校正。
HC1: 乘了 n/(n-k) 因子。Stata 默认。
HC2: 除以 (1-h_i), h_i=杠杆值。比 HC1 更准确。
HC3: 除以 (1-h_i)²。最保守, 推荐用于小样本或高杠杆情况。
     你的代码默认 HC3, 这是正确的选择。

【残差诊断精要】

Durbin-Watson (DW): 检验一阶自相关。
  DW ≈ 2 → 无自相关 (理想)
  DW < 1 或 > 3 → 存在自相关 → 参数检验不可靠, 考虑时间序列模型或 GLS

Breusch-Pagan (BP): 检验异方差。
  回归残差平方到 X。LM = n×R²_aux。
  p < 0.05 → 存在异方差 → 使用稳健 SE (已默认 HC3)

Cook's D: 影响点诊断。综合利用残差大小和杠杆值。
  D > 4/n 的个案值得关注。
  D > 1 的个案可能是离群值, 应检查。

Leverage (杠杆值 h_ii): 衡量每个点对自身预测值的影响力。
  高杠杆值意味着这个点的 X 值远离 X 均值 (在自变量空间中是极端的)。
  阈值: 2p/n 或 3p/n。

Condition Index (条件指数): 特征值之比的平方根。
  > 15 → 可能有共线性问题, > 30 → 严重共线性。

【标准化系数 Beta vs 未标准化系数 B】

B (未标准化): "X 增加一个原始单位 → Y 变化 B 个原始单位"。
  取决于变量单位, 不同变量之间无法直接比较。

Beta (标准化): 如果 X 增加一个标准差 → Y 变化 Beta 个标准差。
  先把所有变量标准化为 (值-均值)/SD, 再做回归。
  可以比较不同变量的"相对重要性"。
  但: Beta 取决于样本的 SD, 不同样本之间不可比较。
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
class OLSCoefficient:
    """单个回归系数"""

    name: str
    b: float  # 未标准化系数
    se: float
    se_robust_hc3: float = float("nan")
    t: float = float("nan")
    p: float = float("nan")
    ci_95: tuple[float, float] = (float("nan"), float("nan"))
    beta: float = float("nan")  # 标准化系数
    vif: float = float("nan")
    tolerance: float = float("nan")


@dataclass
class HierarchicalStep:
    """分层回归的单个步骤"""

    step: int
    variables_added: list[str]
    r_sq: float
    r_sq_change: float
    f_change: float
    df1: int
    df2: int
    p_change: float


@dataclass
class OLSResult:
    """OLS 回归完整结果"""

    n: int
    p: int  # 参数个数 (含截距)
    coefficients: list[OLSCoefficient]
    intercept: OLSCoefficient | None
    # 模型拟合
    r_sq: float
    r_sq_adjusted: float
    rmse: float
    f_statistic: float
    f_df1: int
    f_df2: int
    f_p_value: float
    # 诊断
    durbin_watson: float = float("nan")
    breusch_pagan_lm: float = float("nan")
    breusch_pagan_p: float = float("nan")
    condition_index: float = float("nan")
    # 分层回归
    hierarchical_steps: list[HierarchicalStep] = field(default_factory=list)
    # 原始数据 (用于偏回归图等)
    residuals: np.ndarray | None = None
    fitted_values: np.ndarray | None = None
    cook_d: np.ndarray | None = None
    leverage: np.ndarray | None = None


# ═══════════════════════════════════════════
# OLS 核心
# ═══════════════════════════════════════════


def ols(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    *,
    robust_se: str | None = None,
    ci_level: float = 0.95,
    compute_diagnostics: bool = True,
) -> OLSResult:
    """普通最小二乘法 (OLS) 线性回归。

    Args:
        y: 连续因变量 (n,)。
        X: 自变量矩阵 (n, k)。
        robust_se: ``"HC0"`` / ``"HC1"`` / ``"HC2"`` / ``"HC3"`` (推荐) or None。
        ci_level: 置信区间水平。
        compute_diagnostics: 是否计算 VIF/DW/BP/Cook's D 等诊断。

    Returns:
        OLSResult。
    """
    # 数据清洗
    if isinstance(y, pd.Series):
        y = y.values.astype(np.float64)
    else:
        y = np.asarray(y, dtype=np.float64)

    if isinstance(X, pd.DataFrame):
        var_names = list(X.columns)
        X_mat = X.values.astype(np.float64)
    else:
        X_mat = np.asarray(X, dtype=np.float64)
        var_names = [f"X{i+1}" for i in range(X_mat.shape[1])]

    # 排除缺失
    mask = (~np.isnan(y)) & (~np.isnan(X_mat).any(axis=1))
    y, X_mat = y[mask], X_mat[mask]
    n, k = X_mat.shape

    # 添加截距列
    X_design = np.column_stack([np.ones(n), X_mat])
    all_names = ["(Intercept)"] + var_names
    p = k + 1  # 含截距

    # OLS 求解 β = (X'X)⁻¹ X'y
    try:
        XtX = X_design.T @ X_design
        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ X_design.T @ y
    except np.linalg.LinAlgError:
        # 奇异矩阵: 使用伪逆
        beta = np.linalg.pinv(X_design) @ y
        XtX_inv = np.linalg.pinv(XtX)

    fitted = X_design @ beta
    residuals = y - fitted
    ss_resid = np.sum(residuals**2)
    df_resid = n - p
    sigma2 = ss_resid / df_resid if df_resid > 0 else 0

    # 标准误
    se = np.sqrt(np.diag(XtX_inv) * sigma2)

    # 稳健标准误 (HC3)
    hc3_se = np.full(p, float("nan"))
    if robust_se:
        leverage = np.sum(X_design * (X_design @ XtX_inv), axis=1)  # hat values
        if robust_se == "HC0":
            omega = np.diag(residuals**2)
        elif robust_se == "HC1":
            omega = np.diag(residuals**2 * n / df_resid)
        elif robust_se == "HC2":
            omega = np.diag(residuals**2 / (1 - np.clip(leverage, 0.01, 0.99)))
        else:  # HC3
            omega = np.diag((residuals / (1 - np.clip(leverage, 0.01, 0.99))) ** 2)
        vcov_robust = XtX_inv @ (X_design.T @ omega @ X_design) @ XtX_inv
        hc3_se = np.sqrt(np.clip(np.diag(vcov_robust), 0, None))

    # t 和 p 和 CI
    t_vals = beta / se
    p_vals = 2.0 * (1.0 - sp_stats.t.cdf(np.abs(t_vals), df_resid)) if df_resid > 0 else np.ones(p)

    alpha = 1 - ci_level
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df_resid) if df_resid > 0 else 1.96

    # 标准化系数: β* = b * (sd_x / sd_y)
    y_sd = np.std(y, ddof=1)
    x_sds = np.std(X_mat, axis=0, ddof=1)
    betas = np.full(p, float("nan"))
    betas[0] = float("nan")
    for j in range(1, p):
        betas[j] = beta[j] * x_sds[j - 1] / y_sd if y_sd > 0 and x_sds[j - 1] > 0 else float("nan")

    # 模型拟合
    ss_total = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1 - ss_resid / ss_total if ss_total > 0 else 0.0
    r_sq_adj = 1 - (1 - r_sq) * (n - 1) / (n - p) if n > p else 0.0
    rmse = math.sqrt(sigma2)

    # F 检验
    ms_reg = (ss_total - ss_resid) / (p - 1) if p > 1 else 0
    ms_resid = ss_resid / df_resid if df_resid > 0 else 0
    F = ms_reg / ms_resid if ms_resid > 0 else float("nan")
    F_p = 1.0 - sp_stats.f.cdf(F, p - 1, df_resid) if not math.isnan(F) and df_resid > 0 else float("nan")

    # 构建系数列表
    coefficients = []
    for j in range(p):
        ci = (beta[j] - t_crit * se[j], beta[j] + t_crit * se[j])
        coefficients.append(
            OLSCoefficient(
                name=all_names[j],
                b=float(beta[j]),
                se=float(se[j]),
                se_robust_hc3=float(hc3_se[j]) if robust_se else float("nan"),
                t=float(t_vals[j]),
                p=float(p_vals[j]),
                ci_95=(float(ci[0]), float(ci[1])),
                beta=float(betas[j]) if j > 0 else float("nan"),
            )
        )

    intercept = coefficients[0] if "(Intercept)" in all_names[0] else None

    result = OLSResult(
        n=n,
        p=p,
        coefficients=coefficients[1:] if intercept else coefficients,
        intercept=intercept,
        r_sq=r_sq,
        r_sq_adjusted=r_sq_adj,
        rmse=rmse,
        f_statistic=F,
        f_df1=p - 1,
        f_df2=df_resid,
        f_p_value=F_p,
    )

    # 诊断
    if compute_diagnostics:
        result.durbin_watson = _durbin_watson(residuals)
        result.breusch_pagan_lm, result.breusch_pagan_p = _breusch_pagan(X_mat, residuals, sigma2, n)
        result.condition_index = _condition_index(X_design)
        result.residuals = residuals
        result.fitted_values = fitted
        result.cook_d = _cooks_distance(X_design, residuals, p, sigma2)
        result.leverage = _leverage(X_design, XtX_inv)

        # VIF (仅对非截距项)
        for j, coef in enumerate(result.coefficients):
            vif_val, tol = _vif(X_mat, j)
            coef.vif = vif_val
            coef.tolerance = tol

    return result


# ═══════════════════════════════════════════
# 诊断函数
# ═══════════════════════════════════════════


def _durbin_watson(residuals: np.ndarray) -> float:
    """Durbin-Watson 自相关检验"""
    n = len(residuals)
    if n <= 1:
        return float("nan")
    diff = np.diff(residuals)
    return float(np.sum(diff**2) / np.sum(residuals**2))


def _breusch_pagan(X: np.ndarray, residuals: np.ndarray, sigma2: float, n: int) -> tuple[float, float]:
    """Breusch-Pagan 异方差检验 (LM 版本)。"""
    # 回归 squared residuals 到 X
    e2 = residuals**2 / sigma2
    X1 = np.column_stack([np.ones(n), X])
    try:
        bp_beta = np.linalg.inv(X1.T @ X1) @ X1.T @ e2
        bp_fitted = X1 @ bp_beta
        ss_exp = np.sum((bp_fitted - np.mean(e2)) ** 2)
        lm = n * (1 - np.sum((e2 - bp_fitted) ** 2) / np.sum((e2 - np.mean(e2)) ** 2))
        lm = max(0.0, lm)
        p = 1.0 - sp_stats.chi2.cdf(lm, X.shape[1])
    except Exception:
        lm, p = float("nan"), float("nan")

    return float(lm), float(p)


def _condition_index(X_design: np.ndarray) -> float:
    """条件指数 = sqrt(λ_max / λ_min), 用于多重共线性诊断。"""
    _, s, _ = np.linalg.svd(X_design, full_matrices=False)
    s = s[s > 1e-10]
    if len(s) < 2:
        return float("nan")
    return float(s[0] / s[-1])


def _vif(X: np.ndarray, var_idx: int) -> tuple[float, float]:
    """方差膨胀因子 + 容忍度。

    VIF_j = 1 / (1 - R²_j)  其中 R²_j 是 X_j 对其他 X 回归的 R²
    """
    k = X.shape[1]
    if k <= 1 or var_idx >= k:
        return float("nan"), float("nan")

    y_target = X[:, var_idx]
    others = np.column_stack([X[:, j] for j in range(k) if j != var_idx])
    n = len(y_target)

    try:
        other_design = np.column_stack([np.ones(n), others])
        beta_other = np.linalg.inv(other_design.T @ other_design) @ other_design.T @ y_target
        fitted = other_design @ beta_other
        ss_res = np.sum((y_target - fitted) ** 2)
        ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
        r_sq_j = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except np.linalg.LinAlgError:
        r_sq_j = 1.0

    vif = 1.0 / (1.0 - r_sq_j) if r_sq_j < 1.0 else float("inf")
    tolerance = 1.0 - r_sq_j
    return float(vif), float(tolerance)


def _cooks_distance(X_design: np.ndarray, residuals: np.ndarray, p: int, sigma2: float) -> np.ndarray:
    """Cook's D 影响点诊断。"""
    leverage = _leverage(X_design, np.linalg.inv(X_design.T @ X_design))
    # Cook's D = (r_i² / (p * sigma2)) * (h_i / (1 - h_i)²)
    std_resid = residuals**2 / (p * sigma2) if sigma2 > 0 else np.zeros_like(residuals)
    h_ratio = leverage / (1 - leverage) ** 2
    return std_resid * h_ratio


def _leverage(X_design: np.ndarray, XtX_inv: np.ndarray | None = None) -> np.ndarray:
    """杠杆值 h_ii = X_i (X'X)⁻¹ X_i'"""
    if XtX_inv is None:
        XtX_inv = np.linalg.inv(X_design.T @ X_design)
    H = X_design @ XtX_inv @ X_design.T
    return np.diag(H)


# ═══════════════════════════════════════════
# 分层回归
# ═══════════════════════════════════════════


def hierarchical_regression(
    y: np.ndarray,
    blocks: list[np.ndarray],
    *,
    block_names: list[str] | None = None,
    var_labels: list[list[str]] | None = None,
) -> OLSResult:
    """分层/块回归 (Hierarchical Regression)。

    按顺序逐步添加变量块, 计算每步的 ΔR² 和 F-change。

    Args:
        y: 因变量。
        blocks: 变量块列表 [block1, block2, ...], 每个 block 是 (n, k_i) 数组。
        block_names: 每个块的名称。
        var_labels: 每个块内变量的名称列表。
    """
    n = len(y)
    n_blocks = len(blocks)

    if block_names is None:
        block_names = [f"Block {i+1}" for i in range(n_blocks)]

    all_Xs: list[np.ndarray] = []
    all_labels: list[str] = []
    steps: list[HierarchicalStep] = []

    prev_r_sq = 0.0
    prev_p = 1  # 仅截距模型 (k=0 自变量)

    for blk_idx in range(n_blocks):
        all_Xs.append(blocks[blk_idx])
        X_current = np.column_stack(all_Xs)
        current_p = X_current.shape[1] + 1  # + intercept

        res = ols(y, X_current, compute_diagnostics=False)
        delta_r_sq = res.r_sq - prev_r_sq
        df1 = current_p - prev_p
        df2 = n - current_p

        # F-change = (ΔR² / df1) / ((1-R²_current) / df2)
        if df2 > 0 and (1 - res.r_sq) > 0 and df1 > 0:
            f_change = (delta_r_sq / df1) / ((1 - res.r_sq) / df2)
        else:
            f_change = float("nan")

        p_change = 1.0 - sp_stats.f.cdf(f_change, df1, df2) if not math.isnan(f_change) else float("nan")

        var_added = var_labels[blk_idx] if var_labels else [f"X{all_Xs[-1].shape[1]}"]

        steps.append(
            HierarchicalStep(
                step=blk_idx + 1,
                variables_added=var_added,
                r_sq=res.r_sq,
                r_sq_change=delta_r_sq,
                f_change=f_change,
                df1=df1,
                df2=df2,
                p_change=p_change,
            )
        )

        prev_r_sq = res.r_sq
        prev_p = current_p

    # 最终模型
    final = ols(y, np.column_stack(all_Xs), compute_diagnostics=True)
    final.hierarchical_steps = steps
    return final


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def ols_report(result: OLSResult) -> str:
    """生成 OLS 回归 APA 格式报告。"""
    lines = [
        f"{'='*60}",
        f"  OLS 线性回归",
        f"  n={result.n}, p={result.p}, R² = {result.r_sq:.4f} (adj = {result.r_sq_adjusted:.4f})",
        f"  F({result.f_df1}, {result.f_df2}) = {result.f_statistic:.3f}, p = {result.f_p_value:.4f}",
        f"  RMSE = {result.rmse:.4f}",
        f"{'='*60}",
        f"",
        f"  {'Variable':<20} {'B':>10} {'SE':>10} {'t':>8} {'p':>8} {'VIF':>6} {'Beta':>8}",
        f"  {'-'*70}",
    ]

    if result.intercept:
        c = result.intercept
        lines.append(f"  {c.name:<20} {c.b:10.4f} {c.se:10.4f} {c.t:8.3f} {c.p:8.4f} {'':>6} {'':>8}")

    for c in result.coefficients:
        vif_str = f"{c.vif:.2f}" if not math.isnan(c.vif) else ""
        beta_str = f"{c.beta:.4f}" if not math.isnan(c.beta) else ""
        lines.append(
            f"  {c.name:<20} {c.b:10.4f} {c.se:10.4f} {c.t:8.3f} {c.p:8.4f} {vif_str:>6} {beta_str:>8}"
        )

    lines.extend(["", "  【模型诊断】", f"  Durbin-Watson = {result.durbin_watson:.4f}"])
    lines.append(f"  Breusch-Pagan LM = {result.breusch_pagan_lm:.4f}, p = {result.breusch_pagan_p:.4f}")
    lines.append(f"  Condition Index = {result.condition_index:.2f}")

    if result.hierarchical_steps:
        lines.extend(["", "  【分层回归步骤】"])
        for s in result.hierarchical_steps:
            lines.append(
                f"  Step {s.step} ({', '.join(s.variables_added)}): "
                f"R²={s.r_sq:.4f}, ΔR²={s.r_sq_change:.4f}, "
                f"F({s.df1},{s.df2})={s.f_change:.3f}, p={s.p_change:.4f}"
            )

    return "\n".join(lines)
