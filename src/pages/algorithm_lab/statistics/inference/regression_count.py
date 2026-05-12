"""计数回归 — Poisson / Negative Binomial 全参数实现

复刻 SPSS GENLIN (Generalized Linear Models) + 负二项扩展:
- Poisson 回归 (IRLS 迭代)
- 负二项回归 NB2 (交替 IRLS + α 估计)
- 过离散检验 (dispersion + score test)
- IRR (发生率比) + CI
- offset / exposure 支持

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【为什么 OLS 不适合计数数据】

计数数据（离职人数、投诉量、申请数、事故次数）有三个特征
让 OLS 完全不适合：

1. 非负性：计数 ≥ 0。OLS 可能预测出负数（如预期离职 -3.2 人）。
2. 离散性：计数只能是整数（0, 1, 2, ...）。OLS 假定连续值。
3. 均值-方差关系：计数越大，方差也越大。OLS 假定方差恒定。

试想：一个部门平均每月 2 起投诉，方差约 2。
      另一个部门平均每月 50 起投诉，方差约 50。
      OLS 要求两边方差相等 → 必然低估高计数部门的变异。

Poisson 回归直接建模"计数"而非"连续响应"，
通过 log 连接函数保证预测值为正。

【Poisson 分布的假设】

    P(Y=y) = (exp(-μ) × μ^y) / y!
    其中 μ = E[Y] = Var[Y]

Poisson 有一个强制性假定：均值 = 方差。
当方差 > 均值时，存在"过离散"(overdispersion)。
过离散不会使 Poisson 的系数偏掉（仍然是无偏估计），
但会使标准误偏小 → p 值假性显著 → 错误地宣称存在效果。

现实数据几乎总是存在过离散（除非数据来自严格的随机过程）。

【log 连接函数 → 乘法效应的世界】

    ln(μ) = X × β
    μ = exp(X × β)

对于 X 增加 1 单位：
    μ_new / μ_old = exp(β × (X+1)) / exp(β × X) = exp(β)

    exp(β) = IRR（发生率比, Incidence Rate Ratio）

IRR 的解读：
    IRR = 1.0 → X 无影响
    IRR = 1.5 → X 增加 1，期望计数乘以 1.5（增加 50%）
    IRR = 0.7 → X 增加 1，期望计数乘以 0.7（减少 30%）

注意：IRR 是乘法效应！不是加法。
     X 从 0 到 1：μ = 100 × exp(0.3) = 135（增加 35）
     X 从 10 到 11：μ = 100 × exp(3.0) ÷ exp(2.7) ≈ 仍然增加 35

这个性质有时被称为"等比增加"。

【offset / exposure — 当观测单位不等时】

不同的部门有不同的人数 → 离职数不可直接比较。
解决：用 log(部门人数) 作为 offset。

    ln(μ_i) = X_i × β + ln(exposure_i)
    → ln(μ_i / exposure_i) = X_i × β
    → μ_i / exposure_i = exp(X_i × β) = 率（rate）
    → μ_i = exposure_i × rate_i

直觉：
    如果销售部（100 人）和研发部（20 人）的各项 X 相同，
    有 offset 的模型预测的离职人数是 5:1 的关系（因为人数比 5:1），
    这才是公平的比较。

offset 的系数被固定为 1（不是估计的），因为它来自已知的基数。
这不同于普通自变量。

【过离散的诊断 — 为什么标准误会偏】

Poisson 假定 Var(Y) = μ
实际数据往往 Var(Y) = μ + α × μ²（NB2 形式）
                       或 Var(Y) = μ × φ（quasi-Poisson 形式）

过离散参数 = Pearson χ² / (n - p)
    φ ≈ 1 → 无过离散，Poisson 适用
    φ > 1.5 → 中等过离散，建议 NB
    φ > 2.0 → 严重过离散，Poisson 的 p 值不可靠

标准误偏差：
    如果 φ = 2.0（方差是均值的两倍）：
    Poisson 的 SE = 真实的 SE × 0.707
                  = 低估了约 30%
    → t 统计量虚高约 1.41 倍
    → p 值远低于真实值
    → 大量假阳性

【负二项回归 — 多了 alpha 参数吸收额外变异】

NB2（最常用的负二项参数化）：
    μ_i = exp(X_i × β)
    Var(Y_i) = μ_i + α × μ_i²

α = 0 → Poisson（特殊情形）
α > 0 → 过离散（α 越大，过离散越严重）

直觉：
    Poisson 说"均值 = 方差"（唯一指定）。
    NB 说"方差 = 均值 + alpha × 均值²"（多了一个调参自由度）。
    这个额外的参数 α 吸收了被 Poisson 遗漏的变异。
    所以 NB 的系数和 Poisson 基本相同，但 SE 更宽（更诚实）。

估计方法（交替估计）：
    1. 初始：用 Poisson 估计 β^，Pearson χ² 估计 α
    2. IRLS：固定 α，用 W = diag(μ / (1 + α × μ)) 更新 β
    3. 更新 α：用 score equation 或 moment estimator
    4. 重复 2-3 直到收敛

【Poisson vs NB — 决策准则】

- 先跑 Poisson
- 检查 dispersion (Pearson χ²/df)
  → ≈ 1.0：过离散 p > 0.05 → 用 Poisson（更简单）
  → > 1.5 或 p < 0.05 → 用 NB（需要修正过离散）
- 比较 AIC：
  → NB 的 AIC 更小 → 值得付出额外参数的代价
  → Poisson 的 AIC 更小 → 过离散不严重，Poisson 就行

【IRLS 算法 — 为什么它有效】

IRLS = Iteratively Reweighted Least Squares（迭代重加权最小二乘）
本质：把非线性的 GLM 问题转化为一系列加权的线性 OLS 问题。

每步迭代：
    1. 用当前 β 算出 μ
    2. 基于 μ 的方差函数计算权重 W
       Poisson: W = diag(μ)（方差 = 均值）
       NB: W = diag(μ/(1 + α×μ))（方差 = μ + αμ²）
    3. 构造"工作响应变量" z = η + (y-μ)/μ
       这个 z 是"如果数据真的是线性的，观察值应该是什么"
    4. 用 WLS 解出新的 β：β = (X'WX)⁻¹ X'Wz
    5. 重复直到收敛

这个算法比 Newton-Raphson 更稳定（不需要二阶 Hessian），
特别适合 GLM 这类方差是均值函数的模型。
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
class CountCoefficient:
    """计数回归单个系数"""

    name: str  # 变量名
    b: float  # 未标准化系数 (log 尺度)
    se: float  # 标准误
    z: float  # Wald z 统计量
    p: float  # p 值
    irr: float  # 发生率比 = exp(b)
    irr_ci_95: tuple[float, float]  # IRR 的 95% CI = exp(b ± z_crit × SE)


@dataclass
class CountRegressionResult:
    """计数回归完整结果"""

    n: int  # 有效样本量
    p: int  # 参数个数 (含截距)
    coefficients: list[CountCoefficient]  # 自变量系数
    intercept: CountCoefficient | None  # 截距项
    log_likelihood: float  # 对数似然
    aic: float  # AIC = -2LL + 2k
    bic: float  # BIC = -2LL + k ln(n)
    deviance: float  # 偏差 D = 2 Σ[y_i ln(y_i/μ_i) - (y_i-μ_i)]
    pearson_chi2: float  # Pearson χ² = Σ(y-μ)²/μ
    dispersion: float  # 过离散参数 = Pearson χ² / (n-p)
    p_overdispersion: float  # 过离散检验 p 值 (score test)
    converged: bool  # 是否收敛
    iterations: int  # 迭代次数
    method: str  # "poisson" | "negative_binomial"
    alpha: float | None = None  # NB 离散参数 α（仅 NB）
    alpha_se: float | None = None  # α 的标准误（仅 NB）
    predicted_counts: np.ndarray | None = None  # 预测计数


# ═══════════════════════════════════════════
# 内部：Poisson 对数似然
# ═══════════════════════════════════════════


def _poisson_loglik(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson 对数似然 = Σ [y*ln(μ) - μ - ln(y!)]。

    Stirling 近似用于大 y 时的 ln(y!) 。
    """
    ll = y * np.log(np.maximum(mu, 1e-15)) - mu
    # 减去 ln(y!)，用 scipy 的 gammaln 精确计算（gammaln(y+1) = ln(y!)）
    from scipy.special import gammaln
    ll -= gammaln(y + 1.0)
    return float(np.sum(ll))


def _negbin_loglik(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """负二项 (NB2) 对数似然。

    NB2 的概率质量函数 = Γ(y+1/α) / (Γ(y+1)·Γ(1/α))
                        × (αμ/(1+αμ))^y × (1/(1+αμ))^(1/α)

    对数化后 = gammaln(y+1/α) - gammaln(y+1) - gammaln(1/α)
              + y*ln(αμ) - (y+1/α)*ln(1+αμ)
    """
    from scipy.special import gammaln
    a_inv = 1.0 / alpha
    ll = (
        gammaln(y + a_inv)
        - gammaln(y + 1.0)
        - gammaln(a_inv)
        + y * np.log(np.maximum(alpha * mu, 1e-15))
        - (y + a_inv) * np.log(1.0 + alpha * mu)
    )
    return float(np.sum(ll))


# ═══════════════════════════════════════════
# Poisson 回归 (IRLS)
# ═══════════════════════════════════════════


def poisson_regression(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    *,
    ci_level: float = 0.95,
    exposure: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> CountRegressionResult:
    """Poisson 回归 — 用于计数因变量的广义线性模型。

    通过 IRLS (Iteratively Reweighted Least Squares) 最大化
    Poisson 似然。IRLS 对 GLM 特别稳定，不需要二阶导。

    适用场景：
    - 离职人数、投诉量、申请数、事故次数等计数变量
    - 因变量是非负整数，方差随均值增大而增大
    - 需要乘法效应解释（IRR = exp(B)）

    不适用场景：
    - 连续非负变量（考虑 Gamma 回归，本库未实现）
    - 有大量过离散的数据 → 考虑 negative_binomial_regression()
    - 0 过多（zero-inflation）→ 需专门的零膨胀模型

    Args:
        y: 因变量，非负整数计数。可含缺失值。
        X: 自变量矩阵。支持 DataFrame 和 ndarray。
        ci_level: IRR 置信区间的置信水平（默认 0.95）。
        exposure: 暴露量（如各部门人数、各群体的观测时长）。
            传入后会作为 offset 加入模型：ln(μ) = XB + ln(exposure)。
            此时模型解释为"率"（rate = count / exposure）而非"计数"。
            如果不传 exposure，模型直接建模原始计数。
        max_iter: IRLS 最大迭代次数。
        tol: 收敛容差。

    Returns:
        CountRegressionResult (method="poisson")。

    Example:
        # 建模部门离职人数，用部门规模作为 exposure
        >>> result = poisson_regression(
        ...     y=df["turnover_count"],
        ...     X=df[["satisfaction", "tenure_avg"]],
        ...     exposure=df["headcount"],
        ... )
        >>> result.coefficients[0].irr  # satisfaction 增加 1 单位的 IRR
    """
    return _count_regression_irls(
        y, X, ci_level=ci_level, exposure=exposure,
        max_iter=max_iter, tol=tol, model="poisson",
    )


# ═══════════════════════════════════════════
# 负二项回归 (NB2, 交替 IRLS)
# ═══════════════════════════════════════════


def negative_binomial_regression(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    *,
    ci_level: float = 0.95,
    exposure: np.ndarray | None = None,
    alpha_init: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> CountRegressionResult:
    """负二项回归 (NB2) — 处理过离散的计数因变量。

    NB2 在 Poisson 基础上增加 α 参数来吸收额外变异：
    Var(Y) = μ + α × μ²。

    α = 0 → 退化为 Poisson（方差 = 均值）
    α > 0 → 方差大于均值（过离散）

    使用交替估计：
    1. Poisson 估计初始 β
    2. moment estimator 估计 α
    3. 固定 α，IRLS 更新 β
    4. 重复 2-3 直至收敛

    注意：如果数据的过离散不严重（dispersion ≈ 1），NB 和 Poisson
    的结果会非常接近。此时直接用 Poisson 就行——少一个参数，更简洁。

    Args:
        y: 因变量（非负计数）。
        X: 自变量矩阵。
        ci_level: 置信水平（默认 0.95）。
        exposure: 暴露量 offset。
        alpha_init: α 的初始值（默认 1.0）。NB IRLS 的权重取决于 α，
            起点不同不影响最终结果（如果收敛），但影响迭代次数。
        max_iter: 最大迭代次数。
        tol: 收敛容差。

    Returns:
        CountRegressionResult (method="negative_binomial")。

    Example:
        # 当 Poisson 显示 dispersion > 1.5 时考虑 NB
        >>> result_nb = negative_binomial_regression(
        ...     y=df["complaint_count"],
        ...     X=df[["staff_count", "avg_salary"]],
        ... )
        >>> result_nb.alpha  # NB 的离散参数
        >>> result_nb.dispersion  # 应该 ≈ 1.0
    """
    # ── 数据清洗 ──
    y_arr = np.asarray(y, dtype=np.float64)
    if isinstance(X, pd.DataFrame):
        var_names = list(X.columns)
        X_arr = X.to_numpy(dtype=np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        var_names = [f"X{i+1}" for i in range(X_arr.shape[1])]

    if exposure is not None:
        exp_arr = np.asarray(exposure, dtype=np.float64)
    else:
        exp_arr = None

    # 按行删除 NaN
    mask = (~np.isnan(y_arr)) & (~np.isnan(X_arr).any(axis=1))
    if exp_arr is not None:
        mask = mask & (~np.isnan(exp_arr))
    y_arr, X_arr = y_arr[mask], X_arr[mask]
    if exp_arr is not None:
        exp_arr = exp_arr[mask]
    n, k = X_arr.shape

    if n == 0:
        raise ValueError("清洗后无有效样本。")
    if k == 0:
        raise ValueError("至少需要一个自变量。")
    if np.any(y_arr < 0):
        raise ValueError("因变量 y 含负值，计数回归要求 y ≥ 0。")

    # 添加截距
    X_design = np.column_stack([np.ones(n, dtype=np.float64), X_arr])
    p = k + 1

    # offset
    offset = np.zeros(n, dtype=np.float64)
    if exp_arr is not None:
        offset = np.log(np.maximum(exp_arr, 1e-10))

    # ── 阶段 1：Poisson 起步 ──
    poisson_result = _count_regression_irls(
        y_arr, X_arr, ci_level=ci_level, exposure=exp_arr,
        max_iter=max_iter, tol=tol, model="poisson",
    )

    # 从 Poisson 结果取初始 beta
    beta = np.zeros(p, dtype=np.float64)
    beta[0] = poisson_result.intercept.b if poisson_result.intercept else 0.0
    for j, coef in enumerate(poisson_result.coefficients):
        beta[j + 1] = coef.b

    # ── 交替估计 NB α 和 β ──
    alpha = max(alpha_init, 1e-6)
    converged = False
    iterations = 0

    for outer in range(max_iter):
        iterations = outer + 1

        # ── 子阶段 A：固定 α，IRLS 更新 β ──
        for inner in range(max_iter):
            eta = X_design @ beta + offset
            mu = np.exp(eta)

            # NB 权重：W = diag(μ / (1 + αμ))
            w = mu / (1.0 + alpha * mu)
            # 工作响应：z = η + (y-μ) / μ
            z = eta + (y_arr - mu) / np.maximum(mu, 1e-10)

            W_sqrt = np.sqrt(np.maximum(w, 1e-15))
            XW = X_design * W_sqrt[:, np.newaxis]
            zW = z * W_sqrt

            try:
                XWX_inv = np.linalg.inv(XW.T @ XW)
            except np.linalg.LinAlgError:
                XWX_inv = np.linalg.pinv(XW.T @ XW)

            beta_new = XWX_inv @ (XW.T @ zW)
            delta = np.max(np.abs(beta_new - beta))
            beta = beta_new

            if delta < tol:
                break

        # ── 子阶段 B：更新 α (moment estimator) ──
        eta = X_design @ beta + offset
        mu = np.exp(eta)
        residual = y_arr - mu
        # Pearson 残差平方 = (y-μ)² / Var，NB 中 Var = μ + αμ²
        # 矩估计量：α = max( Σ[(y-μ)²/μ² - 1/μ] / (n-p), 0 )
        # 更稳健的方法：解 Σ((y-μ)²/(μ + αμ²)) = n-p 对于 α
        # 用简单的 Pearson estimator：
        pearson_components = (residual ** 2 - mu) / np.maximum(mu ** 2, 1e-10)
        pearson_components = pearson_components[mu > 1e-10]
        if len(pearson_components) > p:
            alpha_new = max(float(np.mean(pearson_components)), 1e-10)
        else:
            alpha_new = alpha_init

        alpha_change = abs(alpha_new - alpha) / max(alpha, 1e-10)
        alpha = alpha_new

        if alpha_change < tol:
            converged = True
            break

    # ── 最终诊断 ──
    eta_final = X_design @ beta + offset
    mu_final = np.exp(eta_final)

    se_final = np.sqrt(np.maximum(np.diag(XWX_inv), 0.0))
    z_final = beta / np.where(se_final > 0, se_final, 1.0)
    p_final = 2.0 * (1.0 - sp_stats.norm.cdf(np.abs(z_final)))

    z_crit = sp_stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0)
    irr_final = np.exp(beta)
    irr_low = np.exp(beta - z_crit * se_final)
    irr_high = np.exp(beta + z_crit * se_final)

    # 模型拟合
    ll = _negbin_loglik(y_arr, mu_final, alpha)
    deviance = 2.0 * np.sum(y_arr * np.log(np.maximum(y_arr, 1e-15) / np.maximum(mu_final, 1e-15)) - (y_arr - mu_final))
    pearson_chi2 = float(np.sum((y_arr - mu_final) ** 2 / np.maximum(mu_final + alpha * mu_final ** 2, 1e-15)))
    dispersion = pearson_chi2 / (n - p) if (n - p) > 0 else float("nan")

    # 过离散检验：score test for H0: α = 0
    # 用 Pearson 残差的平方对 fitted values 的回归
    from scipy.special import gammaln
    pearson_res_sq = (y_arr - mu_final) ** 2 / np.maximum(mu_final, 1e-10)
    score_stat = 0.0
    for i in range(n):
        score_stat += (pearson_res_sq[i] - mu_final[i]) / (mu_final[i] * np.sqrt(2.0))
    score_stat = score_stat / np.sqrt(n) if n > 0 else 0.0
    p_over = 2.0 * (1.0 - sp_stats.norm.cdf(abs(score_stat)))
    p_over = max(0.0, min(1.0, float(p_over)))

    aic = -2.0 * ll + 2.0 * p
    bic = -2.0 * ll + p * np.log(n)

    intercept = CountCoefficient(
        name="(Intercept)",
        b=float(beta[0]), se=float(se_final[0]), z=float(z_final[0]),
        p=float(p_final[0]), irr=float(irr_final[0]),
        irr_ci_95=(float(irr_low[0]), float(irr_high[0])),
    )

    coefficients = []
    for j in range(k):
        coefficients.append(CountCoefficient(
            name=var_names[j],
            b=float(beta[j + 1]), se=float(se_final[j + 1]),
            z=float(z_final[j + 1]), p=float(p_final[j + 1]),
            irr=float(irr_final[j + 1]),
            irr_ci_95=(float(irr_low[j + 1]), float(irr_high[j + 1])),
        ))

    return CountRegressionResult(
        n=n, p=p,
        coefficients=coefficients,
        intercept=intercept,
        log_likelihood=ll,
        aic=aic, bic=bic,
        deviance=float(deviance),
        pearson_chi2=pearson_chi2,
        dispersion=float(dispersion),
        p_overdispersion=p_over,
        converged=converged,
        iterations=iterations,
        method="negative_binomial",
        alpha=float(alpha),
        alpha_se=None,
        predicted_counts=mu_final,
    )


# ═══════════════════════════════════════════
# 内部：IRLS 核心（Poisson 共用）
# ═══════════════════════════════════════════


def _count_regression_irls(
    y_arr: np.ndarray,
    X_arr: np.ndarray,
    *,
    ci_level: float,
    exposure: np.ndarray | None,
    max_iter: int,
    tol: float,
    model: str,  # "poisson" or "negative_binomial"
) -> CountRegressionResult:
    """IRLS 核心算法（Poisson 和 NB 共享）。"""
    n, k = X_arr.shape
    p = k + 1

    X_design = np.column_stack([np.ones(n, dtype=np.float64), X_arr])

    if exposure is not None:
        offset = np.log(np.maximum(exposure, 1e-10))
    else:
        offset = np.zeros(n, dtype=np.float64)

    # 初始化：mu 从观测值出发，保持与 offset 的一致性
    # 不能用 (y+mean)/2，因为 offset 差异大时初始 eta=offset 与 log(mu) 不匹配
    mu = np.maximum(y_arr + 0.5, 1e-3)

    # 用 log(mu) - offset 的平均值初始化截距
    beta = np.zeros(p, dtype=np.float64)
    log_mu_init = np.log(mu)
    beta[0] = float(np.mean(log_mu_init - offset))
    # 用此 β 重新计算 mu 以保持一致性
    mu = np.exp(offset + X_design @ beta)

    converged = False
    iterations = 0

    for it in range(1, max_iter + 1):
        iterations = it

        # 当前 eta = Xβ + offset
        eta = X_design @ beta + offset
        # 工作响应变量：z = η + (y - μ) / μ
        z = eta + (y_arr - mu) / np.maximum(mu, 1e-10)

        # 权重矩阵（Poisson: W = μ）
        if model == "poisson":
            w = mu  # Var = μ
        else:
            w = mu

        W_sqrt = np.sqrt(np.maximum(w, 1e-15))
        XW = X_design * W_sqrt[:, np.newaxis]
        # 从工作响应中减去 offset，因为 X 矩阵不包含 offset
        z_adj = (z - offset) * W_sqrt

        try:
            XWX_inv = np.linalg.inv(XW.T @ XW)
        except np.linalg.LinAlgError:
            XWX_inv = np.linalg.pinv(XW.T @ XW)

        beta_new = XWX_inv @ (XW.T @ z_adj)

        # 更新 μ = exp(offset + Xβ)
        eta_new = offset + X_design @ beta_new
        mu_new = np.exp(np.maximum(eta_new, -20))  # exp(-20) ≈ 2e-9，防止下溢

        # 收敛判断
        delta = np.max(np.abs(beta_new - beta))
        beta = beta_new
        mu = mu_new

        if delta < tol:
            converged = True
            break

    # ── 标准误 ──
    eta_final = offset + X_design @ beta
    mu_final = np.exp(np.maximum(eta_final, -20))

    # 信息矩阵 = X'WX（Fisher information）
    w_final = mu_final if model == "poisson" else mu_final
    WX_final = X_design * np.sqrt(np.maximum(w_final, 1e-15))[:, np.newaxis]
    info = WX_final.T @ WX_final
    try:
        info_inv = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        info_inv = np.linalg.pinv(info)
    se_all = np.sqrt(np.maximum(np.diag(info_inv), 0.0))

    # ── Wald z ──
    z_all = beta / np.where(se_all > 0, se_all, 1.0)
    p_all = 2.0 * (1.0 - sp_stats.norm.cdf(np.abs(z_all)))

    # ── IRR ──
    z_crit = sp_stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0)
    irr_all = np.exp(beta)
    irr_low = np.exp(beta - z_crit * se_all)
    irr_high = np.exp(beta + z_crit * se_all)

    # ── 模型拟合 ──
    ll = _poisson_loglik(y_arr, mu_final)
    deviance = 2.0 * np.sum(
        y_arr * np.log(np.maximum(y_arr, 1e-15) / np.maximum(mu_final, 1e-15))
        - (y_arr - mu_final)
    )
    pearson_chi2 = float(np.sum((y_arr - mu_final) ** 2 / np.maximum(mu_final, 1e-10)))
    dispersion = pearson_chi2 / (n - p) if (n - p) > 0 else float("nan")

    # 过离散检验（score test for H0: dispersion = 1）
    # 回归 (y-μ)²/μ 对 μ，t 检验斜率为 0
    pearson_res_sq = (y_arr - mu_final) ** 2 / np.maximum(mu_final, 1e-10)
    mu_centered = mu_final - np.mean(mu_final)
    # 简单相关性检验
    if np.std(mu_centered) > 1e-10 and np.std(pearson_res_sq) > 1e-10:
        r_over, _ = sp_stats.pearsonr(pearson_res_sq, mu_centered)
        # 用 t 检验近似
        t_over = r_over * np.sqrt((n - 2) / max(1 - r_over ** 2, 1e-10))
        p_over = 2.0 * (1.0 - sp_stats.t.cdf(abs(t_over), n - 2))
    else:
        p_over = 0.5

    aic = -2.0 * ll + 2.0 * p
    bic = -2.0 * ll + p * np.log(n)

    # ── 系数组装 ──
    if isinstance(X_arr, np.ndarray):
        var_names = [f"X{i+1}" for i in range(k)]
    else:
        var_names = [str(i) for i in range(k)]

    intercept = CountCoefficient(
        name="(Intercept)",
        b=float(beta[0]), se=float(se_all[0]), z=float(z_all[0]),
        p=float(p_all[0]), irr=float(irr_all[0]),
        irr_ci_95=(float(irr_low[0]), float(irr_high[0])),
    )
    coefficients = []
    for j in range(k):
        coefficients.append(CountCoefficient(
            name=var_names[j],
            b=float(beta[j + 1]), se=float(se_all[j + 1]),
            z=float(z_all[j + 1]), p=float(p_all[j + 1]),
            irr=float(irr_all[j + 1]),
            irr_ci_95=(float(irr_low[j + 1]), float(irr_high[j + 1])),
        ))

    return CountRegressionResult(
        n=n, p=p,
        coefficients=coefficients,
        intercept=intercept,
        log_likelihood=ll,
        aic=aic, bic=bic,
        deviance=float(deviance),
        pearson_chi2=pearson_chi2,
        dispersion=float(dispersion),
        p_overdispersion=float(p_over),
        converged=converged,
        iterations=iterations,
        method="poisson",
        alpha=None,
        alpha_se=None,
        predicted_counts=mu_final,
    )


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def count_regression_report(r: CountRegressionResult) -> str:
    """计数回归报告文本。

    Args:
        r: CountRegressionResult（Poisson 或 NB）。

    Returns:
        格式化报告字符串。
    """
    method_label = {
        "poisson": "Poisson 回归",
        "negative_binomial": "负二项回归 (NB2)",
    }.get(r.method, r.method)

    lines = [
        f"{'='*65}",
        f"  {method_label}",
        f"  n={r.n}, 参数={r.p}, 收敛={'是' if r.converged else '否'} ({r.iterations} 步)",
        f"{'='*65}",
        "",
        f"  {'变量':<20} {'B':>10} {'SE':>10} {'z':>8} {'p':>8} {'IRR':>8} {'95% CI IRR':>18}",
        f"  {'-'*84}",
    ]

    if r.intercept:
        ci = r.intercept
        lines.append(
            f"  {ci.name:<20} {ci.b:>10.4f} {ci.se:>10.4f} {ci.z:>8.3f} {ci.p:>8.4f} "
            f"{ci.irr:>8.3f} {'':>18}"
        )

    for coef in r.coefficients:
        ci_str = f"[{coef.irr_ci_95[0]:.3f}, {coef.irr_ci_95[1]:.3f}]"
        lines.append(
            f"  {coef.name:<20} {coef.b:>10.4f} {coef.se:>10.4f} {coef.z:>8.3f} {coef.p:>8.4f} "
            f"{coef.irr:>8.3f} {ci_str:>18}"
        )

    lines.append("")
    lines.append(f"  {'─'*50}")
    lines.append(f"  模型拟合")
    lines.append(f"  {'─'*50}")
    lines.append(f"  Log-Likelihood:  {r.log_likelihood:.4f}")
    lines.append(f"  Deviance:        {r.deviance:.4f}")
    lines.append(f"  Pearson χ²:     {r.pearson_chi2:.3f}")
    lines.append(f"  Dispersion:      {r.dispersion:.4f}")

    # 过离散解读
    if not math.isnan(r.dispersion):
        disp = r.dispersion
        if disp < 0.8:
            lines.append(f"    → 离散不足 (dispersion < 1)，可能存在欠离散")
        elif disp < 1.2:
            lines.append(f"    → 无显著过离散 (dispersion ≈ 1)，Poisson 假设成立")
        elif disp < 2.0:
            lines.append(f"    → 中等过离散 (dispersion={disp:.2f} > 1)，建议使用 NB")
        else:
            lines.append(f"    → 严重过离散 (dispersion={disp:.2f} >> 1)，Poisson 不可靠")

    if r.p_overdispersion is not None and not math.isnan(r.p_overdispersion):
        lines.append(f"  过离散检验 p:    {r.p_overdispersion:.4f}")
        if r.p_overdispersion < 0.05:
            lines.append(f"    → p<0.05，Poisson 假设被拒绝，建议使用 NB")
        else:
            lines.append(f"    → p≥0.05，未检测到显著过离散")

    lines.append(f"  AIC:             {r.aic:.2f}")
    lines.append(f"  BIC:             {r.bic:.2f}")

    if r.alpha is not None:
        lines.append(f"  NB α (离散参数): {r.alpha:.4f}")

    lines.append(f"{'='*65}")
    return "\n".join(lines)
