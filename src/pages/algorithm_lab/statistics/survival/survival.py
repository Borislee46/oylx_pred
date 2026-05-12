"""生存分析 (Survival Analysis) — 核心实现

复刻 SPSS Survival 模块:
- Kaplan-Meier 生存函数估计 + Greenwood SE + 中位生存期
- Log-Rank / Breslow / Tarone-Ware 组间比较
- Cox 比例风险回归 (Breslow/Efron ties)
- Schoenfeld 残差与比例风险假设检验 (Grambsch-Therneau)
- 生存表 (Life Table)

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【生存分析要回答什么问题】

普通的回归/ANOVA 不能处理"删失" (censoring):
  有些人没发生事件就退出了研究 (失访), 或研究结束时还没发生事件。
  你不能简单地排除他们或把他们当"成功" — 两种做法都会产生偏倚。

生存分析同时利用了"事件时间"和"删失与否"的信息。

【Kaplan-Meier (KM) 估计 — 用乘法算生存】

S(t) = ∏_{t_j ≤ t} (1 - d_j / n_j)
d_j: 在时刻 t_j 发生事件的人数
n_j: 在时刻 t_j 之前仍处于风险集中的人数

直觉: 每个事件时刻的生存概率 = 1 - 该时刻的死亡率,
总体生存率 = 各时刻生存率的累乘。
如果中途有删失, 对 n_j 有贡献但不算入事件。

Greenwood 公式: 生存率的标准误。将各事件时刻的条件方差通过
Δ 方法 (delta method) 积累起来。

中位生存期: 当 S(t) 降至 0.5 时的时间。如果曲线从未低于 0.5,
中位生存期不可估计 (通常在 >50% 的人仍然"活着"时报告)。

log-log CI: 对 log(-log(S(t))) 构造对称 CI 再变换回来,
这样保证 CI 始终在 [0,1] 范围内。比直接对 S(t) 构造 Wald CI 好。

【Log-Rank 检验 vs Breslow vs Tarone-Ware】

都检验多组生存曲线是否相同。区别在于对"时间"的加权:

Log-Rank: 所有事件时间权重相等。
  → 对后期差异和早期差异同样敏感。
  → 最常用, 在"比例风险"假设下统计效力最高。

Breslow (Gehan): 权重 = n_risk (风险集大小)。
  → 早期事件权重更大 (因为早期风险集更大)。
  → 对早期差异更敏感。

Tarone-Ware: 权重 = sqrt(n_risk)。
  → 介于 Log-Rank 和 Breslow 之间。

如果你的两条生存曲线在早期交叉但在后期分离 → Log-Rank 可能不敏感,
考虑 Breslow。

【Cox 比例风险回归 — 多变量生存分析】

h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)

核心假设: 比例风险 (PH 假设) — 不同个体的风险比是随时间不变的。
  HR = exp(β) 是常数, 不因为 t 的变化而变化。

β > 0 → HR > 1 → 风险增加 (活得更短)
β < 0 → HR < 1 → 风险降低 (保护因子)
β = 0 → HR = 1 → 无影响

h₀(t) (基线风险函数) 不被估计 — Cox 回归是"半参数"的:
  不指定基线风险的具体形状, 只估计协变量的效应。
  这是 Cox 回归最大的优势: 不需要假定生存时间的分布。

Newton-Raphson 迭代: 梯度 = 观察值 - 期望值 (基于风险集),
  Hessian = 风险集内的协变量协方差矩阵。

Breslow vs Efron 处理结 (ties):
  多个事件发生在完全相同的时间。
  Breslow: 简单近似, 将同时间的风险集不同区分对待。
  Efron: 更精确, 对结内不同个体分配不同的"分数"权。
  现代推荐 Efron。
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
class KMSurvival:
    """Kaplan-Meier 生存估计"""

    time: np.ndarray
    n_risk: np.ndarray
    n_events: np.ndarray
    n_censored: np.ndarray
    survival: np.ndarray
    std_error: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    # 摘要
    median_survival: float = float("nan")
    median_ci_lower: float = float("nan")
    median_ci_upper: float = float("nan")


@dataclass
class LogRankResult:
    """组间秩检验结果"""

    method: str  # "log-rank" | "breslow" | "tarone-ware"
    chi2: float
    df: int
    p_value: float


@dataclass
class CoxPHResult:
    """Cox 比例风险回归结果"""

    n: int
    n_events: int
    log_likelihood: float
    # 系数
    coefficients: list[dict]
    # 全局检验
    wald_chi2: float
    wald_df: int
    wald_p: float
    likelihood_ratio_chi2: float
    likelihood_ratio_p: float


# ═══════════════════════════════════════════
# Kaplan-Meier
# ═══════════════════════════════════════════


def kaplan_meier(
    time: np.ndarray,
    event: np.ndarray,
    *,
    ci_level: float = 0.95,
) -> KMSurvival:
    """Kaplan-Meier 生存函数估计。

    Args:
        time: 随访时间。
        event: 事件指示变量 (1=事件, 0=删失)。

    Returns:
        KMSurvival。
    """
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    mask = ~np.isnan(t) & ~np.isnan(e.astype(float))
    t, e = t[mask], e[mask]

    # 按时间排序
    order = np.argsort(t)
    t = t[order]
    e = e[order]

    # 识别唯一事件时间
    unique_times = []
    n_risk_list = []
    n_events_list = []
    n_censored_list = []

    i = 0
    n = len(t)
    while i < n:
        current_t = t[i]
        j = i
        events = 0
        censored = 0
        while j < n and t[j] == current_t:
            if e[j]:
                events += 1
            else:
                censored += 1
            j += 1

        if events > 0:
            unique_times.append(current_t)
            n_risk_list.append(n - i)
            n_events_list.append(events)
            n_censored_list.append(censored)

        i = j

    m = len(unique_times)
    n_risk = np.array(n_risk_list)
    n_events = np.array(n_events_list)

    # KM 估计: S(t) = ∏ (1 - d_j / n_j)
    survival = np.ones(m)
    prod = 1.0
    for j in range(m):
        prod *= (1 - n_events[j] / n_risk[j]) if n_risk[j] > 0 else 1.0
        survival[j] = prod

    # Greenwood SE
    se = np.zeros(m)
    for j in range(m):
        var_sum = 0.0
        for k in range(j + 1):
            denom = n_risk[k] * (n_risk[k] - n_events[k])
            if denom > 0 and n_events[k] > 0:
                var_sum += n_events[k] / denom
        se[j] = survival[j] * math.sqrt(var_sum)

    # log-log CI
    z = sp_stats.norm.ppf(1 - (1 - ci_level) / 2)
    ci_lower = np.zeros(m)
    ci_upper = np.zeros(m)
    for j in range(m):
        if survival[j] > 0 and se[j] > 0:
            theta = math.exp(z * se[j] / (survival[j] * math.log(survival[j])))
            ci_lower[j] = survival[j] ** theta
            ci_upper[j] = survival[j] ** (1.0 / theta)
        else:
            ci_lower[j] = survival[j]
            ci_upper[j] = survival[j]

    # 中位生存期
    median_surv = float("nan")
    median_ci_lo = float("nan")
    median_ci_upper = float("nan")
    if len(survival) > 0:
        idx = np.where(survival <= 0.5)[0]
        if len(idx) > 0:
            median_surv = unique_times[idx[0]]
            # 使用 CI 曲线插值
            if len(ci_lower) > idx[0]:
                median_ci_lo = unique_times[np.where(ci_lower <= 0.5)[0][0]] if np.any(ci_lower <= 0.5) else float("nan")
            if len(ci_upper) > idx[0]:
                median_ci_upper = unique_times[np.where(ci_upper <= 0.5)[0][0]] if np.any(ci_upper <= 0.5) else float("nan")

    return KMSurvival(
        time=np.array(unique_times),
        n_risk=n_risk,
        n_events=n_events,
        n_censored=np.array(n_censored_list),
        survival=survival,
        std_error=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        median_survival=median_surv,
        median_ci_lower=median_ci_lo,
        median_ci_upper=median_ci_upper,
    )


# ═══════════════════════════════════════════
# Log-Rank 检验
# ═══════════════════════════════════════════


def log_rank_test(
    time: np.ndarray,
    event: np.ndarray,
    group: np.ndarray,
    *,
    method: str = "log-rank",
) -> LogRankResult:
    """组间生存曲线比较 (Log-Rank + Breslow + Tarone-Ware)。

    Args:
        time: 随访时间。
        event: 事件指示变量。
        group: 分组变量 (任意组数)。
        method: ``"log-rank"`` (默认) / ``"breslow"`` (对早期差异敏感) / ``"tarone-ware"``。

    Returns:
        LogRankResult。
    """
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    g = np.asarray(group)

    mask = ~np.isnan(t) & ~np.isnan(e.astype(float))
    t, e, g = t[mask], e[mask], g[mask]

    groups = sorted(set(g))
    k = len(groups)
    if k < 2:
        return LogRankResult(method=method, chi2=float("nan"), df=0, p_value=1.0)

    # 唯一事件时间
    sorted_idx = np.argsort(t)
    t, e, g = t[sorted_idx], e[sorted_idx], g[sorted_idx]

    # 在每个事件时间计算 O-E
    unique_event_times = np.unique(t[e])
    n_event_times = len(unique_event_times)

    O = np.zeros(k)
    E = np.zeros(k)
    V = np.zeros((k, k))

    # 累积风险集
    for et in unique_event_times:
        at_risk = t >= et
        n_at_risk = np.sum(at_risk)
        n_events = np.sum(e[t == et])

        if n_at_risk == 0:
            continue
        weight = 1.0

        if method == "breslow":
            weight = n_at_risk
        elif method == "tarone-ware":
            weight = math.sqrt(n_at_risk)

        for idx, grp in enumerate(groups):
            in_grp = g == grp
            og = np.sum(e[at_risk & in_grp & (t == et)])

            # 期望
            n_grp_at_risk = np.sum(at_risk & in_grp)
            eg = n_grp_at_risk * n_events / n_at_risk if n_at_risk > 0 else 0

            O[idx] += weight * og
            E[idx] += weight * eg

    # (O - E)' V⁻¹ (O - E)  →  Rank-based comparison
    O_E = O - E
    if k == 2:
        # 两样本: chi2 = (O-E)² / Var
        var_est = 0.0
        for et in unique_event_times:
            at_risk = t >= et
            n_at = np.sum(at_risk)
            n_ev = np.sum(e[t == et])
            if n_at <= 1:
                continue
            n1 = np.sum(at_risk & (g == groups[0]))
            n2 = np.sum(at_risk & (g == groups[1]))
            var_est += n1 * n2 * n_ev * (n_at - n_ev) / (n_at**2 * (n_at - 1))
        chi2 = O_E[0] ** 2 / var_est if var_est > 0 else 0.0
    else:
        chi2 = np.sum(O_E**2 / np.maximum(E, 1e-10))

    p = 1.0 - sp_stats.chi2.cdf(chi2, k - 1)

    return LogRankResult(
        method=method,
        chi2=float(chi2),
        df=k - 1,
        p_value=float(p),
    )


# ═══════════════════════════════════════════
# Cox 比例风险回归
# ═══════════════════════════════════════════


def cox_ph(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    *,
    ties: str = "breslow",
    ci_level: float = 0.95,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> CoxPHResult:
    """Cox 比例风险回归 (Newton-Raphson)。

    Args:
        time: 随访时间。
        event: 事件指示变量。
        X: 协变量矩阵 (n, k)。
        ties: ``"efron"`` (推荐) / ``"breslow"``。
    ci_level: HR 置信区间的置信水平（默认 0.95）。
        max_iter: 最大迭代次数。
        tol: 收敛容差。

    Returns:
        CoxPHResult。
    """
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    X_mat = np.asarray(X, dtype=np.float64)

    mask = ~np.isnan(t) & ~np.isnan(e.astype(float)) & (~np.isnan(X_mat).any(axis=1))
    t, e, X_mat = t[mask], e[mask], X_mat[mask]
    n, k = X_mat.shape

    # 按时间降序排序 (方便风险集计算)
    order = np.argsort(-t)
    t = t[order]
    e = e[order]
    X_mat = X_mat[order]

    n_events = int(np.sum(e))

    # 初始化 β = 0
    beta = np.zeros(k)

    for iteration in range(max_iter):
        grad = np.zeros(k)
        hess = np.zeros((k, k))

        i = 0
        while i < n:
            if not e[i]:
                i += 1
                continue

            # 收集同一时间点的所有事件
            tied_indices = [i]
            j = i + 1
            while j < n and t[j] == t[i] and e[j]:
                tied_indices.append(j)
                j += 1
            d = len(tied_indices)

            # 风险集: 从第一个 tied 事件起 (降序时间)
            risk_set = np.arange(i, n)
            X_risk = X_mat[risk_set]
            score = X_risk @ beta
            exp_score = np.exp(score - np.max(score))
            S = float(np.sum(exp_score))

            # 预计算风险集加权和 (避免 O(k²) 逐事件循环)
            XW = X_risk * exp_score[:, np.newaxis]
            Xw_vec = XW.sum(axis=0)  # shape (k,): Σ_j X_j exp(score)
            XWX = X_risk.T @ XW  # (k, k): Σ X_j X_l exp(score)

            if ties == "efron" and d > 1:
                # Efron: 分母 S_r = S - (r/d) * S_tied
                S_tied = float(np.sum(exp_score[:d]))
                Xw_tied = np.sum(X_mat[tied_indices] * exp_score[:d, np.newaxis], axis=0)
                # 分子: Σ_{i∈D} X_i（加一次，不在 r 循环内累加）
                for tidx in tied_indices:
                    grad += X_mat[tidx]
                # 分母: 对每个 r，累加 -grad[log(S_r)]
                for r in range(d):
                    S_r = S - (r / d) * S_tied
                    if S_r <= 0:
                        continue
                    Xw_r = Xw_vec - (r / d) * Xw_tied
                    w_mean = Xw_r / S_r
                    grad -= w_mean
                    hess_c = np.outer(w_mean, w_mean) - XWX / S_r
                    hess += hess_c
            else:
                # Breslow: 所有 tied 事件共用一个分母
                w_mean = Xw_vec / S
                hess_c = np.outer(w_mean, w_mean) - XWX / S
                for tidx in tied_indices:
                    grad += X_mat[tidx] - w_mean
                    hess += hess_c

            i = tied_indices[-1] + 1

        # Newton-Raphson 更新
        try:
            delta = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(-hess) @ grad

        beta_new = beta + delta

        if np.max(np.abs(delta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # 标准误: sqrt(-Hess⁻¹ 对角线)
    try:
        cov_matrix = np.linalg.inv(-hess)
        se = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        se = np.full(k, float("nan"))

    z_vals = beta / se
    p_vals = 2.0 * (1.0 - sp_stats.norm.cdf(np.abs(z_vals)))
    hr = np.exp(beta)

    # Wald 全局检验
    wald_chi2 = np.sum(z_vals**2)
    wald_p = 1.0 - sp_stats.chi2.cdf(wald_chi2, k)

    # 对数似然 (使用与梯度循环一致的 ties 处理)
    log_lik = 0.0
    log_lik_null = 0.0
    i = 0
    while i < n:
        if not e[i]:
            i += 1
            continue
        tied_indices = [i]
        j = i + 1
        while j < n and t[j] == t[i] and e[j]:
            tied_indices.append(j)
            j += 1
        d = len(tied_indices)
        n_risk = n - i

        # 模型似然
        risk_set = np.arange(i, n)
        score = X_mat[risk_set] @ beta
        exp_score = np.exp(score - np.max(score))
        S = float(np.sum(exp_score))
        sum_x_beta = float(np.sum(X_mat[tidx] @ beta for tidx in tied_indices))

        if ties == "efron" and d > 1:
            S_tied = float(np.sum(exp_score[:d]))
            # 分子: Σ_{i∈D} X_i β（加一次）
            log_lik += sum_x_beta
            # 分母: Σ_{r} -log(S_r)
            for r in range(d):
                S_r = S - (r / d) * S_tied
                log_lik -= math.log(max(S_r, 1e-15))
            # 空模型似然 (β=0: exp(0)=1, S=n_risk, S_tied=d)
            for r in range(d):
                log_lik_null -= math.log(n_risk - r)
        else:
            log_lik += sum_x_beta - d * math.log(max(S, 1e-15))
            log_lik_null -= d * math.log(n_risk)

        i = tied_indices[-1] + 1

    # 似然比检验
    lr_chi2 = 2.0 * (log_lik - log_lik_null)
    lr_p = 1.0 - sp_stats.chi2.cdf(max(lr_chi2, 0), k)

    z_crit = sp_stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0)

    coefficients = []
    for j in range(k):
        ci_lo = beta[j] - z_crit * se[j]
        ci_hi = beta[j] + z_crit * se[j]
        coefficients.append(
            {
                "name": f"X{j+1}",
                "beta": float(beta[j]),
                "se": float(se[j]),
                "z": float(z_vals[j]),
                "p": float(p_vals[j]),
                "hr": float(hr[j]),
                "hr_ci_95": (float(np.exp(ci_lo)), float(np.exp(ci_hi))),
            }
        )

    return CoxPHResult(
        n=n,
        n_events=n_events,
        log_likelihood=float(log_lik),
        coefficients=coefficients,
        wald_chi2=float(wald_chi2),
        wald_df=k,
        wald_p=float(wald_p),
        likelihood_ratio_chi2=float(lr_chi2),
        likelihood_ratio_p=float(lr_p),
    )


# ═══════════════════════════════════════════
# Schoenfeld 残差 + 比例风险假设检验
# ═══════════════════════════════════════════


@dataclass
class SchoenfeldResult:
    """Schoenfeld 残差与 PH 假设检验 (Grambsch-Therneau)"""

    variables: list[str]
    rho: list[float]  # 残差与 g(time) 的相关系数
    chi2: list[float]  # 单个变量 χ²
    p_values: list[float]  # 单个变量 p
    global_chi2: float
    global_df: int
    global_p: float
    residuals: np.ndarray  # (n_events, k) 缩放 Schoenfeld 残差


def schoenfeld_test(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    *,
    transform: str = "km",
) -> SchoenfeldResult:
    """Schoenfeld 残差检验 — 检验 Cox PH 假设。

    对每个协变量，检验其 Schoenfeld 残差是否与时间相关。
    若 p < 0.05，表明该变量的效应随时间变化，违反了比例风险假设。

    Args:
        time: 随访时间。
        event: 事件指示变量 (1=事件, 0=删失)。
        X: 协变量矩阵 (n, k)。
        beta: 已拟合的 Cox 回归系数。
        transform: 时间变换 ``"km"`` (KM 生存估计, 默认) /
                   ``"rank"`` (秩) / ``"identity"`` (原始时间)。

    Returns:
        SchoenfeldResult。
    """
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    X_mat = np.asarray(X, dtype=np.float64)

    mask = ~np.isnan(t) & ~np.isnan(e.astype(float)) & (~np.isnan(X_mat).any(axis=1))
    t, e, X_mat = t[mask], e[mask], X_mat[mask]
    n, k = X_mat.shape

    # 按时间降序
    order = np.argsort(-t)
    t = t[order]
    e = e[order]
    X_mat = X_mat[order]

    event_idx = np.where(e)[0]
    n_events = len(event_idx)

    # ── 逐事件计算原始 Schoenfeld 残差 ──
    raw_residuals = np.zeros((n_events, k))
    event_times = np.zeros(n_events)

    for idx, ei in enumerate(event_idx):
        event_times[idx] = t[ei]
        risk_set = np.arange(ei, n)
        X_risk = X_mat[risk_set]
        score = X_risk @ beta
        exp_score = np.exp(score - np.max(score))
        S = float(np.sum(exp_score))
        Xw = np.sum(X_risk * exp_score[:, np.newaxis], axis=0)
        w_mean = Xw / S
        raw_residuals[idx] = X_mat[ei] - w_mean

    # ── 缩放残差: r* = n_events × V × r_raw ──
    # V 是协方差矩阵的近似
    V = np.cov(raw_residuals, rowvar=False)
    scaled_residuals = n_events * (raw_residuals @ V)

    # ── 时间变换 g(t) ──
    if transform == "km":
        # KM 生存估计作为时间变换
        km = kaplan_meier(t, e)
        # 在事件时间点插值
        g_t = np.zeros(n_events)
        km_times = km.time
        km_surv = km.survival
        for idx, et in enumerate(event_times):
            # 找到对应时间的生存率
            match = np.where(np.abs(km_times - et) < 1e-10)[0]
            if len(match) > 0:
                g_t[idx] = km_surv[match[0]]
            else:
                g_t[idx] = 1.0
        g_mean = np.mean(g_t)
        g_centered = g_t - g_mean
    elif transform == "rank":
        ranks = sp_stats.rankdata(event_times)
        g_centered = ranks - np.mean(ranks)
    else:  # identity
        g_centered = event_times - np.mean(event_times)

    g_var = np.sum(g_centered**2)
    if g_var <= 0:
        g_var = 1.0

    # ── 逐变量检验: 残差 ~ g(time) 回归 ──
    rho_list = []
    chi2_list = []
    p_list = []

    for j in range(k):
        r_j = scaled_residuals[:, j]
        # 相关系数
        rho = float(np.corrcoef(r_j, g_centered)[0, 1]) if n_events > 2 else 0.0
        rho_list.append(rho)
        # χ² = n_events × rho² (Grambsch-Therneau 1994)
        chi2_j = n_events * rho**2
        chi2_list.append(float(chi2_j))
        p_list.append(float(1.0 - sp_stats.chi2.cdf(chi2_j, 1)))

    # ── 全局检验 ──
    # 所有变量的残差联合检验
    global_chi2 = float(np.sum(chi2_list))
    global_p = float(1.0 - sp_stats.chi2.cdf(global_chi2, k))

    return SchoenfeldResult(
        variables=[f"X{j+1}" for j in range(k)],
        rho=rho_list,
        chi2=chi2_list,
        p_values=p_list,
        global_chi2=global_chi2,
        global_df=k,
        global_p=global_p,
        residuals=scaled_residuals,
    )


def schoenfeld_report(r: SchoenfeldResult) -> str:
    """Schoenfeld PH 检验报告。"""
    lines = [
        f"{'='*55}",
        f"  Schoenfeld 比例风险假设检验",
        f"{'='*55}",
        f"  {'变量':<10} {'rho':>8} {'χ²':>8} {'p':>8}",
    ]
    for i in range(len(r.variables)):
        sig = "*" if r.p_values[i] < 0.05 else " "
        lines.append(
            f"  {r.variables[i]:<10} {r.rho[i]:8.3f} {r.chi2[i]:8.3f} {r.p_values[i]:8.4f} {sig}"
        )
    lines.extend([
        f"  {'─'*40}",
        f"  全局检验: χ²({r.global_df}) = {r.global_chi2:.3f}, p = {r.global_p:.4f}",
    ])
    if r.global_p < 0.05:
        lines.append("  ⚠ p < 0.05: 至少一个变量违反 PH 假设，考虑分层或时变系数。")
    else:
        lines.append("  ✓ p ≥ 0.05: 未检测到 PH 假设违反。")
    lines.append(f"{'='*55}")
    return "\n".join(lines)


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def km_report(km: KMSurvival) -> str:
    """KM 生存估计报告。"""
    lines = [
        f"{'='*50}",
        f"  Kaplan-Meier 生存估计",
        f"  事件时间点数: {len(km.time)}",
    ]
    if not math.isnan(km.median_survival):
        lines.append(
            f"  中位生存期: {km.median_survival:.2f}  "
            f"[95% CI: {km.median_ci_lower:.2f}, {km.median_ci_upper:.2f}]"
        )
    lines.extend([
        "",
        f"  {'时间':>8} {'风险数':>6} {'事件':>6} {'删失':>6} {'生存率':>8} {'SE':>8}",
    ])
    for j in range(min(len(km.time), 20)):  # 前 20 行
        lines.append(
            f"  {km.time[j]:8.2f} {km.n_risk[j]:6d} {km.n_events[j]:6d} "
            f"{km.n_censored[j]:6d} {km.survival[j]:8.4f} {km.std_error[j]:8.4f}"
        )
    return "\n".join(lines)


def cox_report(r: CoxPHResult) -> str:
    """Cox PH 报告。"""
    lines = [
        f"{'='*50}",
        f"  Cox 比例风险回归",
        f"  n = {r.n}, 事件数 = {r.n_events}",
        f"  Wald χ²({r.wald_df}) = {r.wald_chi2:.3f}, p = {r.wald_p:.4f}",
        f"{'='*50}",
        f"  {'变量':<8} {'β':>8} {'SE':>8} {'Z':>8} {'p':>8} {'HR':>8} {'95% CI':>15}",
    ]
    for c in r.coefficients:
        lines.append(
            f"  {c['name']:<8} {c['beta']:8.4f} {c['se']:8.4f} {c['z']:8.3f} "
            f"{c['p']:8.4f} {c['hr']:8.4f} [{c['hr_ci_95'][0]:.4f}, {c['hr_ci_95'][1]:.4f}]"
        )
    return "\n".join(lines)
