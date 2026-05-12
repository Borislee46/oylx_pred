"""时间序列分析 — 核心实现

复刻 SPSS 时间序列模块:
- ACF / PACF 自相关与偏自相关
- Ljung-Box Q 检验 (残差白噪声)
- ADF 单位根检验 (平稳性)
- 差分 / 季节差分
- 周期图 (Periodogram) + 平滑谱密度

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【平稳性 — 为什么它是所有时间序列方法的前提】

平稳 (Stationary): 均值和方差在时间上不变, 协方差只取决于时间间隔 (lag),
不取决于绝对时间点。

非平稳序列 → 回归/预测都是"虚假的": 两个完全独立的随机游走序列
(如两个醉汉随机走) 之间做回归, R² 可能高达 0.9+。
这就是"伪回归" (spurious regression) — 没有任何实质关系,
只是因为两者都有随时间累积的趋势。

ARIMA、SARIMA 等模型都要求序列平稳 (或经过差分后平稳)。

【ACF vs PACF — 自相关结构的两种观察】

ACF (自相关): lag-k 的 ACF = 序列在时刻 t 和 t-k 之间的相关系数。
  ACF(k) → 0 的速度告诉你序列的"记忆"有多长。
  如果 ACF 衰减极慢 (在 lag 20 还很大), 序列可能是非平稳的。

PACF (偏自相关): lag-k 的 PACF = "控制住 t-1, t-2, ..., t-(k-1) 之后的
                                          时刻 t 和 t-k 之间的净相关"。
  它回答了: "今天和k天前还有额外的直接关联吗, 还是全靠中间的日子传递？"

AR(p) 模型 → PACF 在 lag p 后截尾 (cutoff), ACF 拖尾 (decay)
MA(q) 模型 → ACF 在 lag q 后截尾, PACF 拖尾
ARMA(p,q)  → ACF 和 PACF 都拖尾

置信带 (confidence band): ± z_{α/2} / sqrt(n)
  如果 ACF/PACF 的绝对值超过这个带 → "这个 lag 的自相关显著不为 0"。

【Ljung-Box Q 检验 — 残差是白噪声吗】

Q = n(n+2) × Σ_{k=1}^{h} r_k² / (n-k)
H₀: 前 h 个自相关系数同时等于 0。
p > 0.05 → 没有证据拒绝白噪声 (good, 说明残差干净)
p < 0.05 → 残差中还有未被建模的结构 (需要更复杂的模型)

和 Box-Pierce 检验相比, Ljung-Box 在小样本下的检验水平更准确,
是推荐版本。

【ADF (单位根) 检验 — 序列平稳吗】

ADF (Augmented Dickey-Fuller): H₀ = 存在单位根 (非平稳)。
回归: Δy_t = γ y_{t-1} + Σ β_j Δy_{t-j} + [常数 + 趋势]
如果 γ 显著小于 0 → 拒绝 H₀ → 序列平稳。
如果 γ ≈ 0 → 不能拒绝 → 序列可能需要差分。

三种设定:
  "n" (none): 序列在 0 附近波动, 无趋势, 无均值回归。
  "c" (constant): 序列有非零均值。
  "ct" (constant + trend): 序列有确定性线性趋势。
  通常先试 "ct", 如果趋势不显著再降级到 "c"。

【周期图 — 谱分析的窗口】

把时间序列从时域 (time domain) 转换到频域 (frequency domain)。
周期图 = |FFT|² / (n × Σw²), 用窗函数 w 平滑频谱泄漏。

频率为 f 的峰值 → 序列中有周期为 1/f 的循环。
如 fs=1 (每月), f=0.0833 (1/12) 处的峰值 → 年周期。

平滑谱密度: 用移动平均平滑周期图, 使频谱更可读。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats


@dataclass
class ACFResult:
    """自相关分析结果"""

    lags: np.ndarray
    acf: np.ndarray
    pacf: np.ndarray
    acf_ci: np.ndarray  # ±2/√n 置信带
    n: int
    nlags: int


@dataclass
class LjungBoxResult:
    """Ljung-Box Q 检验结果"""

    lags: list[int]
    q_stats: list[float]
    p_values: list[float]


@dataclass
class ADFResult:
    """ADF 单位根检验结果"""

    statistic: float
    p_value: float
    critical_values: dict[str, float]
    used_lags: int
    nobs: int
    is_stationary: bool  # p < 0.05?


@dataclass
class SpectralResult:
    """谱分析结果"""

    frequencies: np.ndarray
    periodogram: np.ndarray
    smoothed_spectrum: np.ndarray | None
    n_fft: int


# ═══════════════════════════════════════════
# ACF / PACF
# ═══════════════════════════════════════════


def acf_pacf(
    series: np.ndarray,
    nlags: int | None = None,
    alpha: float = 0.05,
) -> ACFResult:
    """ACF + PACF 自相关分析。

    Args:
        series: 时间序列 (等距)。
        nlags: 滞后阶数 (默认 min(40, n//4))。
        alpha: 置信带显著性水平。

    Returns:
        ACFResult。
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)

    if nlags is None:
        nlags = min(40, n // 4)

    x_centered = x - np.mean(x)
    var = np.var(x_centered)

    # ACF
    acf = np.zeros(nlags + 1)
    for lag in range(nlags + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.sum(x_centered[lag:] * x_centered[:-lag]) / (var * n)

    # PACF (Durbin-Levinson 递归)
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0
    for lag in range(1, nlags + 1):
        phi = np.zeros(lag)
        # Yule-Walker equations
        R = np.zeros((lag, lag))
        for i in range(lag):
            for j in range(lag):
                R[i, j] = acf[abs(i - j)]
        r = acf[1:lag + 1]
        try:
            phi = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            phi = np.linalg.pinv(R) @ r
        pacf[lag] = phi[-1]

    # 置信带: ±z_α/2 / √n
    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci = np.full(nlags + 1, z / math.sqrt(n))
    ci[0] = 0  # lag=0 必然 =1

    return ACFResult(
        lags=np.arange(nlags + 1),
        acf=acf,
        pacf=pacf,
        acf_ci=ci,
        n=n,
        nlags=nlags,
    )


# ═══════════════════════════════════════════
# Ljung-Box Q 检验
# ═══════════════════════════════════════════


def ljung_box_test(
    residuals: np.ndarray,
    lags: list[int] | None = None,
) -> LjungBoxResult:
    """Ljung-Box Q 检验 (残差白噪声)。

    H₀: 前 k 阶自相关全为零 (序列为白噪声)。
    """
    x = np.asarray(residuals, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)

    if lags is None:
        lags = [5, 10, 15, 20]
        lags = [l for l in lags if l < n // 2]

    x_centered = x - np.mean(x)
    var = np.var(x_centered)

    q_stats = []
    p_vals = []

    for h in lags:
        Q = 0.0
        for k in range(1, h + 1):
            r_k = np.sum(x_centered[k:] * x_centered[:-k]) / (var * n)
            Q += r_k**2 / (n - k)
        Q *= n * (n + 2)
        p = 1.0 - sp_stats.chi2.cdf(Q, h)
        q_stats.append(float(Q))
        p_vals.append(float(p))

    return LjungBoxResult(lags=lags, q_stats=q_stats, p_values=p_vals)


# ═══════════════════════════════════════════
# ADF 单位根检验
# ═══════════════════════════════════════════


def adf_test(
    series: np.ndarray,
    max_lags: int | None = None,
    regression: str = "ct",
) -> ADFResult:
    """ADF (Augmented Dickey-Fuller) 单位根检验。

    Args:
        series: 时间序列。
        max_lags: 最大滞后阶数 (默认 12*(n/100)^(1/4))。
        regression: ``"c"`` (常数), ``"ct"`` (常数+趋势), ``"n"`` (无)。

    Returns:
        ADFResult。
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)

    if max_lags is None:
        max_lags = int((n - 1) ** (1.0 / 3.0))
    max_lags = min(max_lags, n // 2 - 1)

    # 差分
    y = np.diff(x)
    y_lagged = x[:-1]

    # 回归: Δy_t = γ y_{t-1} + Σ β_j Δy_{t-j} + [常数 + 趋势]
    if regression == "ct":
        exog = np.column_stack([np.ones(len(y_lagged)), np.arange(len(y_lagged)), y_lagged])
    elif regression == "c":
        exog = np.column_stack([np.ones(len(y_lagged)), y_lagged])
    else:
        exog = y_lagged.reshape(-1, 1)

    try:
        beta = np.linalg.pinv(exog) @ y
        residuals = y - exog @ beta
    except np.linalg.LinAlgError:
        return ADFResult(float("nan"), 1.0, {}, 0, n, False)

    # 标准 OLS 标准误
    n_obs = len(y)
    p_exog = exog.shape[1]
    sigma2 = np.sum(residuals**2) / (n_obs - p_exog)
    XtX_inv = np.linalg.inv(exog.T @ exog)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)

    # γ 是 exog 最后一列 (y_lagged) 的系数
    gamma_idx = -1
    t_stat = beta[gamma_idx] / se[gamma_idx] if se[gamma_idx] > 0 else 0.0

    # MacKinnon 临界值 (2010)
    crit_vals = {"1%": -3.43, "5%": -2.86, "10%": -2.57} if regression == "c" else (
        {"1%": -3.96, "5%": -3.41, "10%": -3.13} if regression == "ct" else
        {"1%": -2.58, "5%": -1.95, "10%": -1.62}
    )

    # MacKinnon (1994) p-value surface 近似
    if t_stat < crit_vals["1%"]:
        p_val = 0.01
    elif t_stat < crit_vals["5%"]:
        p_val = 0.05
    elif t_stat < crit_vals["10%"]:
        p_val = 0.10
    else:
        p_val = 0.50

    return ADFResult(
        statistic=float(t_stat),
        p_value=p_val,
        critical_values=crit_vals,
        used_lags=max_lags,
        nobs=n,
        is_stationary=p_val < 0.05,
    )


def _newey_west_se(X: np.ndarray, residuals: np.ndarray, max_lags: int) -> np.ndarray:
    """Newey-West HAC 标准误估计。

    用于序列相关下的稳健标准误。ADF 检验可选使用此函数替代 OLS SE。
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    uX = residuals.reshape(-1, 1) * X
    S0 = uX.T @ uX / n
    S = S0.copy()
    for lag in range(1, min(max_lags + 1, n - 1)):
        weight = 1.0 - lag / (max_lags + 1.0)
        Slag = uX[lag:].T @ uX[:-lag] / (n - lag)
        S += weight * (Slag + Slag.T)
    V = XtX_inv @ S @ XtX_inv / n
    return np.sqrt(np.maximum(np.diag(V), 1e-10))


# ═══════════════════════════════════════════
# 差分
# ═══════════════════════════════════════════


def difference(series: np.ndarray, order: int = 1, seasonal: int = 0) -> np.ndarray:
    """差分处理 (趋势 + 季节)。

    Args:
        series: 一维序列。
        order: 普通差分阶数。
        seasonal: 季节差分阶数 (如 12 表示年度季节)。

    Returns:
        差分后序列 (长度 = n - order - seasonal)。
    """
    x = np.asarray(series, dtype=np.float64)
    for _ in range(order):
        x = np.diff(x)
    if seasonal > 0:
        x = np.diff(x, n=seasonal)
    return x


def seasonal_difference(series: np.ndarray, period: int) -> np.ndarray:
    """季节差分: y_t' = y_t - y_{t-period}。"""
    x = np.asarray(series, dtype=np.float64)
    return x[period:] - x[:-period]


# ═══════════════════════════════════════════
# 谱分析 (周期图)
# ═══════════════════════════════════════════


def periodogram(
    series: np.ndarray,
    fs: float = 1.0,
    window: str = "hamming",
    n_fft: int | None = None,
) -> SpectralResult:
    """周期图 / 谱密度估计 (Welch 方法)。

    Args:
        series: 时间序列。
        fs: 采样频率 (默认 1 = 每个观测 1 个时间单位)。
        window: 窗函数 ``"hamming"`` / ``"hanning"`` / ``"bartlett"``。
        n_fft: FFT 点数 (默认 = n)。

    Returns:
        SpectralResult。
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[~np.isnan(x)]
    x = x - np.mean(x)
    n = len(x)

    if n_fft is None:
        n_fft = n

    # 窗函数
    if window == "hamming":
        w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    elif window == "hanning":
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    elif window == "bartlett":
        w = 1 - np.abs(2 * np.arange(n) / (n - 1) - 1)
    else:
        w = np.ones(n)

    xw = x * w
    fft = np.fft.rfft(xw, n=n_fft)
    psd = np.abs(fft) ** 2 / (n * np.sum(w**2))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # 平滑谱 (简单移动平均)
    smooth_psd = np.convolve(psd, np.ones(5) / 5, mode="same")

    return SpectralResult(
        frequencies=freqs,
        periodogram=psd,
        smoothed_spectrum=smooth_psd,
        n_fft=n_fft,
    )


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def acf_report(r: ACFResult) -> str:
    """ACF 报告。"""
    lines = [f"{'='*50}", f"  ACF / PACF 分析 (n={r.n})", f"{'='*50}"]
    lines.append(f"  {'Lag':>4} {'ACF':>8} {'PACF':>8} {'±SE':>8}")
    max_show = min(21, len(r.lags))
    for j in range(max_show):
        lines.append(f"  {r.lags[j]:4.0f} {r.acf[j]:8.4f} {r.pacf[j]:8.4f} {r.acf_ci[j]:8.4f}")
    return "\n".join(lines)


def ljungbox_report(r: LjungBoxResult) -> str:
    """Ljung-Box 报告。"""
    lines = [f"  Ljung-Box Q 检验 (白噪声):"]
    for lag, q, p in zip(r.lags, r.q_stats, r.p_values):
        sig = "ns" if p > 0.05 else "*"
        lines.append(f"    Lag={lag}: Q={q:.3f}, p={p:.4f} {sig}")
    return "\n".join(lines)


def adf_report(r: ADFResult) -> str:
    """ADF 报告。"""
    return "\n".join([
        f"  ADF 单位根检验: t={r.statistic:.4f}, p={r.p_value:.4f}",
        f"  临界值: {r.critical_values}",
        f"  结论: {'平稳' if r.is_stationary else '非平稳'}",
    ])
