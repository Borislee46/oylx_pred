"""Bootstrap 自助法 — 全参数实现

复刻 SPSS Bootstrap 模块 + scipy.stats.bootstrap:
- 百分位 CI (Percentile)
- BCa CI (偏差校正与加速)
- 分层 Bootstrap
- SE 估计
- 点估计策略 (建议使用原始样本统计量, 重抽样分布均值供参考)

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【Bootstrap 的核心直觉】

传统推断: 假定数据来自某个分布 → 推导统计量的理论分布 → 算 CI/p 值。
         问题: 很多统计量没有简单分布公式 (如中位数、相关系数)。

Bootstrap: 把样本当作"迷你总体", 反复从样本中有放回地重抽样。
          每次重抽样算一次统计量 → 得到统计量的经验分布 → 从经验分布
          直接取分位数作为 CI。不需要任何分布假设。

这就是 Efron (1979) 的创见: 用计算代替数学推导。

【百分位 CI vs BCa CI】

百分位 CI (Percentile):
  直接把 Bootstrap 分布的 (α/2) 和 (1-α/2) 分位数作为 CI 边界。
  简单直观, 但有两个问题:
  1. 如果 Bootstrap 分布不是以原估计为中心对称的, 可能偏移。
  2. 如果原估计有偏, Bootstrap 分位数"继承"了这个偏。

BCa CI (Bias-Corrected and Accelerated):
  解决了百分位 CI 的两个问题:
  z₀: 偏差校正因子 — Bootstrap 分布中 < 原始估计的比例。
      如果只有 40% 的 Bootstrap 统计量小于原始估计 → z₀ < 0,
      CI 向更大值的方向调整。
  a (加速因子): Jackknife 估计的曲率参数。
      衡量统计量对每个单独数据点的敏感度。
      如果 a=0 (统计量方差稳定), BCa = 百分位。

  BCa 需要 n_resamples ≥ 1000 (越大越好, 2000 推荐)。

【Jackknife 加速因子的含义】

从原始数据中每次删除一个观测点 (leave-one-out), 算 n 次统计量。
a = (Jackknife 统计量的偏度) / 6。
本质: 衡量统计量估计值在样本空间中的变化率。

直观理解: a 很大的统计量 (如方差) 表示"少一个点会极大改变估计"。
这种统计量的 CI 需要用 a 来修正。

【分层 Bootstrap】
当数据有自然分层 (如来自不同学校、地区), 且各层样本量差异大时使用。
操作: 在每层内部分别有放回地重抽样, 保持每层样本量与原来相同。
保证: 不会出现某个 Bootstrap 样本中某层完全缺失的情况。

【Bootstrap 的局限】
  - 不能创造信息: 如果样本本身有偏 (不是总体的好代表), Bootstrap 无法纠正。
  - 小样本 (n<20): Bootstrap 效果差, 因为重抽样分布非常离散。
  - 中位数: Bootstrap 不光滑。因为中位数对重抽样的微小变化是跳跃式的。
    建议: n>500 时用 Bootstrap 中位数, 否则优先其他方法。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════


@dataclass
class BootstrapResult:
    """Bootstrap 结果"""

    original_stat: float
    bootstrap_mean: float
    bootstrap_se: float
    bootstrap_bias: float
    ci_percentile: tuple[float, float]
    ci_bca: tuple[float, float] | None
    n_resamples: int
    method: str  # "simple" | "stratified"
    ci_level: float


# ═══════════════════════════════════════════
# 核心 Bootstrap
# ═══════════════════════════════════════════


def bootstrap(
    data: np.ndarray,
    statistic_fn,
    *,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    method: str = "percentile",
    random_seed: int | None = 42,
) -> BootstrapResult:
    """简单 Bootstrap (有放回重抽样)。

    Args:
        data: 一维数组。
        statistic_fn: 统计量函数 callable(data) -> float。
        n_resamples: 重抽样次数 (建议 ≥1000, BCa 建议 ≥2000)。
        ci_level: 置信水平。
        method: ``"percentile"`` (百分位 CI) or ``"bca"`` (BCa CI)。
        random_seed: 随机种子。

    Returns:
        BootstrapResult。
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    rng = np.random.default_rng(random_seed)

    # 原始统计量
    original = float(statistic_fn(arr))

    # 重抽样
    boot_stats = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot_stats[b] = statistic_fn(arr[idx])

    boot_mean = float(np.mean(boot_stats))
    boot_se = float(np.std(boot_stats, ddof=1))
    boot_bias = boot_mean - original

    # 百分位 CI
    alpha = 1 - ci_level
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    ci_perc = (float(np.percentile(boot_stats, lo)), float(np.percentile(boot_stats, hi)))

    # BCa CI
    ci_bca = None
    if method == "bca" and n_resamples >= 1000:
        ci_bca = _bca_ci(arr, statistic_fn, boot_stats, original, alpha, rng, n_resamples)

    return BootstrapResult(
        original_stat=original,
        bootstrap_mean=boot_mean,
        bootstrap_se=boot_se,
        bootstrap_bias=boot_bias,
        ci_percentile=ci_perc,
        ci_bca=ci_bca,
        n_resamples=n_resamples,
        method="simple",
        ci_level=ci_level,
    )


def _bca_ci(
    data: np.ndarray,
    stat_fn,
    boot_stats: np.ndarray,
    original: float,
    alpha: float,
    rng: np.random.Generator,
    n_resamples: int,
) -> tuple[float, float]:
    """BCa 置信区间 (偏差校正与加速)。

    z0: 偏差校正因子 = Φ⁻¹(比例 of boot_stats < original)
    a: 加速因子 (Jackknife)
    """
    n = len(data)

    # z0: 偏差校正
    prop_less = np.mean(boot_stats < original)
    z0 = float(_normal_ppf(max(min(prop_less, 0.999), 0.001)))

    # a: Jackknife 加速因子
    jack_stats = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack_data = np.delete(data, i)
        jack_stats[i] = stat_fn(jack_data)
    jack_mean = np.mean(jack_stats)
    numer = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_stats) ** 2)) ** 1.5
    a = numer / denom if denom > 0 else 0.0

    # 调整分位
    z_alpha_2 = _normal_ppf(alpha / 2)
    z_1_alpha_2 = _normal_ppf(1 - alpha / 2)

    p_lo = _normal_cdf(z0 + (z0 + z_alpha_2) / (1 - a * (z0 + z_alpha_2)))
    p_hi = _normal_cdf(z0 + (z0 + z_1_alpha_2) / (1 - a * (z0 + z_1_alpha_2)))

    lo = float(np.percentile(boot_stats, max(p_lo * 100, 0)))
    hi = float(np.percentile(boot_stats, min(p_hi * 100, 100)))

    return (lo, hi)


def _normal_ppf(p: float) -> float:
    """标准正态分位数 (避免 scipy 依赖小函数)。"""
    from scipy import stats as sp_stats

    return float(sp_stats.norm.ppf(p))


def _normal_cdf(x: float) -> float:
    from scipy import stats as sp_stats

    return float(sp_stats.norm.cdf(x))


# ═══════════════════════════════════════════
# 分层 Bootstrap
# ═══════════════════════════════════════════


def bootstrap_stratified(
    data: np.ndarray,
    strata: np.ndarray,
    statistic_fn,
    *,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    random_seed: int | None = 42,
) -> BootstrapResult:
    """分层 Bootstrap (保持每层样本量不变)。

    Args:
        data: 一维数组。
        strata: 分层标签 (与 data 等长)。
        statistic_fn: 统计量函数 callable(data) -> float。

    Returns:
        BootstrapResult (method="stratified")。
    """
    arr = np.asarray(data, dtype=np.float64)
    s_arr = np.asarray(strata)
    mask = ~np.isnan(arr)
    arr, s_arr = arr[mask], s_arr[mask]

    rng = np.random.default_rng(random_seed)

    original = float(statistic_fn(arr))

    # 按层分别存储索引
    unique_strata = np.unique(s_arr)
    strata_indices = {s: np.where(s_arr == s)[0] for s in unique_strata}

    boot_stats = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = []
        for s, si in strata_indices.items():
            n_s = len(si)
            boot_idx = rng.choice(si, size=n_s, replace=True)
            idx.extend(boot_idx)
        idx = np.array(idx)
        boot_stats[b] = statistic_fn(arr[idx])

    boot_mean = float(np.mean(boot_stats))
    boot_se = float(np.std(boot_stats, ddof=1))
    boot_bias = boot_mean - original

    alpha = 1 - ci_level
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    ci_perc = (float(np.percentile(boot_stats, lo)), float(np.percentile(boot_stats, hi)))

    return BootstrapResult(
        original_stat=original,
        bootstrap_mean=boot_mean,
        bootstrap_se=boot_se,
        bootstrap_bias=boot_bias,
        ci_percentile=ci_perc,
        ci_bca=None,
        n_resamples=n_resamples,
        method="stratified",
        ci_level=ci_level,
    )


# ═══════════════════════════════════════════
# 便捷方法: 均值/中位数等常见统计量
# ═══════════════════════════════════════════


def bootstrap_mean(
    data: np.ndarray,
    *,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    method: str = "percentile",
    random_seed: int | None = 42,
) -> BootstrapResult:
    """Bootstrap 总体均值的区间估计。"""
    return bootstrap(data, lambda x: float(np.mean(x)),
                     n_resamples=n_resamples, ci_level=ci_level, method=method,
                     random_seed=random_seed)


def bootstrap_median(
    data: np.ndarray,
    *,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    random_seed: int | None = 42,
) -> BootstrapResult:
    """Bootstrap 总体中位数的区间估计 (注意: 中位数非光滑统计量)。"""
    return bootstrap(data, lambda x: float(np.median(x)),
                     n_resamples=n_resamples, ci_level=ci_level, method="percentile",
                     random_seed=random_seed)


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    random_seed: int | None = 42,
) -> BootstrapResult:
    """Bootstrap Pearson 相关系数的区间估计 (成对重抽样)。"""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = (~np.isnan(x_arr)) & (~np.isnan(y_arr))
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    n = len(x_arr)

    original = float(np.corrcoef(x_arr, y_arr)[0, 1])
    rng = np.random.default_rng(random_seed)

    boot_stats = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot_stats[b] = float(np.corrcoef(x_arr[idx], y_arr[idx])[0, 1])

    boot_mean = float(np.mean(boot_stats))
    boot_se = float(np.std(boot_stats, ddof=1))

    alpha = 1 - ci_level
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    ci_perc = (float(np.percentile(boot_stats, lo)), float(np.percentile(boot_stats, hi)))

    return BootstrapResult(
        original_stat=original,
        bootstrap_mean=boot_mean,
        bootstrap_se=boot_se,
        bootstrap_bias=boot_mean - original,
        ci_percentile=ci_perc,
        ci_bca=None,
        n_resamples=n_resamples,
        method="simple",
        ci_level=ci_level,
    )


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def bootstrap_report(r: BootstrapResult, label: str = "统计量") -> str:
    """Bootstrap 报告文本。"""
    lines = [
        f"{'='*50}",
        f"  Bootstrap 分析: {label}",
        f"  方法: {r.method}, 重抽样次数: {r.n_resamples}",
        f"{'='*50}",
        f"  原始估计:      {r.original_stat:.4f}",
        f"  Bootstrap 均值: {r.bootstrap_mean:.4f}",
        f"  Bootstrap SE:   {r.bootstrap_se:.4f}",
        f"  Bias:           {r.bootstrap_bias:.4f}",
        f"  Percentile {int(r.ci_level*100)}% CI: [{r.ci_percentile[0]:.4f}, {r.ci_percentile[1]:.4f}]",
    ]
    if r.ci_bca:
        lines.append(f"  BCa {int(r.ci_level*100)}% CI:            [{r.ci_bca[0]:.4f}, {r.ci_bca[1]:.4f}]")
    lines.append("=" * 50)
    return "\n".join(lines)
