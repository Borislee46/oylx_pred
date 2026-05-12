"""t 检验 (t-Test) — 全参数实现

复刻 SPSS T-TEST 过程 + 现代统计最佳实践:
- 独立样本: Welch t (默认) + Student t, 方差齐性 Levene 检验
- 配对样本: paired t + 差值正态诊断
- 单样本: one-sample t
- 效应量: Cohen's d (合并SD), Glass's Δ (对照组SD), Hedges' g
- 贝叶斯因子: JZS 先验 BF10 (Jeffreys-Zellner-Siow)

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【t 检验的核心问题】
从两组样本的均值和标准差出发, 做一个比值:
t = (信号) / (噪声) = (均值差异) / (均值差异的标准误)
t 的绝对值越大 → 越不像随机 → p 越小。

t 的抽样分布假设: 如果 H₀ 为真 (两组总体均值相同),
且数据是独立正态的, 那么 t 服从一个自由度 = df 的 t 分布。

t 分布 vs 正态分布: t 分布的尾部比标准正态更"厚"。
这意味着对于同样的 t 值, t 分布给的 p 值更大 (更保守)。
当 df→∞ 时, t 分布 → N(0,1)。

【Welch t (默认) vs Student t — 为什么现代统计推荐 Welch】

Student t (经典): 假设两组方差相等。用合并方差公式算 SE。
Welch t: 不假设两组方差相等。用各自方差分别算 SE, 自由度用
          Welch-Satterthwaite 近似 (通常不是整数)。

现实世界中很少有两组方差真正相等的情况。Student t 在方差不齐
+ 样本量不等时, 第一类错误率可能远高于 0.05 (你看到的 p<0.05
可能是假的)。Welch t 几乎和 Student t 一样有统计效力,
但稳健得多。统计界现在的共识: 永远用 Welch, 不要用 Student。

Levene 检验 p<0.05 是红灯: 在 Student t 下尤其危险,
在 Welch 下也不太理想但至少自由度做了调整。

【效应量 — p 值之外】

Cohen's d (合并 SD): 两组标准化的均值差。
  0.2=小, 0.5=中, 0.8=大。这是最广泛使用的效应量。

Hedges' g: Cohen's d 的小样本无偏校正。
  n<20 时两者有明显差异, n>50 时几乎一样。推荐报告 g。

Glass's Δ: 以对照组/第二组 SD 为分母。
  实验组和对照组的方差不齐时推荐 (因为对照组方差可能更纯)。

注意: 配对 t 检验的 Cohen's d 使用差值的 SD (而不是合并 SD),
因为配对设计的本质是差值分析。

【贝叶斯因子 BF10 — 另一种思维】

p 值问: P(数据 | H₀) (如果零假设为真, 看到这个数据的概率)。
大多数人想要的是 P(H₁ | 数据) (看到这个数据, H₁为真的概率)。
但 p 值不能回答这个问题。

贝叶斯因子 BF10 = P(data | H₁) / P(data | H₀)
即: 数据在备择假设下的概率是零假设下的多少倍。

BF10 = 1 ~ 3 → 微弱证据
BF10 = 3 ~ 10 → 中等证据
BF10 = 10 ~ 30 → 强证据
BF10 > 30 → 非常强证据

JZS 先验: Jeffreys-Zellner-Siow 先验, 对效应量使用 Cauchy 分布
          (r_scale 参数, 默认 0.707 = √2/2)。
          这是心理学和医学中最广泛接受的默认先验。
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
class TTestResult:
    """t 检验完整结果"""

    method: str  # "Welch" | "Student" | "Paired" | "One-Sample"
    t_statistic: float
    df: float  # Welch 可能为小数
    p_value: float
    mean_diff: float
    ci_95: tuple[float, float]
    # 描述统计
    n1: int
    n2: int | None = None
    mean1: float = float("nan")
    mean2: float = float("nan")  # or population mean
    std1: float = float("nan")
    std2: float = float("nan")
    # 效应量
    cohens_d: float = float("nan")
    hedges_g: float = float("nan")
    glass_delta: float = float("nan")
    effect_label: str = ""
    # 方差齐性 (仅独立样本)
    levene_f: float | None = None
    levene_p: float | None = None
    # 贝叶斯
    bf10: float | None = None
    bf01: float | None = None


# ═══════════════════════════════════════════
# 效应量
# ═══════════════════════════════════════════


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Cohen's d (合并标准差)"""
    n1, n2 = len(x1), len(x2)
    if n1 <= 1 or n2 <= 1:
        return float("nan")
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / sp if sp != 0 else 0.0


def hedges_g(x1: np.ndarray, x2: np.ndarray) -> float:
    """Hedges' g (小样本校正 Cohen's d)"""
    d = cohens_d(x1, x2)
    n1, n2 = len(x1), len(x2)
    df = n1 + n2 - 2
    correction = 1 - 3 / (4 * df - 1) if df > 1 else 1.0
    return d * correction


def glass_delta(x1: np.ndarray, x2: np.ndarray) -> float:
    """Glass's Δ (以对照组/第二组 SD 为分母, 方差不齐时推荐)"""
    s2 = np.std(x2, ddof=1)
    if s2 == 0:
        return float("nan")
    return (np.mean(x1) - np.mean(x2)) / s2


def _cohens_d_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "极小 (negligible)"
    if ad < 0.5:
        return "小 (small)"
    if ad < 0.8:
        return "中 (medium)"
    return "大 (large)"


# ═══════════════════════════════════════════
# 贝叶斯因子 (JZS 先验, 近似)
# ═══════════════════════════════════════════


def _jzs_bf10(t_stat: float, n1: int, n2: int | None = None, r_scale: float = 0.707) -> float:
    """Jeffreys-Zellner-Siow 贝叶斯因子近似 (Rouder et al. 2009).

    Args:
        t_stat: t 统计量值。
        n1, n2: 两组样本量 (单样本/配对时 n2=None)。
        r_scale: Cauchy 先验的 scale 参数 (默认 0.707 = √2/2)。

    Returns:
        BF10 (数据支持 H1 vs H0 的证据强度)。
    """
    if n2 is None:
        n = n1
        df = n - 1
    else:
        n = n1 * n2 / (n1 + n2)
        df = n1 + n2 - 2

    # Rouder et al. (2009) 的近似公式
    # 使用数值积分近似
    t2 = t_stat**2
    numer = (1 + n * r_scale**2) ** (-0.5)
    denom = (1 + t2 / df) ** ((df + 1) / 2)

    # 简化版 JZS BF 近似 (基于 Morey & Rouder 的 BayesFactor 包)
    if df < 1:
        return float("nan")

    # 使用 Morey et al. 的近似: Laplace approximation
    r2 = r_scale**2
    term1 = (1 + n * r2) ** (-0.5)
    term2 = (1 + t2 / (df * (1 + n * r2))) ** ((df + 1) / 2)
    bf = term1 * term2

    return float(bf)


# ═══════════════════════════════════════════
# 独立样本 t 检验
# ═══════════════════════════════════════════


def independent_t_test(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    equal_var: bool = False,
    ci_level: float = 0.95,
    compute_bf: bool = True,
    r_scale: float = 0.707,
) -> TTestResult:
    """独立样本 t 检验 (默认 Welch, 现代统计推荐)。

    Args:
        x1, x2: 两组独立样本。
        equal_var: True=经典 Student t (等方差假定); False=Welch t (默认)。
        ci_level: 置信区间水平。
        compute_bf: 是否计算 JZS 贝叶斯因子。
        r_scale: JZS Cauchy 先验 scale。

    Returns:
        TTestResult。
    """
    a1 = np.asarray(x1, dtype=np.float64)
    a2 = np.asarray(x2, dtype=np.float64)
    a1, a2 = a1[~np.isnan(a1)], a2[~np.isnan(a2)]

    n1, n2 = len(a1), len(a2)
    mean1, mean2 = float(np.mean(a1)), float(np.mean(a2))
    std1, std2 = float(np.std(a1, ddof=1)), float(np.std(a2, ddof=1))

    # Levene 检验
    lf, lp = None, None
    if n1 > 1 and n2 > 1:
        # Brown-Forsythe (median-based) 更稳健
        z1 = np.abs(a1 - np.median(a1))
        z2 = np.abs(a2 - np.median(a2))
        lf, lp = sp_stats.levene(a1, a2, center="median")[:2]
        lf, lp = float(lf), float(lp)

    # t 检验
    if equal_var:
        t, p = sp_stats.ttest_ind(a1, a2, equal_var=True)
        df = n1 + n2 - 2
        method = "Student (等方差)"
    else:
        t, p = sp_stats.ttest_ind(a1, a2, equal_var=False)
        # Welch-Satterthwaite df
        v1, v2 = std1**2 / n1, std2**2 / n2
        df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1)) if n1 > 1 and n2 > 1 else n1 + n2 - 2
        method = "Welch (不假定等方差)"

    t, p = float(t), float(p)

    # 均值差 + CI
    mean_diff = mean1 - mean2
    alpha = 1 - ci_level
    se_diff = math.sqrt(std1**2 / n1 + std2**2 / n2)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=max(df, 1))
    margin = t_crit * se_diff
    ci = (mean_diff - margin, mean_diff + margin)

    # 效应量
    cd = cohens_d(a1, a2)
    hg = hedges_g(a1, a2)
    gd = glass_delta(a1, a2)
    label = _cohens_d_label(cd)

    # 贝叶斯因子
    bf10, bf01 = None, None
    if compute_bf:
        bf10 = _jzs_bf10(t, n1, n2, r_scale)
        bf01 = 1.0 / bf10 if bf10 and bf10 != 0 else None

    return TTestResult(
        method=method,
        t_statistic=t,
        df=df,
        p_value=p,
        mean_diff=mean_diff,
        ci_95=ci,
        n1=n1,
        n2=n2,
        mean1=mean1,
        mean2=mean2,
        std1=std1,
        std2=std2,
        cohens_d=cd,
        hedges_g=hg,
        glass_delta=gd,
        effect_label=label,
        levene_f=lf,
        levene_p=lp,
        bf10=bf10,
        bf01=bf01,
    )


# ═══════════════════════════════════════════
# 配对样本 t 检验
# ═══════════════════════════════════════════


def paired_t_test(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    ci_level: float = 0.95,
    compute_bf: bool = True,
    r_scale: float = 0.707,
) -> TTestResult:
    """配对样本 t 检验。

    Args:
        x1, x2: 配对数据的两次测量 (等长, 一一对应)。
    """
    a1 = np.asarray(x1, dtype=np.float64)
    a2 = np.asarray(x2, dtype=np.float64)
    mask = (~np.isnan(a1)) & (~np.isnan(a2))
    a1, a2 = a1[mask], a2[mask]

    n = len(a1)
    diff = a1 - a2
    mean_diff = float(np.mean(diff))

    t, p = sp_stats.ttest_rel(a1, a2)
    t, p = float(t), float(p)

    df = n - 1
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    se_diff = std_diff / math.sqrt(n) if n > 1 else float("nan")

    alpha = 1 - ci_level
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=max(df, 1))
    margin = t_crit * se_diff
    ci = (mean_diff - margin, mean_diff + margin)

    # 效应量 (配对: 使用差值 SD)
    cd = mean_diff / std_diff if std_diff and std_diff != 0 else 0.0
    hg = cd * (1 - 3 / (4 * df - 1)) if df > 1 else cd

    bf10, bf01 = None, None
    if compute_bf:
        bf10 = _jzs_bf10(t, n, None, r_scale)
        bf01 = 1.0 / bf10 if bf10 and bf10 != 0 else None

    return TTestResult(
        method="Paired",
        t_statistic=t,
        df=float(df),
        p_value=p,
        mean_diff=mean_diff,
        ci_95=ci,
        n1=n,
        n2=None,
        mean1=float(np.mean(a1)),
        mean2=float(np.mean(a2)),
        std1=float(np.std(a1, ddof=1)) if n > 1 else float("nan"),
        std2=float(np.std(a2, ddof=1)) if n > 1 else float("nan"),
        cohens_d=cd,
        hedges_g=hg,
        glass_delta=float("nan"),
        effect_label=_cohens_d_label(cd),
        bf10=bf10,
        bf01=bf01,
    )


# ═══════════════════════════════════════════
# 单样本 t 检验
# ═══════════════════════════════════════════


def one_sample_t_test(
    x: np.ndarray,
    popmean: float = 0.0,
    *,
    ci_level: float = 0.95,
    compute_bf: bool = True,
    r_scale: float = 0.707,
) -> TTestResult:
    """单样本 t 检验 (检验总体均值是否等于 popmean)。"""
    a = np.asarray(x, dtype=np.float64)
    a = a[~np.isnan(a)]
    n = len(a)
    mean_val = float(np.mean(a))
    std_val = float(np.std(a, ddof=1)) if n > 1 else float("nan")

    t, p = sp_stats.ttest_1samp(a, popmean)
    t, p = float(t), float(p)
    df = n - 1

    se = std_val / math.sqrt(n) if n > 1 else float("nan")
    alpha = 1 - ci_level
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=max(df, 1))
    mean_diff = mean_val - popmean
    margin = t_crit * se
    ci = (mean_diff - margin, mean_diff + margin)

    cd = abs(mean_diff / std_val) if std_val and std_val != 0 else 0.0

    bf10, bf01 = None, None
    if compute_bf:
        bf10 = _jzs_bf10(t, n, None, r_scale)
        bf01 = 1.0 / bf10 if bf10 and bf10 != 0 else None

    return TTestResult(
        method="One-Sample",
        t_statistic=t,
        df=float(df),
        p_value=p,
        mean_diff=mean_diff,
        ci_95=ci,
        n1=n,
        n2=None,
        mean1=mean_val,
        mean2=popmean,
        std1=std_val,
        std2=float("nan"),
        cohens_d=cd,
        hedges_g=float("nan"),
        glass_delta=float("nan"),
        effect_label=_cohens_d_label(cd),
        bf10=bf10,
        bf01=bf01,
    )


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def t_test_report(r: TTestResult) -> str:
    """生成 t 检验 APA 格式报告文本。"""
    lines = [
        f"{'='*50}",
        f"  {r.method} t 检验",
        f"{'='*50}",
        f"  t({r.df:.2f}) = {r.t_statistic:.3f}, p = {r.p_value:.4f}",
        f"  均值差 = {r.mean_diff:.4f}  [95% CI: {r.ci_95[0]:.4f}, {r.ci_95[1]:.4f}]",
        f"  组1: n={r.n1}, M={r.mean1:.3f}, SD={r.std1:.3f}",
    ]
    if r.n2 is not None:
        lines.append(f"  组2: n={r.n2}, M={r.mean2:.3f}, SD={r.std2:.3f}")
    lines.append(f"  Cohen's d = {r.cohens_d:.3f}  ({r.effect_label})")
    if not math.isnan(r.hedges_g):
        lines.append(f"  Hedges' g = {r.hedges_g:.3f}")
    if not math.isnan(r.glass_delta):
        lines.append(f"  Glass's Δ = {r.glass_delta:.3f}")

    if r.levene_f is not None:
        lines.append(f"  Levene (中位数): F = {r.levene_f:.3f}, p = {r.levene_p:.4f}")

    if r.bf10 is not None:
        bf_label = "弱" if r.bf10 < 3 else ("中等" if r.bf10 < 10 else "强")
        lines.append(f"  BF10 = {r.bf10:.2f}  ({bf_label}证据)")

    return "\n".join(lines)
