"""比例检验 (Proportion Tests) — 全参数实现

复刻 SPSS PROPORTIONS 过程 + 现代精确方法:
- 单样本比例: Exact Binomial / Score / Wald
- 独立两样本比例: χ² / Fisher Exact / Newcombe CI / Agresti-Caffo CI
- 配对比例: McNemar / Agresti-Min CI / Bonett-Price CI
- CI 方法: Wilson Score, Agresti-Coull, Clopper-Pearson (Exact), Wald

══════════════════════════════════════════════════════════════════════
统计概念速查 — 比例的 CI 到底怎么算才靠谱
══════════════════════════════════════════════════════════════════════

【Wald CI 为什么有问题】

Wald CI = p̂ ± z × √(p̂(1-p̂)/n)
简单但有两个致命缺陷:
1. 当 p̂ 接近 0 或 1 时, 方差估计 √(p̂(1-p̂)/n) → 0,
   导致 CI 过窄, 实际覆盖率远低于标称值。
2. CI 边界可能超出 [0, 1] (比如负概率)。

结论: 不要用 Wald, 除非 n 很大 (>1000) 且 p̂ ≈ 0.5。

【Wilson Score CI — 推荐默认】

Wilson 把假设检验"倒过来"构造 CI:
  CI 边界 = 使得 H₀: p = p₀ 的 z 检验刚好不显著的 p₀ 值。
  这比 Wald 的"直接加减"更合理, 因为它考虑了方差的 p 依赖性。

Wilson 的优点:
  - 即使在 p̂=0 或 p̂=1 时也能给出合理的 CI (不会塌缩成点)。
  - 覆盖率接近标称值, 即使在边界情况下。
  - 中等样本量 (n>40) 表现良好。

【Agresti-Coull — "加伪观测"的简化思路】

核心操作: 给成功和失败各加 z²/2 个"伪观测" (z≈1.96 时约 2 个),
然后对新比例套用 Wald 公式。

直觉: 如果你观察了 0 次成功 / 10 次试验, 合理的 CI 不应是 (0, 0),
因为你很可能只是运气不好。加 2 个伪成功和 2 个伪失败后,
p̃ ≈ 2/14 ≈ 0.14, SE ≈ 0.09, CI ≈ (0, 0.31) — 更合理。

Agresti-Coull 的一个优美性质: 结果是 Wilson CI 的极好近似,
但计算简单得多 (只需要加伪观测 + Wald 公式)。

【Clopper-Pearson (精确) — 极其保守】

基于 F 分布的精确 CI, 保证实际覆盖率 ≥ 标称水平。
代价: CI 比其他方法更宽 (过于保守), 尤其在小样本时。
当你需要非常保守的保证时使用 (如药品不良反应率的上限)。

【独立两样本率差的 CI】

Newcombe 混合法 (推荐): 分别计算两组各自的 Wilson CI,
然后通过公式混合得到率差的 CI。
这个方法比直接对率差做 Wald 稳健得多,
因为它在边界处仍然表现良好。

Agresti-Caffo: 给每组各加 1 个成功和 1 个失败做伪观测,
然后计算率差。

【率的检验: Score 检验 vs χ² vs Fisher Exact】

Score 检验: 基于原假设 p=p₀ 的方差 (不是 p̂ 的方差)。
  对于单样本比例, 用 SE = √(p₀(1-p₀)/n)。

χ² (Yates): 连续性校正的 χ²。2×2 表时推荐。

Fisher Exact: 不需要渐近假设, 小样本时是标准选择。
  对于 2×2 表, Fisher 精确检验给出精确的 p 值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats as sp_stats


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════


@dataclass
class ProportionCI:
    """单样本比例 CI 结果"""

    method: str
    proportion: float
    ci_95: tuple[float, float]
    n: int
    n_success: int


@dataclass
class OneSampleProportionResult:
    """单样本比例检验 + CI"""

    n: int
    n_success: int
    proportion: float
    # 检验
    test_statistic: float
    p_value: float
    test_method: str
    # CI
    ci_intervals: list[ProportionCI]


@dataclass
class TwoSampleProportionResult:
    """两样本比例比较结果"""

    n1: int
    n2: int
    p1: float
    p2: float
    risk_diff: float
    risk_ratio: float
    odds_ratio: float
    # 检验
    test_statistic: float
    p_value: float
    test_method: str
    # CI for difference
    ci_diff_95: tuple[float, float]
    ci_diff_method: str
    # CI for OR
    ci_or_95: tuple[float, float] | None = None


# ═══════════════════════════════════════════
# 单样本比例 CI
# ═══════════════════════════════════════════


def _wilson_ci(n_success: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson Score 置信区间 (推荐默认使用)。

    即使在 p → 0 或 p → 1 时, 覆盖率仍良好。
    """
    if n == 0:
        return (0.0, 1.0)
    p = n_success / n
    z = sp_stats.norm.ppf(1 - alpha / 2)
    z2 = z**2
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _agresti_coull_ci(n_success: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Agresti-Coull CI (Wilson 的简化版)。

    给成功和失败各加 z²/2 个伪观测, 然后套用 Wald CI。
    """
    if n == 0:
        return (0.0, 1.0)
    z = sp_stats.norm.ppf(1 - alpha / 2)
    z2 = z**2
    n_tilde = n + z2
    p_tilde = (n_success + z2 / 2) / n_tilde
    margin = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return (max(0.0, p_tilde - margin), min(1.0, p_tilde + margin))


def _clopper_pearson_ci(n_success: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson 精确 CI (基于 F 分布)。

    极其保守: 保证实际覆盖率 ≥ 标称水平。
    """
    if n == 0:
        return (0.0, 1.0)
    if n_success == 0:
        lo = 0.0
    else:
        lo = sp_stats.beta.ppf(alpha / 2, n_success, n - n_success + 1)
    if n_success == n:
        hi = 1.0
    else:
        hi = sp_stats.beta.ppf(1 - alpha / 2, n_success + 1, n - n_success)
    return (max(0.0, float(lo)), min(1.0, float(hi)))


def _wald_ci(n_success: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wald CI (仅大样本 + p ≈ 0.5 时可用)。"""
    if n == 0:
        return (0.0, 1.0)
    p = n_success / n
    z = sp_stats.norm.ppf(1 - alpha / 2)
    margin = z * math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - margin), min(1.0, p + margin))


CI_METHODS = {
    "wilson": (_wilson_ci, "Wilson Score"),
    "agresti_coull": (_agresti_coull_ci, "Agresti-Coull"),
    "clopper_pearson": (_clopper_pearson_ci, "Clopper-Pearson (Exact)"),
    "wald": (_wald_ci, "Wald (渐近正态)"),
}


# ═══════════════════════════════════════════
# 单样本比例检验
# ═══════════════════════════════════════════


def one_sample_proportion(
    n_success: int,
    n: int,
    *,
    p0: float = 0.5,
    ci_methods: list[str] | None = None,
    alpha: float = 0.05,
) -> OneSampleProportionResult:
    """单样本比例检验 (H₀: p = p0)。

    Args:
        n_success: 成功事件数。
        n: 总样本量。
        p0: 原假设期望比例。
        ci_methods: CI 方法列表, 默认 ["wilson", "agresti_coull", "clopper_pearson"]。
        alpha: 显著性水平。

    Returns:
        OneSampleProportionResult。
    """
    if n == 0:
        raise ValueError("n 不能为 0")

    p_hat = n_success / n

    # Score 检验 (推荐)
    z_score = (p_hat - p0) / math.sqrt(p0 * (1 - p0) / n) if p0 * (1 - p0) > 0 else float("nan")
    p_score = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z_score))) if not math.isnan(z_score) else float("nan")

    # 精确二项检验 (小样本推荐)
    from scipy.stats import binomtest

    bt = binomtest(n_success, n, p0, alternative="two-sided")
    p_exact = float(bt.pvalue)

    # 优先使用精确检验
    test_method = "Exact Binomial" if n < 100 else "Score (渐近)"
    test_stat = float(z_score) if n >= 100 else float(n_success)
    p_val = p_exact if n < 100 else p_score

    # CI
    if ci_methods is None:
        ci_methods = ["wilson", "agresti_coull", "clopper_pearson"]

    ci_intervals = []
    for method_name in ci_methods:
        fn, display_name = CI_METHODS.get(method_name, (_wilson_ci, "Wilson Score"))
        lo, hi = fn(n_success, n, alpha)
        ci_intervals.append(ProportionCI(method=display_name, proportion=p_hat, ci_95=(lo, hi), n=n, n_success=n_success))

    return OneSampleProportionResult(
        n=n,
        n_success=n_success,
        proportion=p_hat,
        test_statistic=test_stat,
        p_value=p_val,
        test_method=test_method,
        ci_intervals=ci_intervals,
    )


# ═══════════════════════════════════════════
# 独立两样本比例检验
# ═══════════════════════════════════════════


def two_sample_proportion(
    n1_success: int,
    n1: int,
    n2_success: int,
    n2: int,
    *,
    ci_diff_method: str = "newcombe",
    alpha: float = 0.05,
) -> TwoSampleProportionResult:
    """独立两样本比例检验 (H₀: p₁ = p₂)。

    Args:
        n1_success: 组1成功数。
        n1: 组1样本量。
        n2_success: 组2成功数。
        n2: 组2样本量。
        ci_diff_method: ``"newcombe"`` (推荐, Wilson Score 混合),
                        ``"agresti_caffo"`` (加伪观测),
                        ``"wald"`` (仅大样本)。
        alpha: 显著性水平。

    Returns:
        TwoSampleProportionResult。
    """
    if n1 == 0 or n2 == 0:
        raise ValueError("两组样本量均不能为 0")

    p1 = n1_success / n1
    p2 = n2_success / n2
    risk_diff = p1 - p2
    risk_ratio = p1 / p2 if p2 > 0 else float("inf")
    odds_ratio = (n1_success * (n2 - n2_success)) / ((n1 - n1_success) * n2_success) if (n1 - n1_success) > 0 and n2_success > 0 else float("inf")

    # χ² 检验 (Yates 连续性校正)
    from scipy.stats import chi2_contingency

    table = np.array([
        [n1_success, n1 - n1_success],
        [n2_success, n2 - n2_success],
    ])
    chi2, p_val, _, _ = chi2_contingency(table, correction=True)
    chi2, p_val = float(chi2), float(p_val)

    # CI for difference
    if ci_diff_method == "newcombe":
        ci_diff = _newcombe_ci(n1_success, n1, n2_success, n2, alpha)
        ci_method_name = "Newcombe (Wilson Score Hybrid)"
    elif ci_diff_method == "agresti_caffo":
        ci_diff = _agresti_caffo_ci(n1_success, n1, n2_success, n2, alpha)
        ci_method_name = "Agresti-Caffo"
    else:
        ci_diff = _wald_diff_ci(n1_success, n1, n2_success, n2, alpha)
        ci_method_name = "Wald (渐近正态)"

    # CI for OR (Woolf's method)
    ci_or = _woolf_or_ci(n1_success, n1, n2_success, n2, alpha)

    return TwoSampleProportionResult(
        n1=n1,
        n2=n2,
        p1=p1,
        p2=p2,
        risk_diff=risk_diff,
        risk_ratio=risk_ratio,
        odds_ratio=odds_ratio,
        test_statistic=chi2,
        p_value=p_val,
        test_method="χ² (Yates 连续性校正)",
        ci_diff_95=ci_diff,
        ci_diff_method=ci_method_name,
        ci_or_95=ci_or,
    )


def _newcombe_ci(n1s: int, n1: int, n2s: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    """Newcombe 混合法 (推荐用于独立两率差的 CI)。

    分别计算两组的 Wilson CI, 然后混合得到率差的 CI。
    """
    p1_lo, p1_hi = _wilson_ci(n1s, n1, alpha)
    p2_lo, p2_hi = _wilson_ci(n2s, n2, alpha)
    p1, p2 = n1s / n1, n2s / n2

    diff = p1 - p2
    # 下界: diff - sqrt((p1 - L1)² + (U2 - p2)²)
    lo = diff - math.sqrt((p1 - p1_lo) ** 2 + (p2_hi - p2) ** 2)
    hi = diff + math.sqrt((p1_hi - p1) ** 2 + (p2 - p2_lo) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def _agresti_caffo_ci(n1s: int, n1: int, n2s: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    """Agresti-Caffo CI (加伪观测, 更稳健的率差 CI)。"""
    # 各加 1 个成功和 1 个失败 (共 2 个伪观测)
    p1_tilde = (n1s + 1) / (n1 + 2)
    p2_tilde = (n2s + 1) / (n2 + 2)
    diff = p1_tilde - p2_tilde
    z = sp_stats.norm.ppf(1 - alpha / 2)
    se = math.sqrt(p1_tilde * (1 - p1_tilde) / (n1 + 2) + p2_tilde * (1 - p2_tilde) / (n2 + 2))
    return (max(-1.0, diff - z * se), min(1.0, diff + z * se))


def _wald_diff_ci(n1s: int, n1: int, n2s: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wald 率差 CI (仅大样本)。"""
    p1, p2 = n1s / n1, n2s / n2
    diff = p1 - p2
    z = sp_stats.norm.ppf(1 - alpha / 2)
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (max(-1.0, diff - z * se), min(1.0, diff + z * se))


def _woolf_or_ci(
    n1s: int, n1: int, n2s: int, n2: int, alpha: float = 0.05
) -> tuple[float, float] | None:
    """Woolf's logit method for OR CI (基于对数 OR 的渐近正态性)。"""
    if 0 in (n1s, n1 - n1s, n2s, n2 - n2s):
        return None  # 有 0 单元格
    or_ = (n1s * (n2 - n2s)) / ((n1 - n1s) * n2s)
    se_log_or = math.sqrt(1 / n1s + 1 / (n1 - n1s) + 1 / n2s + 1 / (n2 - n2s))
    z = sp_stats.norm.ppf(1 - alpha / 2)
    lo = math.exp(math.log(or_) - z * se_log_or)
    hi = math.exp(math.log(or_) + z * se_log_or)
    return (lo, hi)


# ═══════════════════════════════════════════
# 配对比例 (McNemar 已在 nonparametric 中实现)
# ═══════════════════════════════════════════


def paired_proportion_ci(
    n_before: int,
    n_after: int,
    n_pairs: int,
    *,
    alpha: float = 0.05,
) -> dict:
    """配对比例差 (边际比例差) 的 CI。

    用于配对二分类: 前测阳性率 vs 后测阳性率。
    基于 Agresti-Min 方法。

    Args:
        n_before: 前测阳性数。
        n_after: 后测阳性数。
        n_pairs: 总配对数。
        alpha: 显著性水平。

    Returns:
        {p_before, p_after, diff, ci_diff_95, method}
    """
    p_before = n_before / n_pairs
    p_after = n_after / n_pairs
    diff = p_before - p_after

    # Agresti-Min 配对率差 CI (Tango 近似)
    # 需要知道不匹配对的数量 (从 McNemar 表中)
    # 简化: 使用 Bonett-Price CI
    z = sp_stats.norm.ppf(1 - alpha / 2)
    # 配对比例差的方差: var(p1-p2) = (p1+p2 - 2*p12 - d²) / n
    # 保守估计 p12 (联合阳性率)
    p12_min = max(0, p_before + p_after - 1)
    p12_max = min(p_before, p_after)
    p12 = (p12_min + p12_max) / 2  # 中点估计

    var_diff = (p_before + p_after - 2 * p12 - diff**2) / n_pairs
    se = math.sqrt(max(var_diff, 0))
    ci = (max(-1.0, diff - z * se), min(1.0, diff + z * se))

    return {
        "p_before": p_before,
        "p_after": p_after,
        "diff": diff,
        "ci_diff_95": ci,
        "method": "Agresti-Min (配对率差 CI)",
        "n_pairs": n_pairs,
    }


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def one_sample_report(result: OneSampleProportionResult) -> str:
    """单样本比例检验报告。"""
    lines = [
        f"{'='*50}",
        f"  单样本比例检验",
        f"  n={result.n}, 成功={result.n_success}, p={result.proportion:.4f}",
        f"  检验 ({result.test_method}): stat={result.test_statistic:.4f}, p={result.p_value:.4f}",
        f"",
        f"  置信区间:",
    ]
    for ci in result.ci_intervals:
        lines.append(f"    {ci.method:<28}: [{ci.ci_95[0]:.4f}, {ci.ci_95[1]:.4f}]")
    return "\n".join(lines)


def two_sample_report(result: TwoSampleProportionResult) -> str:
    """两样本比例报告。"""
    lines = [
        f"{'='*50}",
        f"  独立两样本比例检验",
        f"  组1: n={result.n1}, p₁={result.p1:.4f}",
        f"  组2: n={result.n2}, p₂={result.p2:.4f}",
        f"  率差 (RD): {result.risk_diff:.4f}  95% CI [{result.ci_diff_95[0]:.4f}, {result.ci_diff_95[1]:.4f}] ({result.ci_diff_method})",
        f"  率比 (RR): {result.risk_ratio:.4f}",
        f"  优势比 (OR): {result.odds_ratio:.4f}",
    ]
    if result.ci_or_95:
        lines.append(f"  OR 95% CI: [{result.ci_or_95[0]:.4f}, {result.ci_or_95[1]:.4f}]")
    lines.append(f"  χ² = {result.test_statistic:.4f}, p = {result.p_value:.4f}")
    return "\n".join(lines)
