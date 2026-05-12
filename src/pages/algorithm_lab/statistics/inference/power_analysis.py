"""统计效力分析 — 全参数实现

复刻 SPSS 样本量估计功能 + G*Power 核心逻辑:
- 独立/配对 t 检验效力与样本量
- 单因素 ANOVA 效力与样本量
- 卡方检验效力与样本量
- Pearson 相关效力与样本量
- a priori（事前求样本量）vs post hoc（事后算效力）
- 使用非中心分布 (nct, ncf, ncx2) 精确计算
- brentq 搜索反解样本量

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【什么是统计效力 (Power)】

效力 = P(拒绝 H0 | H1 为真)
     = 1 - β (β = 第二类错误率)
     = 当真实效应存在时，正确检出它的概率。

如果你的研究效力只有 50%，那意味着即使真的有差异，
你有一半的概率会错误地得出"没有显著差异"的结论。
这就像蒙着眼睛丢骰子——结果是否显著更多取决于运气而非数据。

【四个互锁的量 — 固定三个，第四个被决定】

    分析前必须清醒意识到的关系：

    α (显著水平) — 你愿意接受的假阳性率（通常 0.05）
    ↓
    power (1-β) — 你希望有多大概率检出真实效应（通常 0.80）
    ↓
    效应量 — 你关心的差异在实际中有多大（d=0.5, f=0.25 等）
    ↓
    样本量 (n) — 你需要多少个被试来满足上述要求

    增大 n  → power 增大（更多信息 = 更容易辨别信号与噪声）
    增大 α  → power 增大（更容易拒绝 H0，但假阳性风险增大）
    更大的效应 → power 增大（越明显的差异越容易被检出）
    减小 α  → 需要增大 n 来维持 power

    公式（独立 t 检验）：
    n_per_group ≈ 2 × ((z_{1-α/2} + z_{1-β}) / d)²

    直观理解：分子 = "你希望达到的精度 + 你希望达到的确定度"
             分母 = "效应的真实大小"
             效应越小 → 需要的样本急剧增大（d=0.2 是 d=0.5 的 6.25 倍样本）

【非中心分布 — 当 H0 不成立时】

在 H0 成立时，检验统计量（t/F/χ²）服从"中心"分布（均值为 0 或 1）
在 H1 成立时，同一个统计量服从"非中心"分布（均值偏离中心）

非中心参数 (ncp) 量化了这个偏离的程度：

- 非中心 t 分布 (nct)：ncp = d × √(n/2)
- 非中心 F 分布 (ncf)：ncp = f² × N（N = 总样本量）
- 非中心 χ² 分布 (ncx2)：ncp = w² × n

ncp 越大 → 分布越远离 H0 → power 越大
这就是效应量(d/f/w)变大或样本量增大时 power 增大的数学本质。

【为什么 0.80 是 convention】

Cohen (1988) 建议 power = 0.80 作为"合理的平衡"。
含义：愿意接受 20% 的假阴性率（β=0.20）。

为什么不是 0.95？
  因为要把 power 从 0.80 提升到 0.95 需要约 50% 更多的样本——
  在实际中往往不可行。0.80 是务实和严谨之间的折衷。

什么情况下需要更高？
  - 医疗/安全决策（假阴性可能造成伤害）
  - 高成本干预（必须确定有效才执行）
  - 一次性机会（无法重复研究）

什么情况下可以更低？
  - 探索性分析（先找线索，后面再验证）
  - 避免错误地提升效应量夸大（宁可不显著也不能假阳性）

【a priori vs post hoc — 两种使用场景】

a priori（事前）：
  在研究开始之前：给定期望效应量和目标 power，算出需要多少样本。
  用法：power_t_test(n=None, d=0.5, power=0.80)
  → 回答"我需要多少数据才能检出 d=0.5 的效应？"
  这是最推荐的使用方式。

post hoc（事后）：
  研究结束后：给定已有样本量，算出能检出多大效应的效力。
  用法：power_t_test(n=50, d=0.5, power=None)
  → 回答"以我现有的 50 个样本，检出 d=0.5 的效力有多大？"
  注意：事后效力分析存在争议——如果结果不显著，事后效力低是必然的。
  主要用于"解释为什么没发现显著差异"而不是"证明没有差异"。

【效应量的参考基准 (Cohen's conventions)】

t 检验 (Cohen's d):
  d=0.2 — 小（如男女身高差异 2cm，约有差异但需要大样本才能检出）
  d=0.5 — 中（如培训前后绩效提升半档，肉眼可见）
  d=0.8 — 大（如通过/挂科的 GPA 差异，明显到没法忽视）

ANOVA (Cohen's f):
  f=0.10 — 小（组间差异微弱，如不同领导风格的团队产出微小差别）
  f=0.25 — 中（明显的组间差异，如三个培训方法的通过率明显不同）
  f=0.40 — 大（组间差异巨大，如入职渠道对留存率的强烈影响）
  f 与 η² 的换算：f² = η²/(1-η²)

卡方检验 (Cohen's w):
  w=0.10 — 小（如性别对某轻度偏好的微小影响）
  w=0.30 — 中（如学历对职业选择的中等程度关联）
  w=0.50 — 大（如吸烟对肺癌的强关联）

Pearson 相关 (r):
  r=0.10 — 小（微弱但或许真实存在的关联）
  r=0.30 — 中（明显的关联，如工作年限与薪资）
  r=0.50 — 大（强烈的关联，如前后测分数）

这些基准只是参考——你关心的效应量应由业务/科学背景决定，
不是由 Cohen 说了算。一个 2% 的离职率下降可能 d 很小，
但在 10000 人的公司里意味着 200 人——实际价值巨大。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats
from scipy import optimize


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════


@dataclass
class PowerResult:
    """效力分析结果"""

    test_type: str  # "independent_t" | "paired_t" | "anova" | "chi_square" | "correlation"
    n_required: int | None  # 所需样本量（若计算的是样本量）
    power_achieved: float | None  # 实际效力（若计算的是效力）
    effect_size: float  # 效应量 (d / f / w / r)
    alpha: float  # 显著水平
    alternative: str  # "two-sided" | "greater" | "less"
    n_per_group: int | None = None  # ANOVA 每组的样本量
    df: int | None = None  # 分子自由度（ANOVA=K-1, χ²=df）
    df2: int | None = None  # 分母自由度（ANOVA=K(n-1)）
    ncp: float | None = None  # 非中心参数 (实际值)


# ═══════════════════════════════════════════
# 内部工具
# ═══════════════════════════════════════════


def _solve_n(
    compute_power_fn,
    power_target: float,
    effect_size: float,
    alpha: float,
    bounds: tuple[float, float] = (2, 100000),
) -> int:
    """用二分搜索反解样本量：找到使 power(n) = power_target 的 n。

    brentq 在 ncp 很大时 nct.cdf 返回 NaN 会失败，
    改用逐步缩放的二分搜索，自动避开 NaN 区域。

    Args:
        compute_power_fn: 给定 n，返回 power 的函数。
        power_target: 目标效力（如 0.80）。
        effect_size: 效应量（仅用于日志，不直接使用）。
        alpha: 显著水平。
        bounds: 搜索范围（默认 2 ~ 100000）。

    Returns:
        所需样本量（向上取整的整数）。
    """
    lo, hi = bounds[0], bounds[1]

    # 找一个 power > target 且无 NaN 的上界
    # 避免在 ncp 很大时 nct.cdf 返回 NaN
    power_at_hi = compute_power_fn(hi)
    while (math.isnan(power_at_hi) or power_at_hi < power_target) and hi > lo * 2:
        hi = hi // 2  # 缩小搜索范围
        power_at_hi = compute_power_fn(hi)

    if math.isnan(power_at_hi) or power_at_hi < power_target:
        return -1  # 即使超大样本量也达不到目标效力

    power_at_lo = compute_power_fn(lo)
    if math.isnan(power_at_lo):
        while math.isnan(power_at_lo) and lo < hi:
            lo = int(lo * 1.5)
            power_at_lo = compute_power_fn(lo)

    if power_at_lo >= power_target:
        return int(lo)

    # 二分搜索
    for _ in range(100):
        mid = (lo + hi) / 2.0
        power_mid = compute_power_fn(mid)
        if math.isnan(power_mid) or power_mid >= power_target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.5:
            break

    return int(math.ceil(hi))


# ═══════════════════════════════════════════
# t 检验效力分析
# ═══════════════════════════════════════════


def power_t_test(
    n: int | None = None,
    d: float = 0.5,
    alpha: float = 0.05,
    power: float | None = None,
    alternative: str = "two-sided",
) -> PowerResult:
    """t 检验效力与样本量分析。

    通过非中心 t 分布 (nct) 精确计算效力。
    必须提供 n 或 power 中的一个（不能同时指定，不能同时不指定）。

    Args:
        n: 每组样本量。
            - 如果提供: 计算给定 n 和 d 的效力（post hoc 用法）。
            - 如果 None: 反解满足目标 power 所需的 n（a priori 用法）。
        d: Cohen's d 效应量（均值差 / 合并标准差）。
            ``0.2`` = 小，``0.5`` = 中，``0.8`` = 大。
            这个值应该基于你的领域知识或过往研究来设定，
            而不是单纯套用 Cohen 的基准。
        alpha: 显著水平（默认 0.05）。
            ``0.05`` = 接受 5% 的假阳性率（标准）。
            ``0.01`` = 更严格，需要的样本更大。
            ``0.10`` = 更宽松，适用于探索性分析。
        power: 目标统计效力（默认 None）。
            - 如果提供: 反解出所需样本量（a priori 用法）。
            - 如果 None: 计算给定 n 和 d 的实际效力（post hoc 用法）。
            ``0.80`` = Cohen 建议的标准基准。
            ``0.90`` = 需要约 50% 更多样本。
            ``0.95`` = 需要约 100% 更多样本。
        alternative: 对立假设的方向。
            ``"two-sided"`` = μ₁ ≠ μ₂（默认，最常用）。
            ``"greater"`` = μ₁ > μ₂（单侧，效应为正向）。
            ``"less"`` = μ₁ < μ₂（单侧，效应为负向）。
            单侧检验需要跨过更低的门槛 → 同样 sample size 下 power 更高。
            但必须在数据收集之前确定单侧方向——不能拿到数据后再决定。

    Returns:
        PowerResult。

    Raises:
        ValueError: 如果 n 和 power 同时指定或同时为 None。

    Example:
        # a priori: 要检出 d=0.5，power=0.80，需要多少人？
        >>> power_t_test(n=None, d=0.5, power=0.80)
        PowerResult(test_type="independent_t", n_required=64, ...)

        # post hoc: 有 50 人每组，能多大效力检出 d=0.5？
        >>> power_t_test(n=50, d=0.5, power=None)
        PowerResult(... power_achieved=0.697, ...)
    """
    if (n is None and power is None) or (n is not None and power is not None):
        raise ValueError("必须指定 n 或 power 中的一个（但不能同时指定）。")

    if d <= 0:
        raise ValueError(f"效应量 d 必须大于 0（当前 d={d}）。")

    # ── 效力计算函数 (n → power) ──
    def _compute_power(n_val: float) -> float:
        ncp_val = d * math.sqrt(n_val / 2.0)  # 非中心参数
        df_val = 2.0 * n_val - 2.0  # 自由度

        if alternative == "two-sided":
            t_crit = sp_stats.t.ppf(1.0 - alpha / 2.0, df_val)
            # power = P(T > t_crit | ncp) + P(T < -t_crit | ncp)
            p_upper = 1.0 - sp_stats.nct.cdf(t_crit, df_val, ncp_val)
            p_lower = sp_stats.nct.cdf(-t_crit, df_val, ncp_val)
            return float(p_upper + p_lower)
        elif alternative == "greater":
            t_crit = sp_stats.t.ppf(1.0 - alpha, df_val)
            return float(1.0 - sp_stats.nct.cdf(t_crit, df_val, ncp_val))
        else:  # "less"
            t_crit = sp_stats.t.ppf(alpha, df_val)
            return float(sp_stats.nct.cdf(t_crit, df_val, ncp_val))

    if n is not None:
        # post hoc: 计算效力
        ncp_val = d * math.sqrt(n / 2.0)
        power_achieved = _compute_power(n)
        return PowerResult(
            test_type="independent_t",
            n_required=None,
            power_achieved=power_achieved,
            effect_size=d,
            alpha=alpha,
            alternative=alternative,
            n_per_group=n,
            df=2 * n - 2,
            ncp=ncp_val,
        )
    else:
        # a priori: 计算样本量
        n_required = _solve_n(_compute_power, power, d, alpha, (2, 100000))
        ncp_val = d * math.sqrt(n_required / 2.0) if n_required > 0 else 0
        return PowerResult(
            test_type="independent_t",
            n_required=n_required if n_required > 0 else None,
            power_achieved=power if n_required > 0 else None,
            effect_size=d,
            alpha=alpha,
            alternative=alternative,
            n_per_group=n_required if n_required > 0 else None,
            df=2 * n_required - 2 if n_required > 0 else None,
            ncp=ncp_val,
        )


def power_t_test_paired(
    n: int | None = None,
    d: float = 0.5,
    alpha: float = 0.05,
    power: float | None = None,
    alternative: str = "two-sided",
) -> PowerResult:
    """配对 t 检验效力与样本量分析。

    配对设计比独立设计更有统计效力（同样的 n，效力更大），
    因为每个被试充当自己的对照，消除了个体间变异。
    代价是不能用于不可逆的干预（没法"撤销后重新测量"）。

    Args:
        n: 总对数（= 样本量）。如果 None 则反解。
        d: Cohen's d（配对均值差 / 差值的标准差）。
        alpha: 显著水平（默认 0.05）。
        power: 目标效力（默认 None → post hoc）。
        alternative: 对立假设方向。

    Returns:
        PowerResult。

    注意:
        - 配对 d 和独立 d 不能直接比较。配对 d 基于差值的 SD
          （通常更小），所以同样的原始效应量，配对 d 更大。
        - 配对 ncp = d × √n（比独立的 d × √(n/2) 大）。
    """
    if (n is None and power is None) or (n is not None and power is not None):
        raise ValueError("必须指定 n 或 power 中的一个（但不能同时指定）。")

    if d <= 0:
        raise ValueError(f"效应量 d 必须大于 0（当前 d={d}）。")

    def _compute_power(n_val: float) -> float:
        ncp_val = d * math.sqrt(n_val)  # 注意：配对比独立的 ncp 大
        df_val = n_val - 1.0

        if alternative == "two-sided":
            t_crit = sp_stats.t.ppf(1.0 - alpha / 2.0, df_val)
            p_upper = 1.0 - sp_stats.nct.cdf(t_crit, df_val, ncp_val)
            p_lower = sp_stats.nct.cdf(-t_crit, df_val, ncp_val)
            return float(p_upper + p_lower)
        elif alternative == "greater":
            t_crit = sp_stats.t.ppf(1.0 - alpha, df_val)
            return float(1.0 - sp_stats.nct.cdf(t_crit, df_val, ncp_val))
        else:
            t_crit = sp_stats.t.ppf(alpha, df_val)
            return float(sp_stats.nct.cdf(t_crit, df_val, ncp_val))

    if n is not None:
        ncp_val = d * math.sqrt(n)
        power_achieved = _compute_power(n)
        return PowerResult(
            test_type="paired_t",
            n_required=None,
            power_achieved=power_achieved,
            effect_size=d,
            alpha=alpha,
            alternative=alternative,
            n_per_group=n,
            df=n - 1,
            ncp=ncp_val,
        )
    else:
        n_required = _solve_n(_compute_power, power, d, alpha, (2, 100000))
        ncp_val = d * math.sqrt(n_required) if n_required > 0 else 0
        return PowerResult(
            test_type="paired_t",
            n_required=n_required if n_required > 0 else None,
            power_achieved=power if n_required > 0 else None,
            effect_size=d,
            alpha=alpha,
            alternative=alternative,
            n_per_group=n_required if n_required > 0 else None,
            df=n_required - 1 if n_required > 0 else None,
            ncp=ncp_val,
        )


# ═══════════════════════════════════════════
# 单因素 ANOVA 效力分析
# ═══════════════════════════════════════════


def power_anova(
    n: int | None = None,
    k: int = 3,
    f: float = 0.25,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """单因素 ANOVA 效力与样本量分析。

    通过非中心 F 分布 (ncf) 精确计算效力。

    Args:
        n: 每组样本量。
            - 如果提供: 计算给定 n 的效力（post hoc）。
            - 如果 None: 反解满足目标 power 的每组 n（a priori）。
        k: 组数（默认 3）。各组样本量假定相等（平衡设计）。
        f: Cohen's f 效应量。
            ``f² = η² / (1 - η²)`` 其中 η² = 因子解释的方差比例。

            ``0.10`` = 小效应（组间差异微弱，η² ≈ 0.01）
            ``0.25`` = 中效应（组间有可见差异，η² ≈ 0.06）
            ``0.40`` = 大效应（组间差异巨大，η² ≈ 0.14）

            如果你的数据中有已知的 η²，可以换算：
            f = sqrt(η² / (1 - η²))。
        alpha: 显著水平。
        power: 目标效力。

    Returns:
        PowerResult。n_required 是每组样本量，总样本量 = n_required × k。

    Example:
        # 3 组，中效应 f=0.25，power=0.80 → 每组需要多少人？
        >>> power_anova(n=None, k=3, f=0.25, power=0.80)
        PowerResult(test_type="anova", n_required=53, n_per_group=53, ...)
        # 总共需要 53 × 3 = 159 人
    """
    if (n is None and power is None) or (n is not None and power is not None):
        raise ValueError("必须指定 n 或 power 中的一个（但不能同时指定）。")

    if f <= 0:
        raise ValueError(f"效应量 f 必须大于 0（当前 f={f}）。")
    if k < 2:
        raise ValueError(f"至少需要 2 组（当前 k={k}）。")

    def _compute_power(n_per: float) -> float:
        N = n_per * k  # 总样本量
        ncp_val = f * f * N  # 非中心参数
        df1 = k - 1  # 分子自由度
        df2 = N - k  # 分母自由度
        f_crit = sp_stats.f.ppf(1.0 - alpha, df1, df2)
        return float(1.0 - sp_stats.ncf.cdf(f_crit, df1, df2, ncp_val))

    if n is not None:
        ncp_val = f * f * n * k
        df1_val = k - 1
        df2_val = n * k - k
        power_achieved = _compute_power(n)
        return PowerResult(
            test_type="anova",
            n_required=None,
            power_achieved=power_achieved,
            effect_size=f,
            alpha=alpha,
            alternative="two-sided",
            n_per_group=n,
            df=df1_val,
            df2=int(df2_val),
            ncp=ncp_val,
        )
    else:
        n_required = _solve_n(_compute_power, power, f, alpha, (2, 100000))
        if n_required > 0:
            ncp_val = f * f * n_required * k
            df1_val = k - 1
            df2_val = n_required * k - k
        else:
            ncp_val = 0
            df1_val = None
            df2_val = None
        return PowerResult(
            test_type="anova",
            n_required=n_required if n_required > 0 else None,
            power_achieved=power if n_required > 0 else None,
            effect_size=f,
            alpha=alpha,
            alternative="two-sided",
            n_per_group=n_required if n_required > 0 else None,
            df=df1_val,
            df2=int(df2_val) if df2_val is not None else None,
            ncp=ncp_val,
        )


# ═══════════════════════════════════════════
# 卡方检验效力分析
# ═══════════════════════════════════════════


def power_chi_square(
    n: int | None = None,
    w: float = 0.3,
    df: int = 1,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """卡方检验效力与样本量分析。

    通过非中心 χ² 分布 (ncx2) 精确计算效力。
    用于 Pearson 卡方独立性检验（交叉表）。

    Args:
        n: 总样本量。None 则反解。
        w: Cohen's w 效应量。
            w = sqrt( Σ (p_obs_i - p_exp_i)² / p_exp_i )
            即观测频率与期望频率的标准化的差异。

            ``0.10`` — 小效应（如性别对某弱偏好的微弱关联）
            ``0.30`` — 中效应（如专业对口与否对工作满意度）
            ``0.50`` — 大效应（如吸烟与肺癌的强关联）

            对于 2×2 表：w² = φ²（均方列联系数平方）。
        df: 自由度 = (行数-1) × (列数-1)。
            ``1`` = 2×2 表（最常见）。
            ``2`` = 2×3 或 3×2 表。
            ``4`` = 3×3 表。
            注意：自由度越大，同样的 w 需要更多样本。
        alpha: 显著水平。
        power: 目标效力。

    Returns:
        PowerResult。n_required 是总样本量。

    Example:
        # 2×2 表，中效应 w=0.3，power=0.80 → 需要多少样本？
        >>> power_chi_square(n=None, w=0.3, df=1, power=0.80)
        PowerResult(test_type="chi_square", n_required=88, ...)
    """
    if (n is None and power is None) or (n is not None and power is not None):
        raise ValueError("必须指定 n 或 power 中的一个（但不能同时指定）。")

    if w <= 0:
        raise ValueError(f"效应量 w 必须大于 0（当前 w={w}）。")

    def _compute_power(n_val: float) -> float:
        ncp_val = w * w * n_val  # 非中心参数
        chi2_crit = sp_stats.chi2.ppf(1.0 - alpha, df)
        return float(1.0 - sp_stats.ncx2.cdf(chi2_crit, df, ncp_val))

    if n is not None:
        ncp_val = w * w * n
        power_achieved = _compute_power(n)
        return PowerResult(
            test_type="chi_square",
            n_required=None,
            power_achieved=power_achieved,
            effect_size=w,
            alpha=alpha,
            alternative="two-sided",
            df=df,
            ncp=ncp_val,
        )
    else:
        n_required = _solve_n(_compute_power, power, w, alpha, (2, 100000))
        ncp_val = w * w * n_required if n_required > 0 else 0
        return PowerResult(
            test_type="chi_square",
            n_required=n_required if n_required > 0 else None,
            power_achieved=power if n_required > 0 else None,
            effect_size=w,
            alpha=alpha,
            alternative="two-sided",
            df=df,
            ncp=ncp_val,
        )


# ═══════════════════════════════════════════
# Pearson 相关效力分析
# ═══════════════════════════════════════════


def power_correlation(
    n: int | None = None,
    r: float = 0.3,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """Pearson 相关效力与样本量分析。

    基于 Fisher z 变换 + 非中心 t 分布精确计算。

    Fisher z 变换使得相关检验的问题变成 t 检验的问题：
    t = r × √(n-2) / √(1 - r²)，NCP = arctanh(r) × √(n-1)

    Args:
        n: 总样本量（成对观测数）。None 则反解。
        r: Pearson 相关系数（效应量）。
            ``0.10`` — 小（微弱但可能真实的关联）
            ``0.30`` — 中（明显关联，肉眼可见的散点图趋势）
            ``0.50`` — 大（很强关联，如前后测分数）

            r 的范围是 [-1, 1]，但效力分析只关心绝对值（用的是 |r|）。
        alpha: 显著水平。
        power: 目标效力。

    Returns:
        PowerResult。n_required 是总样本量。

    Example:
        # 检出 r=0.3，power=0.80 → 需要多少人？
        >>> power_correlation(n=None, r=0.3, power=0.80)
        PowerResult(test_type="correlation", n_required=85, ...)
    """
    if (n is None and power is None) or (n is not None and power is not None):
        raise ValueError("必须指定 n 或 power 中的一个（但不能同时指定）。")

    r_abs = abs(r)
    if r_abs <= 0 or r_abs >= 1:
        raise ValueError(f"|r| 必须在 (0, 1) 范围内（当前 |r|={r_abs}）。")

    def _compute_power(n_val: float) -> float:
        # Fisher z 变换：ncp = arctanh(ρ) × √(n-1) ≈ z × √(n-1)
        ncp_val = math.atanh(r_abs) * math.sqrt(n_val - 1.0)
        df_val = n_val - 2.0
        t_crit = sp_stats.t.ppf(1.0 - alpha / 2.0, df_val)
        p_upper = 1.0 - sp_stats.nct.cdf(t_crit, df_val, ncp_val)
        p_lower = sp_stats.nct.cdf(-t_crit, df_val, ncp_val)
        return float(p_upper + p_lower)

    if n is not None:
        ncp_val = math.atanh(r_abs) * math.sqrt(n - 1.0)
        power_achieved = _compute_power(n)
        return PowerResult(
            test_type="correlation",
            n_required=None,
            power_achieved=power_achieved,
            effect_size=r,
            alpha=alpha,
            alternative="two-sided",
            df=n - 2,
            ncp=ncp_val,
        )
    else:
        n_required = _solve_n(_compute_power, power, r_abs, alpha, (3, 100000))
        ncp_val = math.atanh(r_abs) * math.sqrt(n_required - 1.0) if n_required > 0 else 0
        return PowerResult(
            test_type="correlation",
            n_required=n_required if n_required > 0 else None,
            power_achieved=power if n_required > 0 else None,
            effect_size=r,
            alpha=alpha,
            alternative="two-sided",
            df=n_required - 2 if n_required > 0 else None,
            ncp=ncp_val,
        )


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def power_report(r: PowerResult) -> str:
    """效力分析报告文本。

    Args:
        r: PowerResult。

    Returns:
        格式化报告字符串。
    """
    test_labels = {
        "independent_t": "独立样本 t 检验",
        "paired_t": "配对 t 检验",
        "anova": "单因素 ANOVA",
        "chi_square": "卡方检验",
        "correlation": "Pearson 相关",
    }
    test_label = test_labels.get(r.test_type, r.test_type)
    alt_label = {"two-sided": "双尾", "greater": "单尾 (>),", "less": "单尾 (<)"}.get(r.alternative, r.alternative)

    lines = [
        f"{'='*55}",
        f"  统计效力分析: {test_label}",
        f"  α={r.alpha}, 对立假设={alt_label}",
        f"{'='*55}",
    ]

    if r.test_type == "anova":
        lines.append(f"  效应量 f = {r.effect_size:.3f}")
    else:
        eff_name = {"independent_t": "d", "paired_t": "d", "chi_square": "w", "correlation": "r"}.get(r.test_type, "?")
        lines.append(f"  效应量 {eff_name} = {r.effect_size:.3f}")

    if r.n_required is not None:
        lines.append(f"  所需样本量: n = {r.n_required}")
        if r.n_per_group is not None and r.test_type in ("independent_t", "anova"):
            lines.append(f"  每组: {r.n_per_group}")
        if r.test_type == "anova":
            lines.append(f"  总计 (k×n): {r.n_required * r.n_per_group if r.n_per_group else '?'}")
        lines.append(f"  非中心参数 ncp = {r.ncp:.3f}")
        if r.df is not None:
            lines.append(f"  自由度 df = {r.df}{f', df2 = {r.df2}' if r.df2 else ''}")
        label = "目标效力" if r.power_achieved is not None else ""
        if r.power_achieved is not None:
            lines.append(f"  目标效力 1-β = {r.power_achieved:.2f}")
    elif r.power_achieved is not None:
        lines.append(f"  实际效力 1-β = {r.power_achieved:.4f}")
        lines.append(f"  n = {r.n_per_group or r.df + 2 if r.df else '?'}")
        lines.append(f"  非中心参数 ncp = {r.ncp:.3f}")
        if r.df is not None:
            lines.append(f"  自由度 df = {r.df}")

    # 效力等级解读
    if r.power_achieved is not None:
        if r.power_achieved >= 0.95:
            lines.append(f"  评价: ★★★ 非常充足的效力")
        elif r.power_achieved >= 0.80:
            lines.append(f"  评价: ★★☆ 充足的效力 (≥0.80)")
        elif r.power_achieved >= 0.50:
            lines.append(f"  评价: ★☆☆ 效力不足 (建议增加样本量)")
        else:
            lines.append(f"  评价: ☆☆☆ 效力严重不足 (检出真实效应的概率不到一半)")

    lines.append(f"{'='*55}")
    return "\n".join(lines)
