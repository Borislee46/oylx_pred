"""P-P 图 (Probability-Probability Plot) — 全参数实现

复刻 SPSS PPLOT /TYPE=P-P 过程。
比较经验累积分布概率 (ECDF) 与理论累积分布概率 (CDF)，
对分布中部的拟合偏差最敏感。

与 Q-Q 图的差异: P-P 图比较 CDF (中部敏感), Q-Q 图比较分位数 (尾部敏感)。

══════════════════════════════════════════════════════════════════════
核心直觉
══════════════════════════════════════════════════════════════════════

【P-P 图到底在看什么】

把数据排序后, 每个数据点 i 的"经验累积概率"近似为 i/n。
然后问: 如果这个数据真的来自某个理论分布, 它的理论累积概率应该是多少？
把 (理论概率, 经验概率) 画成散点图。

如果数据完全符合理论分布 → 所有点落在对角线上。
偏离对角线 = 拟合偏差。

【为什么 P-P 对中部敏感, Q-Q 对尾部敏感】

P-P 图把 X 轴和 Y 轴都限制在 [0, 1] 之间（概率空间）。
在尾部, 0.01 和 0.001 之间的差异在图上只有 0.009 的距离 —— 几乎看不见。
但 0.49 和 0.51 之间的差异同样是 0.02, 非常显眼。
而 Q-Q 图在原始数据空间, 尾部的巨大差值一目了然。

实践中两者互补: P-P 看中部, Q-Q 看尾部。

【四种比例估算公式】

问题: 排序后第 i 个数据点的"经验概率"应该算成多少？
如果 n=10, 最大值算 1.0 (100%) 还是 10/11 ≈ 0.909？

Blom (SPSS 默认): (i - 3/8) / (n + 1/4)
  模拟研究表明正态假设下最优。i=1 不会得 0, i=n 不会得 1 (避免 inf)。

Rankit: (i - 1/2) / n
  最自然, 但尾部偏保守。

Tukey: (i - 1/3) / (n + 1/3)
  Blom 的简化版, 类似效果。

van der Waerden: i / (n + 1)
  最简单的修正, 尾部略保守。

一般情况下选 Blom 即可, 它是对的。

【去趋势 P-P 图 (Detrended)】

即 empirical_prob - theoretical_prob 对水平线 y=0 的偏离。
如果数据完美拟合, Y 轴残差在 0 上下随机波动。
S 型弯曲 → 方差/峰度问题; 弓形 → 偏度/均值问题。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ═══════════════════════════════════════════
# 比例估算公式 (Proportion Estimation)
# ═══════════════════════════════════════════


def _blom_ranks(n: int) -> np.ndarray:
    """Blom 公式: (i - 3/8) / (n + 1/4) — SPSS 默认"""
    i = np.arange(1, n + 1)
    return (i - 0.375) / (n + 0.25)


def _rankit_ranks(n: int) -> np.ndarray:
    """Rankit 公式: (i - 1/2) / n"""
    i = np.arange(1, n + 1)
    return (i - 0.5) / n


def _tukey_ranks(n: int) -> np.ndarray:
    """Tukey 公式: (i - 1/3) / (n + 1/3)"""
    i = np.arange(1, n + 1)
    return (i - 1.0 / 3.0) / (n + 1.0 / 3.0)


def _van_der_waerden_ranks(n: int) -> np.ndarray:
    """van der Waerden 公式: i / (n + 1)"""
    i = np.arange(1, n + 1)
    return i / (n + 1.0)


PROPORTION_FORMULAS = {
    "blom": _blom_ranks,
    "rankit": _rankit_ranks,
    "tukey": _tukey_ranks,
    "van_der_waerden": _van_der_waerden_ranks,
}

SUPPORTED_DISTRIBUTIONS = {
    "normal": "正态分布",
    "uniform": "均匀分布",
    "exponential": "指数分布",
    "lognormal": "对数正态分布",
    "gamma": "伽马分布",
    "t": "t 分布",
    "chi2": "卡方分布",
}


@dataclass
class PPData:
    """P-P 图数据"""

    empirical_prob: np.ndarray
    theoretical_prob: np.ndarray
    sorted_data: np.ndarray
    n: int
    proportion_method: str
    dist_name: str
    dist_params: dict

    @property
    def detrended_residuals(self) -> np.ndarray:
        """去趋势残差 = 经验概率 - 理论概率"""
        return self.empirical_prob - self.theoretical_prob

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "观测值": self.sorted_data,
                "经验累积概率": self.empirical_prob,
                "理论累积概率": self.theoretical_prob,
                "残差": self.detrended_residuals,
            }
        )


def pp_plot(
    data: np.ndarray,
    dist: str = "normal",
    proportion_method: str = "blom",
    dist_params: dict | None = None,
    df: int | None = None,
) -> PPData:
    """生成 P-P 图数据。

    Args:
        data: 一维连续数值数组。
        dist: 理论分布名称 ``"normal"`` / ``"uniform"`` / ``"exponential"``
              / ``"lognormal"`` / ``"gamma"`` / ``"t"`` / ``"chi2"``。
        proportion_method: 经验概率估算公式 ``"blom"`` (默认) / ``"rankit"``
                           / ``"tukey"`` / ``"van_der_waerden"``。
        dist_params: 手动指定理论分布参数。None=由样本估计。
                     正态: {"loc": μ, "scale": σ}
        df: t/卡方分布的自由度。

    Returns:
        PPData 含经验概率、理论概率、去趋势残差。
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        raise ValueError("数据为空（全为缺失）")

    # 排序
    sorted_data = np.sort(arr)

    # 经验累积概率 (使用指定的比例估算公式)
    formula_fn = PROPORTION_FORMULAS.get(proportion_method, _blom_ranks)
    ecdf = formula_fn(n)

    # 理论累积概率
    if dist_params is not None:
        params = dist_params
    else:
        params = {"loc": float(np.mean(arr)), "scale": float(np.std(arr, ddof=1))}
        if dist in ("t",):
            params.pop("loc", None)
            params.pop("scale", None)
            params["df"] = df or (n - 1)
        elif dist in ("chi2",):
            params.pop("loc", None)
            params.pop("scale", None)
            params["df"] = df or n

    tcdf = _theoretical_cdf(sorted_data, dist, params, df)

    return PPData(
        empirical_prob=ecdf,
        theoretical_prob=tcdf,
        sorted_data=sorted_data,
        n=n,
        proportion_method=proportion_method,
        dist_name=dist,
        dist_params=params,
    )


def _theoretical_cdf(x: np.ndarray, dist: str, params: dict, df: int | None = None) -> np.ndarray:
    """计算理论 CDF"""
    dist_lower = dist.lower()
    if dist_lower == "normal":
        loc = params.get("loc", 0)
        scale = params.get("scale", 1)
        return sp_stats.norm.cdf(x, loc=loc, scale=max(scale, 1e-10))
    elif dist_lower == "uniform":
        loc = params.get("loc", x.min())
        scale = params.get("scale", x.max() - x.min())
        return sp_stats.uniform.cdf(x, loc=loc, scale=max(scale, 1e-10))
    elif dist_lower == "exponential":
        scale = params.get("scale", x.mean())
        return sp_stats.expon.cdf(x, scale=max(scale, 1e-10))
    elif dist_lower == "lognormal":
        scale = params.get("scale", np.std(np.log(x[x > 0]))) if (x > 0).any() else 1.0
        # 使用 scipy shape/s
        s = params.get("s", scale)
        loc = params.get("loc", 0)
        scale_param = params.get("scale", np.exp(np.mean(np.log(x[x > 0])))) if (x > 0).any() else 1.0
        return sp_stats.lognorm.cdf(x, s=max(s, 1e-10), loc=loc, scale=max(scale_param, 1e-10))
    elif dist_lower == "gamma":
        # moment estimation
        m = np.mean(x)
        v = np.var(x, ddof=1)
        shape = params.get("a", (m**2) / max(v, 1e-10))
        scale = params.get("scale", v / max(m, 1e-10))
        return sp_stats.gamma.cdf(x, a=max(shape, 0.01), scale=max(scale, 1e-10))
    elif dist_lower == "t":
        d = params.get("df", df or len(x) - 1)
        return sp_stats.t.cdf(x, df=max(d, 1))
    elif dist_lower == "chi2":
        d = params.get("df", df or len(x))
        return sp_stats.chi2.cdf(x, df=max(d, 1))
    else:
        raise ValueError(f"不支持的理论分布: {dist}")


def pp_plot_diagnose(result: PPData) -> str:
    """根据 P-P 图数据自动生成诊断文本。

    检测: S 型弯曲 (方差/峰度问题) → 弓形偏离 (偏度/均值问题) """
    residuals = result.detrended_residuals
    n = result.n
    abs_resid = np.abs(residuals)

    # 中部残差符号模式检测
    mid_start = n // 4
    mid_end = 3 * n // 4
    mid_residuals = residuals[mid_start:mid_end]
    tail_start_residuals = residuals[:mid_start]
    tail_end_residuals = residuals[mid_end:]

    # S 型: 中部正残差, 两端负残差 (或反向)
    mid_positive = np.mean(mid_residuals > 0)
    ends_negative = np.mean(np.concatenate([tail_start_residuals, tail_end_residuals]) < 0)
    s_pattern = mid_positive > 0.65 and ends_negative > 0.65

    # 弓形: 残差整体偏向一侧
    bowed = np.abs(np.mean(residuals)) > 0.5 * np.std(residuals) / np.sqrt(n)

    max_abs_residual = float(np.max(abs_resid))

    parts = []
    if max_abs_residual < 0.03:
        parts.append("数据点紧密贴合对角参考线, 分布拟合良好。")
    elif s_pattern:
        parts.append("图形呈现 'S 型' 或反 'S 型' 弯曲, 提示方差或峰度与理论分布存在系统性偏差。")
    elif bowed:
        parts.append("图形整体偏向一侧 (弓形), 提示数据偏度或均值与理论分布存在偏差。")
    else:
        parts.append("图形整体趋势尚可, 但存在局部波动, 建议结合 Q-Q 图与正态性检验综合判断。")

    parts.append(f"最大绝对残差 = {max_abs_residual:.4f} (n = {n})")
    return "\n".join(parts)
