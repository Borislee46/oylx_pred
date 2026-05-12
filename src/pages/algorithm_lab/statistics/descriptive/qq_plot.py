"""Q-Q 图 (Quantile-Quantile Plot) — 全参数实现

复刻 SPSS PPLOT /TYPE=Q-Q 过程。
比较样本分位数与理论分布分位数，对分布尾部的偏离最敏感。

与 P-P 图的差异: Q-Q 图比较分位数 (尾部敏感), P-P 图比较 CDF (中部敏感)。

══════════════════════════════════════════════════════════════════════
核心直觉与诊断解读
══════════════════════════════════════════════════════════════════════

【Q-Q 图 vs P-P 图 — 到底看哪个】

Q-Q 图: X 轴 = 理论分位数 ("正态分布下你应该在哪"), Y 轴 = 样本分位数
          ("你实际在哪")。对尾部敏感 —— 如果你的分布比正态重尾/轻尾,
          两端会很明显地偏离对角线。

P-P 图: X 轴 = 理论累积概率, Y 轴 = 经验累积概率。
         对中部敏感 —— 尾部 0.01 和 0.001 之间只差 0.009, 肉眼不可见。

两者互补: P-P 看中部拟合好不好, Q-Q 看尾部，正态性检验看统计检验。

【Q-Q 图的四种典型模式 — 怎么"读"一张 Q-Q 图】

U 型 (下凸, 两头都有残差为正):
  数据右偏 (正偏态)。高端的样本值比正态期望值大, 低端样本值也偏大。
  → 右偏分布的特征, 建议对数变换或平方根变换。

倒 U 型 (上凸, 两头都有残差为负):
  数据左偏 (负偏态)。低端样本值比正态期望值小很多。
  → 左偏分布的特征, 建议平方变换。

S 型 (一端上翘一端下垂, 中段平坦):
  重尾分布。数据在两端比正态有更多极端值。
  t 分布 (df 小) 是典型的重尾。
  → 使用稳健方法, 删除离群值, 或用非参数检验。

反 S 型 (一端下垂一端上翘):
  轻尾分布。数据比正态更"收敛", 极端值更少。
  均匀分布或截尾正态是典型的轻尾。
  → 可能还好, 参数方法对轻尾不像对重尾那么敏感。

W 型 (两端交叉, 中段有弧度):
  双峰分布。数据可能来自两个不同的组混合在一起。
  → 考虑分层分析。

【参考线拟合的 4 种方法】

"s" (IQR 拟合, SPSS 默认): 过 (Q1_theoretical, Q1_sample) 和
  (Q3_theoretical, Q3_sample) 的直线。对尾部不敏感, 稳健。

"r" (OLS 回归): 样本分位数 ~ 理论分位数。受尾部离群值影响。

"q" (四分位数过原点): IQR 拟合, 但强制截距 = 0。

"45" (y=x): 参考线完全由理论分布决定。如果数据标准化的方式不同,
  这个线可能毫无意义。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .pp_plot import (
    PROPORTION_FORMULAS,
    SUPPORTED_DISTRIBUTIONS,
    _blom_ranks,
    _theoretical_cdf,
)


@dataclass
class QQData:
    """Q-Q 图数据"""

    theoretical_quantiles: np.ndarray
    sample_quantiles: np.ndarray
    sorted_data: np.ndarray
    n: int
    proportion_method: str
    dist_name: str
    dist_params: dict
    slope: float  # 参考线斜率
    intercept: float  # 参考线截距

    @property
    def detrended_residuals(self) -> np.ndarray:
        """去趋势 Q-Q 残差 = 样本分位数 - 参考线预测值"""
        predicted = self.slope * self.theoretical_quantiles + self.intercept
        return self.sample_quantiles - predicted

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "理论分位数": self.theoretical_quantiles,
                "样本分位数": self.sample_quantiles,
                "去趋势残差": self.detrended_residuals,
                "排序数据": self.sorted_data,
            }
        )


def qq_plot(
    data: np.ndarray,
    dist: str = "normal",
    proportion_method: str = "blom",
    dist_params: dict | None = None,
    df: int | None = None,
    line: str = "s",
) -> QQData:
    """生成 Q-Q 图数据 (复刻 SPSS Q-Q 图)。

    Args:
        data: 一维连续数值数组。
        dist: 理论分布 ``"normal"`` / ``"uniform"`` / ``"exponential"``
              / ``"lognormal"`` / ``"gamma"`` / ``"t"`` / ``"chi2"``。
        proportion_method: 比例估算公式 ``"blom"`` (默认) / ``"rankit"``
                           / ``"tukey"`` / ``"van_der_waerden"``。
        dist_params: 手动指定理论分布参数。None=由样本估计。
        df: t/卡方分布的自由度。
        line: 参考线拟合方式 ``"s"`` (标准化, IQR 拟合) /
              ``"r"`` (回归拟合) / ``"q"`` (四分位数过原点) /
              ``"45"`` (45° 线, y=x)。

    Returns:
        QQData 含理论分位数、样本分位数、参考线参数、去趋势残差。
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        raise ValueError("数据为空 (全为缺失)")

    sorted_data = np.sort(arr)

    # 位置 (proportion) → 理论分位数
    formula_fn = PROPORTION_FORMULAS.get(proportion_method, _blom_ranks)
    positions = formula_fn(n)

    # 理论分位数
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

    theoretical_q = _theoretical_quantiles(positions, dist, params, df)

    # 参考线拟合
    slope, intercept = _fit_reference_line(theoretical_q, sorted_data, line)

    return QQData(
        theoretical_quantiles=theoretical_q,
        sample_quantiles=sorted_data,
        sorted_data=sorted_data,
        n=n,
        proportion_method=proportion_method,
        dist_name=dist,
        dist_params=params,
        slope=float(slope),
        intercept=float(intercept),
    )


def _theoretical_quantiles(positions: np.ndarray, dist: str, params: dict, df: int | None = None) -> np.ndarray:
    """将累积概率 positions 转为理论分位数 (PPF / 逆 CDF)。"""
    dist_lower = dist.lower()
    # 夹紧到 (0, 1) 避免 inf
    p = np.clip(positions, 1e-10, 1 - 1e-10)

    if dist_lower == "normal":
        loc = params.get("loc", 0)
        scale = params.get("scale", 1)
        return sp_stats.norm.ppf(p, loc=loc, scale=max(scale, 1e-10))
    elif dist_lower == "uniform":
        loc = params.get("loc", 0)
        scale = params.get("scale", 1)
        return sp_stats.uniform.ppf(p, loc=loc, scale=max(scale, 1e-10))
    elif dist_lower == "exponential":
        scale = params.get("scale", 1)
        return sp_stats.expon.ppf(p, scale=max(scale, 1e-10))
    elif dist_lower == "lognormal":
        s = params.get("s", 1)
        loc = params.get("loc", 0)
        scale = params.get("scale", 1)
        return sp_stats.lognorm.ppf(p, s=max(s, 1e-10), loc=loc, scale=max(scale, 1e-10))
    elif dist_lower == "gamma":
        a = params.get("a", 1)
        scale = params.get("scale", 1)
        return sp_stats.gamma.ppf(p, a=max(a, 0.01), scale=max(scale, 1e-10))
    elif dist_lower == "t":
        d = params.get("df", df or 1)
        return sp_stats.t.ppf(p, df=max(d, 1))
    elif dist_lower == "chi2":
        d = params.get("df", df or 1)
        return sp_stats.chi2.ppf(p, df=max(d, 1))
    else:
        raise ValueError(f"不支持的理论分布: {dist}")


def _fit_reference_line(
    theoretical: np.ndarray, sample: np.ndarray, method: str = "s"
) -> tuple[float, float]:
    """拟合 Q-Q 图参考线。

    - ``"s"``: IQR 标准化 fit (SPSS 默认) → 过 (Q1, P25) 和 (Q3, P75)
    - ``"r"``: OLS 回归 → sample ~ theoretical
    - ``"q"``: 四分位数过原点 → 过 (0, interquartile mean)
    - ``"45"``: y = x
    """
    if method == "45":
        return 1.0, 0.0

    if method == "q":
        # 过原点, 斜率为 interquartile 均值比
        q1_t, q3_t = np.percentile(theoretical, [25, 75])
        q1_s, q3_s = np.percentile(sample, [25, 75])
        denom = q3_t - q1_t
        slope = (q3_s - q1_s) / denom if denom != 0 else 1.0
        return slope, 0.0

    if method == "r":
        # OLS 回归
        slope, intercept, _, _, _ = sp_stats.linregress(theoretical, sample)
        return float(slope), float(intercept)

    # "s" — SPSS 默认: IQR fit
    q1_t, q3_t = np.percentile(theoretical, [25, 75])
    q1_s, q3_s = np.percentile(sample, [25, 75])
    denom = q3_t - q1_t
    if denom != 0:
        slope = (q3_s - q1_s) / denom
    else:
        slope = 1.0
    intercept = q1_s - slope * q1_t
    return float(slope), float(intercept)


def qq_plot_diagnose(result: QQData) -> str:
    """根据 Q-Q 图数据自动生成诊断文本。

    前提: X轴=理论分位数, Y轴=样本分位数 (SPSS 默认)。
    """
    residuals = result.detrended_residuals
    n = result.n
    abs_resid = np.abs(residuals)

    # 尾部残差
    tail_start = residuals[: max(3, n // 8)]
    tail_end = residuals[-max(3, n // 8) :]
    mid = residuals[n // 4 : 3 * n // 4]

    # U 型 (右偏): 两端残差都为正
    ends_both_positive = np.mean(tail_start) > 0 and np.mean(tail_end) > 0
    # 倒 U 型 (左偏): 两端残差都为负
    ends_both_negative = np.mean(tail_start) < 0 and np.mean(tail_end) < 0
    # S 型 (重尾): 一端正一端负, 中部平坦
    s_shape = (np.mean(tail_start) > 0 > np.mean(tail_end)) or (
        np.mean(tail_start) < 0 < np.mean(tail_end)
    )
    # 反 S 型 (轻尾): 两端趋向交叉
    anti_s = np.mean(np.abs(residuals[: n // 8])) < np.mean(np.abs(mid))

    max_abs_residual = float(np.max(abs_resid))
    parts = []

    if max_abs_residual < 0.5:
        parts.append("数据点基本沿对角参考线分布, 正态性假设合理。")
    elif ends_both_positive:
        parts.append("图形呈 'U 型' (下凸), 提示**右偏** (正偏态)。高端有长尾, 可考虑对数或平方根变换。")
    elif ends_both_negative:
        parts.append("图形呈 '倒 U 型' (上凸), 提示**左偏** (负偏态)。低端有长尾, 可考虑平方或指数变换。")
    elif s_shape:
        parts.append("图形呈 'S 型', 提示**重尾** (峰度 > 0)。极端值比正态分布多, 建议稳健方法或非参数检验。")
    elif anti_s:
        parts.append("图形呈 '反 S 型', 提示**轻尾** (峰度 < 0)。数据分布比正态更集中。")
    else:
        parts.append("图形整体尚可, 部分区域存在波动。")

    parts.append(f"最大去趋势残差 = {max_abs_residual:.4f}  (n = {n}, 参考线拟合: slope={result.slope:.3f})")

    # 去趋势 Q-Q 残差分析
    if max_abs_residual > 1.0:
        parts.append(
            "去趋势 Q-Q 图显示系统性模式, 建议结合 Shapiro-Wilk 或 K-S 检验做正式正态性检查。"
        )

    return "\n".join(parts)
