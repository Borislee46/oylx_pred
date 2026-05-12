"""频率分析 (Frequencies) — 全参数实现

复刻 SPSS FREQUENCIES 过程 + 扩展功能:
- 分类变量: 频数表 (N, %, Valid %, Cumulative %), 条形图, 饼图
- 连续变量: 集中趋势 + 离散程度 + 分位数 + 分布形态 (偏度/峰度 + 标准误)
- 缺失值: 系统缺失 vs 用户自定义缺失, 有效百分比分母自动排除

══════════════════════════════════════════════════════════════════════
统计概念速查：描述统计量意味着什么
══════════════════════════════════════════════════════════════════════

【集中趋势 — 数据的"中心"在哪】

均值 (Mean): 所有值的算术平均。对离群值敏感。适合对称分布。
中位数 (Median): 排序后正中间的值。对离群值不敏感。偏态分布时
                 比均值更能代表"典型值"。
众数 (Mode): 出现最多的值。适合分类变量或多峰分布。
5%截尾均值: 砍掉首尾各 5% 极端值后的均值。兼顾了稳健性和信息利用。

均值 vs 中位数的差值方向揭示偏态:
  均值 > 中位数 → 右偏 (少数极大值拉高了均值)
  均值 < 中位数 → 左偏 (少数极小值拉低了均值)

【离散程度 — 数据"散得有多开"】

极差 (Range): max - min。极度受离群值影响。仅用于初步扫一眼。
IQR (四分位距): Q3 - Q1。中间 50% 数据的范围。稳健（不受离群值影响）。
方差 (Variance): 各值与均值的平方差的均值。(n-1) 分母是无偏估计。
标准差 (SD): sqrt(方差)。和原始数据同单位，最常用的离散指标。
标准误 (Standard Error of Mean): SD / sqrt(n)。这不是描述样本离散度的！
  它是"如果我们重复抽样很多次，每次算一个均值，这些均值的标准差"。
  反映的是均值估计的精确度，n 越大 SE 越小。在报告中别把 SD 和 SE 混淆。

CV (变异系数): SD / Mean × 100%。比较不同量纲变量的相对离散度。

【分布形态 — 数据"长什么样"】

偏度 (Skewness):
  > 0 → 右偏/正偏 (尾巴在右边, 大多数值偏左, 少数极大值)。
        收入数据几乎总是右偏。
  = 0 → 对称 (正态)
  < 0 → 左偏/负偏 (尾巴在左边)。考试成绩可能左偏（有地板效应）。

  经验法则: |偏度| > 1  × SE_skewness → 明显偏态
            |偏度| > 2  → 严重偏态, 需要考虑变换或非参数方法

  这里计算的是 Fisher-Pearson 标准化矩系数 (SPSS 使用的公式),
  与 scipy.stats.skew(bias=False) 一致。

峰度 (Kurtosis) — 注意！这里输出的是 Excess Kurtosis:
  这是 SPSS 的默认设置，定义为 峰度 - 3。
  正态分布的 Excess Kurtosis = 0。

  > 0 → 尖峰/重尾 (Leptokurtic): 尾部比正态分布更厚, 极端值更多。
        金融收益率通常如此。t 分布是重尾的。
  = 0 → 正态峰度 (Mesokurtic)
  < 0 → 平峰/轻尾 (Platykurtic): 数据比正态分布更"均匀"地分布。
        均匀分布是平峰的。

  经验法则: |峰度| > 2  × SE_kurtosis → 明显峰度偏离

SES (Standard Error of Skewness): ≈ sqrt(6/n)  — 仅取决于样本量
SEK (Standard Error of Kurtosis): ≈ sqrt(24/n) — 仅取决于样本量

  这两个标准误用于判断偏度/峰度是否"显著偏离 0"。
  随着 n 增大, SE 变小, 更容易检测到显著偏离。
  大样本下(>300), 正态性检验几乎总是显著 → 此时应看 Q-Q 图 + 实际大小,
  而不是 p 值。

【百分位数与 CI】

百分位数: 第 p 个百分位 = 有 p% 的数据小于等于这个值。
  P25 = Q1, P50 = 中位数, P75 = Q3
  百分位数是非参数的 — 不做正态假设。

均值 95% CI: 使用 t 分布 (n-1 df)。
  解读: "我们有 95% 的信心, 总体均值落在 [lower, upper] 之间。"
  不是 "95% 的数据落在这个区间" — 那是预测区间, 完全不同。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass
class FreqTable:
    """单变量频数分布表"""

    values: list[str]
    counts: list[int]
    percent: list[float]  # 总百分比 (含缺失)
    valid_percent: list[float | None]  # 有效百分比 (排除缺失)
    cumulative_percent: list[float | None]  # 累积有效百分比
    n_total: int
    n_valid: int
    n_missing: int

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "值": self.values,
                "频数": self.counts,
                "百分比": [f"{v:.1f}%" for v in self.percent],
                "有效百分比": [f"{v:.1f}%" if v is not None else "" for v in self.valid_percent],
                "累积百分比": [
                    f"{v:.1f}%" if v is not None else "" for v in self.cumulative_percent
                ],
            }
        )


@dataclass
class DescriptiveStats:
    """连续变量描述统计"""

    n: int
    mean: float
    std_error_mean: float
    median: float
    mode: float | str
    std: float
    variance: float
    skewness: float
    std_error_skewness: float
    kurtosis: float  # excess kurtosis (SPSS default)
    std_error_kurtosis: float
    range_: float
    min_: float
    max_: float
    percentiles: dict[str, float] = field(default_factory=dict)
    ci_mean_95: tuple[float, float] | None = None


# ═══════════════════════════════════════════
# 频数表
# ═══════════════════════════════════════════


def frequency_table(
    data: pd.Series,
    *,
    user_missing: list | set | None = None,
    sort_by: str = "value_asc",
    bins: int | list[float] | None = None,
) -> FreqTable:
    """生成频数分布表 (复刻 SPSS FREQUENCIES)。

    Args:
        data: 单变量序列。
        user_missing: 用户自定义缺失值集合 (如 {99, 999})。
        sort_by: ``"value_asc"`` (按值升序), ``"value_desc"`` (按值降序),
                 ``"count_asc"`` (按频数升序), ``"count_desc"`` (按频数降序)。
        bins: 连续变量分组。int=等宽分箱数, list=自定义区间边界。

    Returns:
        FreqTable 含完整频数/百分比/累积百分比。
    """
    series = data.copy()

    # 标记缺失
    is_system_missing = series.isna()
    if user_missing:
        _um = set(user_missing)
        is_user_missing = series.isin(_um) & ~is_system_missing
    else:
        is_user_missing = pd.Series(False, index=series.index)

    is_any_missing = is_system_missing | is_user_missing

    # 分箱 (连续变量)
    if bins is not None:
        if isinstance(bins, int):
            series = pd.cut(series, bins=bins, right=False)
        else:
            series = pd.cut(series, bins=bins, right=False)

    valid_series = series[~is_any_missing]
    counts = series.value_counts(dropna=False)
    n_total = len(data)
    n_valid = len(valid_series)
    n_missing = n_total - n_valid

    # 排序
    if sort_by == "count_desc":
        counts = counts.sort_values(ascending=False)
    elif sort_by == "count_asc":
        counts = counts.sort_values(ascending=True)
    elif sort_by == "value_desc":
        counts = counts.sort_index(ascending=False)
    else:  # value_asc
        counts = counts.sort_index(ascending=True)

    values: list[str] = []
    freqs: list[int] = []
    pct: list[float] = []
    valid_pct: list[float | None] = []
    cum_pct: list[float | None] = []

    cum = 0.0
    for idx, cnt in counts.items():
        if isinstance(idx, float) and math.isnan(idx):
            label = "系统缺失"
        elif user_missing and idx in user_missing:
            label = f"{idx} (用户缺失)"
        else:
            label = str(idx)

        values.append(label)
        freqs.append(cnt)
        pct.append(cnt / n_total * 100)

        is_missing = (
            (isinstance(idx, float) and math.isnan(idx))
            or (user_missing and idx in user_missing)
        )
        if is_missing:
            valid_pct.append(None)
            cum_pct.append(None)
        else:
            vp = cnt / n_valid * 100 if n_valid > 0 else 0.0
            cum += vp
            valid_pct.append(vp)
            cum_pct.append(cum)

    return FreqTable(
        values=values,
        counts=freqs,
        percent=pct,
        valid_percent=valid_pct,
        cumulative_percent=cum_pct,
        n_total=n_total,
        n_valid=n_valid,
        n_missing=n_missing,
    )


# ═══════════════════════════════════════════
# 描述统计
# ═══════════════════════════════════════════


def _std_error_skewness(n: int) -> float:
    """偏度标准误 SES ≈ sqrt(6 / n) (大样本近似)"""
    if n <= 1:
        return float("nan")
    return math.sqrt(6.0 / n)


def _std_error_kurtosis(n: int) -> float:
    """峰度标准误 SEK ≈ sqrt(24 / n) (大样本近似)"""
    if n <= 1:
        return float("nan")
    return math.sqrt(24.0 / n)


def _excess_kurtosis(x: np.ndarray) -> float:
    """过度峰度 (excess kurtosis)，SPSS 默认。正态分布期望=0。

    公式等同于 scipy.stats.kurtosis(a, fisher=True, bias=False)。
    前半部分计算的是原始峰度 E[(Z⁴)] 的无偏估计,
    后半部分减去 3 = 正态分布的峰度, 使得正态的 excess = 0。
    用 ddof=0 (总体 std) 做标准化是因为样本峰度公式的分母
    已经做了无偏修正, SPSS 要求在标准化时保持一致。
    """
    n = len(x)
    if n <= 3:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    z = (x - x.mean()) / x.std(ddof=0)
    k = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(z**4)
    k -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float(k)


def _skewness(x: np.ndarray) -> float:
    """偏度 (与 SPSS / scipy.stats.skew 一致)。"""
    n = len(x)
    if n <= 1:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    z = (x - x.mean()) / x.std(ddof=0)
    return float(np.sum(z**3) * n / ((n - 1) * (n - 2)))


def _mode(series: pd.Series) -> float | str:
    """众数，多个时返回逗号分隔。"""
    mode_vals = series.mode()
    if len(mode_vals) == 0:
        return "N/A"
    if len(mode_vals) == 1:
        return float(mode_vals.iloc[0])
    return ", ".join(str(v) for v in mode_vals.values)


def descriptive_stats(
    data: pd.Series,
    *,
    user_missing: list | set | None = None,
    percentiles: list[float] | None = None,
    ci_level: float = 0.95,
    midpoints: list[float] | None = None,
) -> DescriptiveStats:
    """计算连续变量的完整描述统计 (复刻 SPSS Frequencies → Statistics 子对话框)。

    Args:
        data: 连续变量序列。
        user_missing: 用户自定义缺失值。
        percentiles: 自定义百分位列表 (如 [5, 25, 50, 75, 95])。
        ci_level: 均值的置信区间水平 (默认 .95 → 95% CI)。
        midpoints: 组距编码的组中值列表，若提供则用组中值推算统计量。

    Returns:
        DescriptiveStats 含全部统计量。
    """
    series = data.copy()

    # 排除缺失
    mask = series.isna()
    if user_missing:
        mask = mask | series.isin(user_missing)
    valid = series[~mask].astype(float)
    n = len(valid)

    if midpoints is not None:
        # 组距编码: 用组中值 × 频数 推算均值等
        mp = np.asarray(midpoints, dtype=np.float64)
        vc = valid.value_counts().sort_index()
        counts = vc.values
        vals = mp[: len(counts)]
        total = counts.sum()
        mean = np.average(vals, weights=counts)
    else:
        vals = valid.values.astype(np.float64)
        total = n
        mean = float(np.mean(vals)) if n > 0 else float("nan")

    arr = vals

    # 集中趋势
    median = float(np.median(arr)) if n > 0 else float("nan")
    mode = _mode(valid) if n > 0 else "N/A"

    # 离散程度
    std = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
    variance = std**2 if n > 1 else float("nan")
    se_mean = std / math.sqrt(n) if n > 1 else float("nan")
    min_ = float(np.min(arr)) if n > 0 else float("nan")
    max_ = float(np.max(arr)) if n > 0 else float("nan")
    range_ = max_ - min_

    # 分布形态
    sk = _skewness(arr) if n > 1 else float("nan")
    ses = _std_error_skewness(n) if n > 0 else float("nan")
    kt = _excess_kurtosis(arr) if n > 3 else float("nan")
    sek = _std_error_kurtosis(n) if n > 0 else float("nan")

    # 自定义百分位
    pct_dict: dict[str, float] = {}
    if percentiles:
        for p in percentiles:
            pct_dict[f"P{p}"] = float(np.percentile(arr, p)) if n > 0 else float("nan")

    # 均值 CI (t 分布)
    ci: tuple[float, float] | None = None
    if n > 1 and std > 0:
        alpha = 1 - ci_level
        t_crit = sp_stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * se_mean
        ci = (mean - margin, mean + margin)

    return DescriptiveStats(
        n=n,
        mean=mean,
        std_error_mean=se_mean,
        median=median,
        mode=mode,
        std=std,
        variance=variance,
        skewness=sk,
        std_error_skewness=ses,
        kurtosis=kt,
        std_error_kurtosis=sek,
        range_=range_,
        min_=min_,
        max_=max_,
        percentiles=pct_dict,
        ci_mean_95=ci,
    )


# ═══════════════════════════════════════════
# 联合接口 (分类变量自动裁决)
# ═══════════════════════════════════════════


def analyze(
    data: pd.Series,
    *,
    user_missing: list | set | None = None,
    sort_by: str = "value_asc",
    bins: int | list[float] | None = None,
    percentiles: list[float] | None = None,
    ci_level: float = 0.95,
) -> tuple[FreqTable, DescriptiveStats | None]:
    """一键频率分析: 自动返回频数表 + 连续变量描述统计。

    若数据为数值型且唯一值较多 (>=10), 自动计算描述统计。
    否则只返回频数表, DescriptiveStats 为 None。
    """
    ft = frequency_table(data, user_missing=user_missing, sort_by=sort_by, bins=bins)

    # 自动裁决是否计算描述统计
    numeric = pd.api.types.is_numeric_dtype(data)
    n_unique = data.nunique()
    if numeric and n_unique >= 10:
        ds = descriptive_stats(
            data,
            user_missing=user_missing,
            percentiles=percentiles or [5, 25, 50, 75, 95],
            ci_level=ci_level,
        )
        return ft, ds

    return ft, None
