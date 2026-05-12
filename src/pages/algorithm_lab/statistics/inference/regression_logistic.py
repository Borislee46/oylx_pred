"""二元 Logistic 回归 — 全参数实现

复刻 SPSS 二元 Logistic 回归过程 (Binary Logistic Regression):
- Newton-Raphson MLE 迭代
- Wald z 检验 + OR 置信区间
- 多种伪 R² (McFadden, Nagelkerke)
- Hosmer-Lemeshow 拟合优度 + ROC AUC + 分类表

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【Logistic 回归解决什么问题】

因变量是二元的（0/1，是/否，录取/不录取）。你不能用 OLS 因为：
1. 预测值可能超出 [0,1] → 不能当概率用
2. 残差不是正态分布（y 只有 0 和 1）
3. 方差不恒定（方差取决于 p，p 不同方差不同）

Logistic 回归的解决方案：把概率变换到 (-∞, +∞) 再建模。

【核心变换：logit 连接函数】

    logit(p) = ln(p / (1-p))
    逆变换：p = 1 / (1 + e^(-XB)) = sigmoid(XB)

logit(0.5) = 0：概率一半时 logit 为零
logit(0.9) ≈ 2.20：高概率对应正数
logit(0.1) ≈ -2.20：低概率对应负数

logit 把 [0,1] 的概率压缩到整个实数轴，使得线性方程 XB 可以安全地产生任意值。

直观理解：logit(p) 是"对数几率"（log-odds），是对事件发生和不发生之比取对数。
logit(0.8) = ln(0.8/0.2) = ln(4) ≈ 1.39 → "发生的几率是不发生的 4 倍"。

【MLE vs OLS — 完全不同的估计方法】

OLS（线性回归）有一条闭式解：β = (X'X)⁻¹X'y，一步到位。

Logistic 回归没有闭式解！因为 p = sigmoid(XB) 是非线性的。
必须用迭代算法找到最大化"似然函数"的 β 值。

    Likelihood = Π p_i^(y_i) * (1-p_i)^(1-y_i)
    对数似然 = Σ [y_i * ln(p_i) + (1-y_i) * ln(1-p_i)]

Newton-Raphson 迭代：
    β_new = β_old + (-H)^(-1) * g
    g (梯度) = X'(y-p)：当前预测与真实值的"总误差方向"
    H (Hessian) = -X'·W·X，其中 W = diag(p(1-p))：加权信息矩阵
    (-H)^(-1) 的 diag 直接给出系数的标准误！

    每步迭代都在当前 β 附近用二阶泰勒展开近似对数似然，
    然后跳向这个近似的最大值。反复操作直到收敛。

    初始值：所有 β = 0（不含截距），截距 = logit(发生率)。

【OR (Odds Ratio) — 核心解读】

    OR = exp(B)

OR 衡量的是"X 每增加 1 单位，事件发生的几率乘以多少"。

    注意：OR 说的是"几率"的倍数，不是概率的倍数！
    OR=2  ≠ "概率翻倍"
    OR=2  = "几率翻倍"

    p=0.1 → odds=0.11，OR=2 → odds=0.22 → p≈0.18（增加了 8%，不是翻倍）
    p=0.5 → odds=1.0，OR=2 → odds=2.0  → p≈0.67（增加了 17%）

OR 的基线依赖：同样的 OR，基础概率不同，概率变化也完全不同。
报告时总是要说明"对于某个特定的基准概率"的变化。

    OR=1   → X 无影响
    OR>1   → 正向关联（X 增加 → 事件更可能发生）
    OR<1   → 负向关联（X 增加 → 事件更不可能发生）
    OR=1.5 → "X 每增加 1，几率增加 50%"

OR 的 95% CI = exp(B ± 1.96 × SE)
如果 CI 跨过 1.0 → 统计不显著。

【伪 R² — 不能和 OLS R² 混为一谈】

OLS 的 R² = "被解释的方差比例"。Logistic 没有这个，但有近似：

McFadden R² = 1 - LL_model / LL_null
    典型范围 0.05~0.4。0.2 以上就算不错。0.4 以上算优秀。
    直觉：模型解释了"多少比例的空模型不确定性"。
    不能和 OLS R² 比较！McFadden 0.2 不等于 OLS 0.2。

Nagelkerke R² = (1 - exp(-2(LL_null - LL)/n)) / (1 - exp(2*LL_null/n))
    把 Cox-Snell R² 缩放到 [0,1] 区间内。
    上限为 1，但实际很少到 1。
    比 McFadden 容易理解，但与 OLS R² 仍然不同。

两种 R² 都只在模型比较中有用（模型 A vs 模型 B），
不能单独说"R²=0.3 的模型好不好"。

【Hosmer-Lemeshow 检验 — 校准度诊断】

把样本按预测概率从小到大分成 g 组（默认 10 组 = 十分位）。
每组计算：观察事件数 O_i vs 期望事件数 E_i。
检验统计量 = Σ (O_i - E_i)² / (E_i(1 - E_i/n_i))，近似 χ²(g-2)。

    H0: 模型拟合良好（观察值接近预测值）
    p > 0.05 → 没有证据表明拟合不良
    p < 0.05 → 模型预测概率与实际事件率不一致 → 校准有问题

已知局限：
- 对分组方式敏感（g 的取值、切点规则）
- 大样本时（n>5000）几乎总是显著
- 小样本时（n<100）几乎总是不显著
- p > 0.05 不等于"模型好用"——它只检验校准度，不检验区分度

【为什么不用 accuracy 衡量 Logistic 模型】

类别不平衡时（如 95% 不录取，5% 录取），一个模型预测"全都不录取"，
accuracy=95%，但毫无价值。

应同时报告：
- AUC（不依赖阈值，整体区分能力）：0.7=可接受，0.8=良好，0.9=优秀
- Sensitivity（真实录取中被正确识别的比例）：业务相关的"召回"
- Specificity（真实不录取中被正确识别的比例）

【模型拟合指标一览】

- AIC = -2*LL + 2*k（k=参数个数）：越小越好，惩罚复杂模型
- BIC = -2*LL + k*ln(n)：比 AIC 更严厉惩罚复杂模型（大样本时）
- 两个模型比较：AIC 或 BIC 更低的模型更好
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
class LogisticCoefficient:
    """Logistic 回归单个系数"""

    name: str
    b: float  # 未标准化回归系数 (log-odds 尺度)
    se: float  # 标准误
    z: float  # Wald z 统计量
    p: float  # p 值 (双侧)
    or_value: float  # 优势比 = exp(b)
    or_ci_95: tuple[float, float]  # OR 的 95% CI
    beta: float = float("nan")  # 标准化系数 (X 变化 1 个标准差时 logit 的变化)


@dataclass
class LogisticResult:
    """Logistic 回归完整结果"""

    n: int  # 有效样本量
    p: int  # 参数个数 (含截距)
    n_events: int  # 事件发生数 (y=1)
    n_nonevents: int  # 事件不发生数 (y=0)
    coefficients: list[LogisticCoefficient]  # 自变量系数
    intercept: LogisticCoefficient | None  # 截距项
    log_likelihood: float  # 模型对数似然
    log_likelihood_null: float  # 空模型对数似然 (仅截距)
    aic: float  # AIC = -2LL + 2k
    bic: float  # BIC = -2LL + k*ln(n)
    mcfadden_r2: float  # McFadden 伪 R²
    nagelkerke_r2: float  # Nagelkerke 伪 R²
    hosmer_lemeshow_chi2: float = float("nan")  # HL χ²
    hosmer_lemeshow_p: float = float("nan")  # HL p 值
    auc: float = float("nan")  # ROC AUC
    classification_table: dict | None = None  # 分类表 {tp, fp, tn, fn, accuracy, sensitivity, specificity}
    predicted_probs: np.ndarray | None = None  # 预测概率
    converged: bool = True  # 是否收敛
    iterations: int = 0  # 迭代次数


# ═══════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid 函数，含数值截断防止溢出。

    Args:
        z: 线性预测值 XB。

    Returns:
        p = 1/(1+exp(-z))，clip 到 [1e-15, 1-1e-15] 防止 log(0)。
    """
    z_clipped = np.clip(z, -500, 500)  # exp(500) 已经溢出 float64
    p = 1.0 / (1.0 + np.exp(-z_clipped))
    return np.clip(p, 1e-15, 1.0 - 1e-15)


def _log_likelihood(y: np.ndarray, p: np.ndarray) -> float:
    """模型对数似然 = Σ[y*ln(p) + (1-y)*ln(1-p)]。

    Args:
        y: 观察到的 0/1 向量。
        p: 预测概率 (sigmoid(XB))，应已 clip 防止 ln(0)。
    """
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _log_likelihood_null(y: np.ndarray) -> float:
    """空模型（仅截距）对数似然。即用 y 的总体均值作为所有预测。

    Args:
        y: 观察到的 0/1 向量。
    """
    pi = float(np.mean(y))
    if pi <= 0 or pi >= 1:
        return 0.0  # 完美分离时
    n = len(y)
    return float(n * (pi * np.log(pi) + (1.0 - pi) * np.log(1.0 - pi)))


# ═══════════════════════════════════════════
# 核心 Logistic 回归
# ═══════════════════════════════════════════


def logistic_regression(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    *,
    ci_level: float = 0.95,
    max_iter: int = 100,
    tol: float = 1e-8,
    compute_diagnostics: bool = True,
) -> LogisticResult:
    """二元 Logistic 回归 (Newton-Raphson MLE)。

    通过 Newton-Raphson 迭代最大化对数似然函数，
    自动检测收敛，并计算所有标准诊断指标。

    Args:
        y: 因变量，二分类 (0/1 或布尔值)。可含缺失值，按行删除。
        X: 自变量矩阵 (n × k)。支持 DataFrame（自动提取列名）和 ndarray。
        ci_level: OR 置信区间的置信水平（默认 0.95）。
        max_iter: Newton-Raphson 最大迭代次数（默认 100）。
        tol: 收敛容差。梯度范数和参数变化同时小于 tol 时停止。
        compute_diagnostics: 是否计算 HL / AUC / 分类表（默认 True）。

    Returns:
        LogisticResult。
    """
    # ── 数据清洗 ──
    y_arr = np.asarray(y, dtype=np.float64)
    if isinstance(X, pd.DataFrame):
        var_names = list(X.columns)
        X_arr = X.to_numpy(dtype=np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        var_names = [f"X{i+1}" for i in range(X_arr.shape[1])]

    mask = (~np.isnan(y_arr)) & (~np.isnan(X_arr).any(axis=1))
    y_arr, X_arr = y_arr[mask], X_arr[mask]
    n, k = X_arr.shape

    if n == 0:
        raise ValueError("清洗后无有效样本，请检查缺失值。")
    if k == 0:
        raise ValueError("至少需要一个自变量（X 列为空）。")

    # 添加截距列
    X_design = np.column_stack([np.ones(n, dtype=np.float64), X_arr])
    p = k + 1  # 参数总数（含截距）
    n_events = int(np.sum(y_arr))
    n_nonevents = n - n_events

    if n_events == 0:
        raise ValueError("因变量全为 0，Logistic 回归无法估计。")
    if n_events == n:
        raise ValueError("因变量全为 1，Logistic 回归无法估计。")

    # ── 初始化：所有 beta=0，截距=logit(发生率) ──
    # logit(发生率) = ln(事件比例 / 非事件比例)
    pi_initial = np.mean(y_arr)
    if pi_initial <= 0 or pi_initial >= 1:
        pi_initial = 0.5
    beta = np.zeros(p, dtype=np.float64)
    beta[0] = float(np.log(pi_initial / (1.0 - pi_initial)))  # 截距初始化

    # ── Newton-Raphson 迭代 ──
    converged = False
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        eta = X_design @ beta
        p_hat = _sigmoid(eta)

        # 梯度 g = X'(y-p)
        residual = y_arr - p_hat
        gradient = X_design.T @ residual

        # Hessian H = -X'·W·X，W = diag(p(1-p))
        w = p_hat * (1.0 - p_hat)
        WX = X_design * w[:, np.newaxis]  # 逐行加权
        hessian = -WX.T @ X_design

        # 参数更新：beta_new = beta - H^-1 * g
        try:
            hessian_inv = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # Hessian 奇异 → 数据有问题（共线、完美分离）
            hessian_inv = np.linalg.pinv(hessian)  # 伪逆兜底

        delta = -hessian_inv @ gradient
        beta = beta + delta

        # 收敛判断：梯度范数 和 参数变化 都小于 tol
        grad_norm = float(np.max(np.abs(gradient)))
        delta_norm = float(np.max(np.abs(delta)))
        if grad_norm < tol and delta_norm < tol:
            converged = True
            break

    # ── 标准误 = sqrt(diag(-H^-1)) ──
    se_all = np.sqrt(np.maximum(np.diag(-hessian_inv), 0.0))

    # ── Wald z 检验 ──
    z_all = beta / np.where(se_all > 0, se_all, 1.0)
    p_all = 2.0 * (1.0 - sp_stats.norm.cdf(np.abs(z_all)))

    # ── OR + CI ──
    z_crit = sp_stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0)
    or_all = np.exp(beta)
    or_ci_low = np.exp(beta - z_crit * se_all)
    or_ci_high = np.exp(beta + z_crit * se_all)

    # ── 标准化系数 Beta（X 变化 1 个标准差时 logit 的变化） ──
    # logit 分布的方差 = π²/3，所以 Beta = b * sd(X) / (π/√3)
    x_stds = np.std(X_arr, axis=0, ddof=1)
    logit_sd = math.pi / math.sqrt(3.0)  # ≈ 1.8138
    beta_std = np.zeros(p, dtype=np.float64)
    beta_std[0] = float("nan")  # 截距不做标准化
    for j in range(k):
        if x_stds[j] > 0:
            beta_std[j + 1] = float(beta[j + 1] * x_stds[j] / logit_sd)

    # ── 模型拟合指标 ──
    # 最终预测概率
    eta_final = X_design @ beta
    p_final = _sigmoid(eta_final)

    ll_model = _log_likelihood(y_arr, p_final)
    ll_null = _log_likelihood_null(y_arr)

    aic = -2.0 * ll_model + 2.0 * p
    bic = -2.0 * ll_model + p * np.log(n)

    # 伪 R²
    if ll_null != 0:
        mcfadden_r2 = float(1.0 - ll_model / ll_null)
    else:
        mcfadden_r2 = 0.0

    # Cox-Snell R² → Nagelkerke 缩放
    # Cox-Snell: 1 - exp(2*(LL_null - LL_model)/n)
    # LL_null < LL_model ≤ 0，所以 LL_null - LL_model < 0，exp(负数) < 1
    cox_snell_r2 = 1.0 - np.exp(2.0 * (ll_null - ll_model) / n)
    # Nagelkerke max: 1 - exp(2*LL_null/n)，范围 (0, 1)
    nagelkerke_max = 1.0 - np.exp(2.0 * ll_null / n)
    nagelkerke_r2 = float(cox_snell_r2 / nagelkerke_max if nagelkerke_max > 0 else 0.0)
    # 数值截断 [0, 1]
    nagelkerke_r2 = max(0.0, min(1.0, nagelkerke_r2))

    # ── 系数组装 ──
    intercept = LogisticCoefficient(
        name="(Intercept)",
        b=float(beta[0]),
        se=float(se_all[0]),
        z=float(z_all[0]),
        p=float(p_all[0]),
        or_value=float(or_all[0]),
        or_ci_95=(float(or_ci_low[0]), float(or_ci_high[0])),
        beta=float("nan"),  # 截距无标准化
    )

    coefficients = []
    for j in range(k):
        coefficients.append(LogisticCoefficient(
            name=var_names[j],
            b=float(beta[j + 1]),
            se=float(se_all[j + 1]),
            z=float(z_all[j + 1]),
            p=float(p_all[j + 1]),
            or_value=float(or_all[j + 1]),
            or_ci_95=(float(or_ci_low[j + 1]), float(or_ci_high[j + 1])),
            beta=float(beta_std[j + 1]) if not np.isnan(beta_std[j + 1]) else float("nan"),
        ))

    # ── 诊断 ──
    hl_chi2 = float("nan")
    hl_p = float("nan")
    auc = float("nan")
    class_table = None

    if compute_diagnostics:
        hl_chi2, hl_p = _hosmer_lemeshow(y_arr, p_final)
        auc = _roc_auc(y_arr, p_final)
        class_table = _classification_table(y_arr, p_final)

    return LogisticResult(
        n=n,
        p=p,
        n_events=n_events,
        n_nonevents=n_nonevents,
        coefficients=coefficients,
        intercept=intercept,
        log_likelihood=ll_model,
        log_likelihood_null=ll_null,
        aic=aic,
        bic=bic,
        mcfadden_r2=mcfadden_r2,
        nagelkerke_r2=nagelkerke_r2,
        hosmer_lemeshow_chi2=hl_chi2,
        hosmer_lemeshow_p=hl_p,
        auc=auc,
        classification_table=class_table,
        predicted_probs=p_final,
        converged=converged,
        iterations=iterations,
    )


# ═══════════════════════════════════════════
# 诊断辅助
# ═══════════════════════════════════════════


def _hosmer_lemeshow(y: np.ndarray, p: np.ndarray, g: int = 10) -> tuple[float, float]:
    """Hosmer-Lemeshow 拟合优度检验。

    将样本按预测概率从小到大分成 g 组，每组计算观察与期望事件数，
    用 χ² 检验评估偏差程度。

    Args:
        y: 观察到的 0/1 向量。
        p: 模型预测概率。
        g: 分组数（默认 10，即十分位）。

    Returns:
        (chi2, p_value)。

    注意:
        H0 = "模型校准良好"。p < 0.05 仅表示有统计学证据表明
        模型预测偏离实际——这在大样本时几乎总是显著的。
        只做参考，不能单独作为模型取舍的依据。
    """
    n = len(y)
    order = np.argsort(p)
    y_sorted = y[order]
    p_sorted = p[order]

    chi2 = 0.0
    expected_events = np.zeros(g)
    observed_events = np.zeros(g)

    for grp in range(g):
        start = int(grp * n / g)
        end = int((grp + 1) * n / g)
        if start >= end:
            continue
        y_grp = y_sorted[start:end]
        p_grp = p_sorted[start:end]
        n_g = len(y_grp)
        o_g = float(np.sum(y_grp))
        e_g = float(np.sum(p_grp))
        expected_events[grp] = e_g
        observed_events[grp] = o_g
        if e_g > 0 and n_g - e_g > 0:
            chi2 += (o_g - e_g) ** 2 / (e_g * (1.0 - e_g / n_g))

    # 自由度 = g - 2（因为估计了两个参数：截距和斜率）
    df_hl = max(g - 2, 1)
    # 合并预期事件数过小的组
    small_cells = (expected_events < 5.0) | ((n / g - expected_events) < 5.0)
    if np.sum(small_cells) > 1:
        df_hl = max(min(g - 2, int(np.sum(~small_cells)) - 1), 1)

    if chi2 <= 0:
        return (0.0, 1.0)
    p_val = float(1.0 - sp_stats.chi2.cdf(chi2, df_hl))
    return (float(chi2), p_val)


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC（梯形积分法）。

    不依赖第三方库（如 sklearn），直接用 numpy 实现。

    Args:
        y: 观察到的 0/1 向量。
        p: 模型预测概率。

    Returns:
        AUC 值 [0, 1]。0.5 = 随机猜测，1 = 完美区分。
    """
    order = np.argsort(p)[::-1]  # 从高到低排序
    y_sorted = y[order]

    # 累积 TP 和 FP
    n_pos = int(np.sum(y_sorted))
    n_neg = int(len(y_sorted) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 1.0

    tpr = np.cumsum(y_sorted) / n_pos  # True Positive Rate = Sensitivity
    fpr = np.cumsum(1 - y_sorted) / n_neg  # False Positive Rate = 1 - Specificity

    # 梯形积分
    # 避免 np.trapz (NumPy 2.0+ 已移除), 手动梯形积分
    auc_val = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return auc_val


def _classification_table(y: np.ndarray, p: np.ndarray, cutoff: float = 0.5) -> dict:
    """分类表（混淆矩阵 + 派生指标）。

    Args:
        y: 观察到的 0/1 向量。
        p: 模型预测概率。
        cutoff: 分类阈值（默认 0.5。高于此值预测为 1）。

    Returns:
        dict with keys: tp, fp, tn, fn, total, accuracy, sensitivity, specificity,
                        ppv (precision), npv, f1。

    注意:
        在类别不平衡的场景下，accuracy 可能严重误导。
        始终结合 sensitivity（检出真正例的能力）和 specificity（排除真负例的能力）综合判断。
    """
    y_pred = (p >= cutoff).astype(int)
    tp = int(np.sum((y_pred == 1) & (y == 1)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    tn = int(np.sum((y_pred == 0) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")  # Positive Predictive Value = Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    f1 = 2.0 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else float("nan")

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
    }


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════


def logistic_regression_report(r: LogisticResult) -> str:
    """Logistic 回归报告文本。

    Args:
        r: LogisticResult。

    Returns:
        格式化报告字符串。
    """
    lines = [
        f"{'='*60}",
        f"  二元 Logistic 回归",
        f"  n={r.n}, 事件={r.n_events}, 非事件={r.n_nonevents}, 参数={r.p}",
        f"  收敛: {'是' if r.converged else '否'} (迭代 {r.iterations} 次)",
        f"{'='*60}",
        "",
        f"  {'变量':<20} {'B':>10} {'SE':>10} {'z':>8} {'p':>8} {'OR':>8} {'95% CI OR':>18} {'Beta':>8}",
        f"  {'-'*92}",
    ]

    # 截距行
    ci = r.intercept
    lines.append(
        f"  {ci.name:<20} {ci.b:>10.4f} {ci.se:>10.4f} {ci.z:>8.3f} {ci.p:>8.4f} "
        f"{ci.or_value:>8.3f} {' ':>18} {'':>8}"
    )

    # 系数行
    for coef in r.coefficients:
        ci_str = f"[{coef.or_ci_95[0]:.3f}, {coef.or_ci_95[1]:.3f}]"
        beta_str = f"{coef.beta:>8.3f}" if not math.isnan(coef.beta) else f"{'':>8}"
        lines.append(
            f"  {coef.name:<20} {coef.b:>10.4f} {coef.se:>10.4f} {coef.z:>8.3f} {coef.p:>8.4f} "
            f"{coef.or_value:>8.3f} {ci_str:>18} {beta_str}"
        )

    lines.append("")
    lines.append(f"  {'─'*60}")
    lines.append(f"  模型拟合")
    lines.append(f"  {'─'*60}")
    lines.append(f"  Log-Likelihood:      {r.log_likelihood:.4f}")
    lines.append(f"  Log-Likelihood (空): {r.log_likelihood_null:.4f}")
    lines.append(f"  McFadden R²:         {r.mcfadden_r2:.4f}")
    lines.append(f"  Nagelkerke R²:       {r.nagelkerke_r2:.4f}")
    lines.append(f"  AIC:                 {r.aic:.2f}")
    lines.append(f"  BIC:                 {r.bic:.2f}")

    if not math.isnan(r.hosmer_lemeshow_chi2):
        lines.append(f"  {'─'*60}")
        lines.append(f"  诊断")
        lines.append(f"  {'─'*60}")
        lines.append(f"  Hosmer-Lemeshow χ²={r.hosmer_lemeshow_chi2:.3f}, p={r.hosmer_lemeshow_p:.4f}")
        lines.append(f"  ROC AUC:             {r.auc:.4f}")

    if r.classification_table is not None:
        ct = r.classification_table
        lines.append(f"")
        lines.append(f"  分类表 (cutoff=0.5):")
        lines.append(f"    实际\\预测   预测=1    预测=0")
        lines.append(f"    实际=1      {ct['tp']:>6}     {ct['fn']:>6}")
        lines.append(f"    实际=0      {ct['fp']:>6}     {ct['tn']:>6}")
        lines.append(f"  Accuracy={ct['accuracy']:.3f}, "
                     f"Sensitivity={ct['sensitivity']:.3f}, "
                     f"Specificity={ct['specificity']:.3f}")
        lines.append(f"  PPV={ct['ppv']:.3f}, NPV={ct['npv']:.3f}, F1={ct['f1']:.3f}")

    lines.append(f"{'='*60}")
    return "\n".join(lines)
