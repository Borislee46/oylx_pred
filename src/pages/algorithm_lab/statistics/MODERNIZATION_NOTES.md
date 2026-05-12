# 统计模块现代化改进备忘录

记录当前实现与最新统计学最佳实践的差异，作为后续迭代的参考。不急于改。

---

## 一、已做到正确的现代选择（优于 SPSS 默认值）

| 功能 | SPSS 默认 | 当前默认 | 评判 |
|------|-----------|---------|------|
| t 检验 | Student + Levene 预检 | **Welch** (`equal_var=False`) | 正确 |
| 事后检验 | LSD / 依赖 Levene 二选一 | 提供 7 种方法 + 决策指南 | 正确 |
| 因子旋转 | Varimax | **Promax** (斜交) | 正确 |
| 相关矩阵多重比较 | 无 | **FDR-BH** | 正确 |
| 比例 CI | Wald | **Wilson Score** (推荐) | 正确 |
| OLS 标准误 | 经典 SE | **HC3 稳健 SE** | 正确 |

---

## 二、可更新的过时实践

### 2.1 两步法前置检验（最优先）

**问题：** Levene → 选 Student/Welch、Mauchly → 选是否 GG 校正。这种"先检验假定，再根据 p 值选方法"的做法已被证明会引入额外的第一类错误膨胀。

**原因：** 前置检验在小样本时效力不足（漏掉真实的假定偏离），在大样本时过度敏感（轻微偏离就 p<0.05）。模拟研究一致建议直接使用对假定偏离稳健的方法。

**建议：**

| 场景 | 当前行为 | 改为 |
|------|---------|------|
| `independent_t_test` | 已默认 Welch，但仍计算 Levene 并展示 | Levene 保留计算但降权为诊断参考，在报告中加一句"方差齐性诊断仅供参考，不应用于选择检验方法" |
| RM ANOVA | Mauchly 检验 + 分别报告未校正/GG/HF 三个 p 值 | **GG 校正 p 值作为推荐结果**置于首位。Mauchly 结果移到补充诊断区 |
| ANOVA `anova_report` | Levene 结果紧接 F 检验展示 | 同上，Levene 降权为诊断参考 |

### 2.2 因子数选择：Kaiser 准则 → 平行分析

**问题：** 特征值 > 1 准则（Kaiser 1954）是粗略经验直觉，非正式统计方法。变量数少时低估因子数，变量数多时高估。

**当前状态：** `factor_analysis()` 已有 `parallel_analysis()` 实现，但 `parallel_n_sim` 默认=0（不执行）。

**建议：** 默认启用平行分析（`parallel_n_sim=100`），报告中平行分析结果放在特征值表之前。Kaiser 准则降为辅助参考线。

### 2.3 Durbin-Watson → 补充 Ljung-Box

**问题：** DW 只能检测一阶自相关（e_t vs e_{t-1}）。对高阶自相关（季度性等）完全盲区。

**当前状态：** `regression_ols.py` 只计算 DW。`survival/time_series.py` 已有完整的 `ljung_box_test` 和 `acf_pacf`。

**建议：** 在 OLS 诊断中补充 Ljung-Box 检验（lags=1, 4, 或 `min(10, n/5)`），至少标记 DW 的局限性。代码复用已有的 `ljung_box_test`。

### 2.4 Cronbach's α → 补充 McDonald's ω

**问题：** α 假设 tau-equivalence（所有题项载荷完全相等），真实量表几乎从不满足。当假设不成立时，α 是信度的**下限**而非点估计——它系统地低估真实信度。

**当前状态：** `reliability.py` 只实现了 Cronbach's α。`factor_analysis.py` 已有 EFA 完整的载荷矩阵和独特方差。

**建议：** 在 `reliability.py` 中增加 `mcdonalds_omega()`：
```
ω = (Σ λ_i)² / [(Σ λ_i)² + Σ θ_i]
```
其中 λ_i 是标准化载荷，θ_i 是独特方差（= 1 - communality）。分层次 ω（hierarchical omega）可作为后续补充。

### 2.5 事后检验：Games-Howell 应为方差不齐时的首选

**当前状态：** `POSTHOC_GUIDE` 有说明，但函数签名没有"根据 Levene 自动选择"的路由。

**建议：** 在 `posthoc()` 路由函数中增加一个 `auto` 模式：如果各组 n 差异 > 2× 或方差差异大 → 自动选 Games-Howell，否则 Tukey。或者不改代码，在 UI 层实现智能推荐。

### 2.6 线性趋势检验：仅支持等距有序因子

**当前状态：** `linear_trend_test()` 使用线性对比系数（-1, 0, 1 等间距），假定分组是有序且等距的。

**建议：** 补充多项式趋势检验（二次、三次），参数化方式与 SPSS ONEWAY 的 Polynomial Contrasts 对齐。

---

## 三、可补充的现代方法（当前缺失）

### 3.1 贝叶斯因子体系

**已有：** t 检验有 JZS BF10（`_jzs_bf10` 近似）。

**缺失：** ANOVA 贝叶斯因子（Rouder et al. 2012）、相关贝叶斯因子（Jeffreys 1961）、列联表贝叶斯因子。这些是 null hypothesis 检验的关键补充，帮助区分"证据缺乏"和"缺乏证据"。

### 3.2 非参数效应量

**已有：** t 检验有 Cohen's d / Hedges' g / Glass's Δ。M-W 已有 rank-biserial r。

**缺失：** Kruskal-Wallis 的 η²_H（已计算但未暴露为独立函数）、Friedman 的 Kendall's W（已计算但未暴露）、ANOVA 的 Cliff's Delta。效果量应统一到每个检验的 result dataclass 中。

### 3.3 缺失值插补

**当前状态：** 全模块使用 pairwise/listwise 删除。统计模块没有插补功能。

**建议：** 作为独立子模块补充多重插补（Multiple Imputation / MICE）的接口。这可以后续引入。

### 3.4 稳健 ANOVA（现代替代）

**已实现：** Welch ANOVA（`welch_anova`）。

**缺失：** 稳健 ANOVA 的现代替代：
- **WRS2 风格**：修剪均值 ANOVA + 百分位 Bootstrap 事后检验
- **ART（Aligned Rank Transform）**：适用于多因素非参数 ANOVA
- **稳健混合模型**：`rlmer` 风格

Welch 解决方差问题，但不能解决离群值问题。当数据同时存在方差不等 + 离群值时，需要稳健方法。

### 3.5 Bootstrap CI 统一化

**当前状态：** `bootstrap.py` 有完整的百分位 + BCa。`ratio_stats.py` 有独立的 Bootstrap CI 实现。OLS、ANOVA、EFA 等没有 Bootstrap 选项。

**建议：** 给每个主分析函数加 `bootstrap_ci: bool = False` 参数，统一复用 `bootstrap.py` 的逻辑。

### 3.6 置信区间 + 效应量的完整覆盖

以下位置缺少 CI 和/或效应量：

| 函数 | 缺失内容 |
|------|---------|
| `kruskal_wallis` | η²_H CI |
| `friedman` | Kendall's W CI |
| `mann_whitney` | Hodges-Lehmann 估计（中位数差的 CI） |
| `cox_ph` | Harrell's C-index（一致性） |

### 3.7 高维数据的正则化回归

**缺失：** LASSO / Ridge / Elastic Net。当 k（变量数）> n 或接近 n 时，OLS 完全不可用。

**建议：** 在 `inference/` 或 `ml/` 中增加 `regularized_regression()`，使用坐标下降法（可用 `scipy.optimize` 实现或依赖 `scikit-learn` 的轻量封装）。

### 3.8 中介与调节分析

**缺失：** 社会科学中最常用的两种分析路径：
- **中介（Mediation）：** X → M → Y，Bootstrap 间接效应检验
- **调节（Moderation）：** X × M 交互项 + 简单斜率检验

**建议：** 新增 `inference/mediation.py`，实现 Baron-Kenny 三步法 + Bootstrap 间接效应 CI + Sobel 检验。

---

## 四、架构与工程改进

### 4.1 根 `__init__.py` 缺少子包导入

`statistics/__init__.py` 只有 `__all__` 字符串列表，没有 `from . import descriptive` 等。

**影响：** `from statistics import descriptive` 报 ImportError。

**修复：** 在根 `__init__.py` 中添加 6 行显式导入。

### 4.2 所有文件超过 300 行（CLAUDE.md Rule 1）

见正文审查。拆分策略：
- **文档字符串** → 模块 README（精简代码主体）
- **事后检验方法集合**（7 种 × ~80 行）→ `anova_posthoc.py`
- **回归诊断集合**（VIF/DW/BP/Cook's D/Leverage）→ `ols_diagnostics.py`

### 4.3 缺少 README.md（CLAUDE.md Rule 3）

### 4.4 缺少 Streamlit UI 连接

整个 statistics 模块在 algorithm_lab 页面中不可交互使用。建议设计和搭建一个"统计分析"标签页。

---

## 五、优先级排序

| 优先级 | 项目 | 工作量 | 类型 |
|--------|------|--------|------|
| P0 | 根 `__init__.py` 添加导入 | 1 行 | 修复 |
| P0 | 添加 README.md | 中 | 合规 |
| P1 | 默认启用平行分析 | 1 行 | 更新 |
| P1 | RM ANOVA 报告默认推荐 GG 校正 p | ~5 行 | 更新 |
| P1 | 两步法检验降权提示（Levene/Mauchly） | ~10 行 | 更新 |
| P1 | 各函数补充效应量 c.f. 3.2 | 中 | 补充 |
| P2 | McDonald's ω | 中 | 补充 |
| P2 | OLS 补充 Ljung-Box | 小 | 补充 |
| P2 | Bootstrap CI 统一化 | 中 | 重构 |
| P3 | 贝叶斯因子扩展（ANOVA/相关/列联表） | 大 | 补充 |
| P3 | 中介/调节分析 | 大 | 补充 |
| P3 | 正则化回归 | 中 | 补充 |
| P3 | 文件拆分（>300 行） | 大 | 重构 |
| P4 | 稳健 ANOVA（WRS2 风格） | 大 | 补充 |
| P4 | 多重插补（MICE） | 大 | 补充 |

---

*记录日期：2026-05-01*
