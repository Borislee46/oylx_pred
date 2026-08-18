---
source: analysis-library/cards/00-descriptive/explore-spss-plots-dialog.md
improve_fingerprint: 0cce30fb4be60266a924434fa36c0810431097f0a43ef797b1663165a1ff13b1
prompt_digest: bbf83924da02e3b0
cached: false
---

# 探索（Explore）—「图」子对话框

- **slug**: explore-spss-plots
- **aliases**: EXAMINE、探索性分析
- **菜单**：**分析 → 描述统计 → 探索**（Analyze → Descriptive Statistics → Explore）

## 1. 回答什么问题

本模块旨在因子（分组变量）的不同水平下，对连续型因变量进行**探索性数据分析（EDA）与图形诊断**，帮助研究者在正式推断前评估数据分布特征及统计假设的合理性：
- **位置与离群值（箱线图）**：展示数据的中位数、四分位距（IQR）。SPSS 默认使用 Tukey 铰链标准：超出 $1.5 \times \text{IQR}$ 的点定义为**异常值（Outliers，图中用圆圈 $\circ$ 标出）**，超出 $3 \times \text{IQR}$ 的点定义为**极端值（Extreme Values，图中用星号 $*$ 标出）**。
- **分布形状（直方图/茎叶图）**：直观判断偏态、双峰及数据聚集趋势。
- **正态性（含检验的正态图）**：通过正态 Q-Q 图联合统计检验，判断样本是否来自正态总体。
- **方差齐性（分布-水平图与 Levene 检验）**：评估各组方差是否随均值（或中位数）水平的变化而变化，并在方差非齐时提供**方差稳定化变换（数据变换）**的参数建议。

## 2. 不适用 / 易滥用

- **大样本下的“唯 p 值论”**：正态性检验对大样本极其敏感，哪怕存在微不足道的非正态偏离，也会导致 $p < .05$。大样本下必须结合 **Q-Q 图**与实际测量尺度进行综合判断。
- **小样本下的低功效**：在极小样本（如每组 $n < 10$）下，正态性检验与箱线图离群值判定均极其不稳定，此时应优先依赖无分布假设的方法或稳健统计量。
- **UI 翻译误导（极易踩坑）**：在中文版 SPSS 的「分布-水平图」选项中，**“幂（Power）”指的是方差稳定化变换的幂指数（如 Box-Cox 变换的 $\lambda$）**，**绝不是**指统计检验的“功效（Statistical Power, $1-\beta$）”。
- **用探索图替代正式推断**：探索图仅提供描述性线索。若要进行正式的组间差异检验或多重比较，仍需依赖 ANOVA、GLM 或对应的非参数设计。

## 3. 数据与设计前提

- **变量类型**：因变量需为连续变量（区间/等比尺度）；因子（分组变量）需为分类变量。
- **独立性**：默认观测值之间相互独立（若为重复测量数据，此处的方差齐性检验结果不直接适用）。
- **缺失值处理策略**：
  - **按列表排除（Listwise，默认）**：只要任何指定变量有缺失，该个案即被全量剔除。
  - **按对排除（Pairwise）**：仅在计算当前特定的统计量或图表时剔除缺失值。选择哪种方式会直接影响有效样本量 $n$ 及后续检验的自由度。

## 4. 模型或检验统计量（含自由度逻辑）

勾选相应选项后，SPSS 输出的核心检验统计量如下：

- **正态性检验**：
  - **Shapiro-Wilk (S-W) 检验**：基于次序统计量，通常在样本量 $n < 50$ 时表现出较好的检验功效。
  - **Kolmogorov-Smirnov (K-S) 检验（经 Lilliefors 显著性修正）**：基于经验累积分布函数，通常用于较大样本。
- **Levene 检验（方差齐性检验）**：
  - 本质是对“组内残差绝对值”进行的单因素 ANOVA。
  - 统计量服从 $F(k-1, N-k)$ 分布（$k$ 为组数，$N$ 为总样本量）。
  - SPSS 默认提供基于均值、基于中位数（即 Brown-Forsythe 检验，对偏态更稳健）、基于截尾均值的多种计算版本。

## 5. 假设与诊断

| 统计假设 | 检查方式 / 诊断工具 | 违背时的典型后果与应对 |
| :--- | :--- | :--- |
| **总体正态性**<br>(Parametric 检验前提) | Q-Q 图、S-W 检验、K-S 检验 | 小样本下后续 t/ANOVA 检验的名义第一类错误率偏离。应对：数据变换或改用非参数检验。 |
| **方差齐性**<br>(Homoscedasticity) | Levene 检验、分布-水平图 (Spread-Level) | ANOVA 等方法的标准误估计产生偏差，导致多重比较结果不可靠。应对：使用 Welch ANOVA 或数据变换。 |

*注：**Spread-Level 评估方差齐性的原理**是将各组的 $\ln(\text{IQR})$ 对 $\ln(\text{Median})$ 进行线性回归拟合。如果斜率为 0，说明离散程度不随水平变化（方差齐）；如果不为 0，该直线的斜率将用于推算建议的数据变换“幂估计值”。*

## 6. 效应量与区间

- **描述性分布特征**：本模块侧重于提供**中位数、IQR、极差、偏度（Skewness）与峰度（Kurtosis）**。
- **均值的 95% 置信区间 (CI)**：在默认输出的描述表格中提供。需要注意，该 CI 计算依赖正态假设，若数据严重偏态，直接报告中位数及 IQR 更为严谨。
- 若需报告标准化的效应量（如 Cohen's $d$, $\eta^2$），应转至对应的推断性统计菜单（如 Compare Means 或 GLM）。

## 7. 与相关方法的取舍

- **仅需简单全局描述**：若无分组变量，仅需均值、频数或标准差，使用 **分析 → 描述统计 → 频率/描述 (Frequencies / Descriptives)** 更为轻量。
- **正态性单独深入**：若重点在于联合其他分布（如指数、均匀分布）进行分布拟合检验，**分析 → 描述统计 → Q-Q / P-P 图** 提供了更丰富的概率图选项。
- **正式的组间比较**：本模块的均值比较仅停留在可视化层面，正式推断需移步 **单因素 ANOVA (One-Way ANOVA)** 或 **非参数检验 (Nonparametric Tests)**。

## 8. 实现速查

### SPSS Syntax 示例
以下语法实现了按分组变量 (`GroupVar`) 对连续变量 (`DepVar`) 进行全面的图形与统计探索（包含描述统计、带离群值标记的箱线图、正态性检验及稳健的 Levene 检验）：

```spss
EXAMINE VARIABLES=DepVar BY GroupVar
  /PLOT=BOXPLOT STEMLEAF NPPLOT SPREADLEVEL
  /STATISTICS=DESCRIPTIVES
  /CINTERVAL 95
  /MISSING=LISTWISE
  /NOTOTAL.
```
*(注：`NPPLOT` 对应“含检验的正态图”，`SPREADLEVEL` 对应含 Levene 检验的分布-水平图。)*

### R 与 Python 核心替代函数
- **R**: 
  - `boxplot(Y ~ X)` 
  - `shapiro.test(Y)` 
  - `car::leveneTest(Y ~ X, center=median)`
  - `car::spreadLevelPlot(lm(Y ~ X))`
- **Python**: 
  - `seaborn.boxplot(x='X', y='Y', data=df)`
  - `scipy.stats.shapiro(Y)`
  - `scipy.stats.levene(Y1, Y2, center='median')`

## 9. 报告清单 (Reporting Checklist)

在科研论文的方法或结果部分报告探索性分析时，建议包含以下要素：
1. **变量定义**：明确指出因子及因变量的测量尺度。
2. **正态性评估**：同时报告图示结论、统计量数值、$p$ 值及评估所用的样本量 $n$。
3. **方差齐性评估**：报告 Levene 检验时，务必写清是基于何种**中心位置**（均值还是中位数）。强烈建议报告基于中位数的结果（Brown-Forsythe）。
4. **箱线图说明**：若在图注中展示箱线图，需说明箱体代表中位数与 IQR，误差线（Whiskers）的计算逻辑，以及离群值的定义标准。

**标准报告短语示例：**
> “Q-Q 图与直方图显示 A 组数据存在明显右偏，B 组与 C 组近似正态。基于中位数的 Levene 检验（Brown-Forsythe test）提示方差齐性可接受（$F(2, 87) = 1.23, p = .297$）。鉴于多数组别正态性尚可且样本量适中（$n > 30$），采用 Welch ANOVA 进行组间比较，并以自助法（Bootstrap）置信区间作为敏感性验证；描述性统计同时报告均值 ± 标准差与中位数（IQR）。”
>
> *注：现代统计实践不建议以 Shapiro-Wilk 等正态性检验的 $p$ 值机械决定是否切换为非参数方法。应优先结合图形诊断、效应量和研究目的综合判断；对大样本可依赖中心极限定理的稳健性，对小样本或严重偏态可考虑稳健方法或置换检验。*

## 10. 参考文献与手册

- Tukey, J. W. (1977). *Exploratory Data Analysis*. Reading, MA: Addison-Wesley. (箱线图与 EDA 思想的奠基作)
- IBM SPSS Statistics Base Algorithms. (具体算法：EXAMINE / Explore 模块的缺失值处理与百分位数插值逻辑)
- Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. *Journal of the Royal Statistical Society: Series B*, 26(2), 211-243. (方差稳定化变换理论)

---

## 附录：图 (Plots) 子对话框选项详解

*此部分主要针对 SPSS 图形化界面的具体选项进行严谨释义，并清理了中文 UI 中常见的翻译干扰。*

### 1. 箱线图 (Boxplots)

| 选项名称 | 统计学含义与布局 |
| :--- | :--- |
| **因子级别并置 (Factor levels together)** | 多因子设计时，系统按**因变量**分组，在一个图表内将不同因子水平的箱形并列展示（最常用于单变量组间比较）。 |
| **因变量并置 (Dependents together)** | 多因变量设计时，系统按**因子水平**分组，在一个图表内比较不同因变量的箱形（注意：若各因变量量纲不一致，此图将失去可读性）。 |
| **无 (None)** | 抑制箱线图输出。 |

### 2. 描述 (Descriptive)

| 选项名称 | 统计学含义与布局 |
| :--- | :--- |
| **茎叶图 (Stem-and-leaf)** | 文本型的直方图替代品，保留了原始数据的有效数字，适用于展示中小样本的具体分布形态。 |
| **直方图 (Histogram)** | 展示数据的频数分布，用于直观评估偏度、峰度与双峰现象。 |

### 3. 含检验的正态图 (Normality plots with tests)
勾选此项后，将强制输出 **Q-Q 图、去势 Q-Q 图 (Detrended Q-Q)**，并在描述统计表中增加 **Kolmogorov-Smirnov** 和 **Shapiro-Wilk** 检验结果。

### 4. 含莱文检验的分布-水平图 (Spread vs. Level with Levene test)
此模块用于评估方差齐性，并为非齐性数据提供变换建议。

| 选项名称 | 统计学含义与操作 |
| :--- | :--- |
| **无 (None)** | 抑制此检验扩展。 |
| **幂估计 (Power estimation)** | 仅计算方差稳定化所需的**变换幂指数**及各组中位数的自然对数散点图，但不进行任何方差齐性检验。（注：此处及下方的“幂”绝非功效，而是指 $Y^\lambda$ 中的 $\lambda$）。 |
| **转换 (Transformed)** | <br>- **自然对数 (Natural log)**：应用 $\ln$ 变换计算 Levene。<br>- **平方 (Square) / 平方根 (Square root) / 倒数 (Reciprocal)等**：应用对应变换后重新计算 Levene，观察变换是否能使方差达到齐性。 |
| **未转换 (Untransformed)** | 对**原始尺度**的数据执行标准的 Levene 检验（最常用选项）。 |
