---
source: analysis-library/cards/01-inference/regression-linear-logistic-curve-quantreg-spss-dialogs.md
improve_fingerprint: 41683aa33f99edbf26956f3e3c1eef88c46eafab6da3182ac5aa6f0f7678328e
prompt_digest: bbf83924da02e3b0
cached: false
---

# 回归 — 曲线估算 / 多元 Logistic / 非线性 / 分位数（SPSS 对话框手册与报告指南）

- **slug**: regression-dialogs-spss-bundle-other
- **说明**：本文档梳理 SPSS 中四类非标准线性回归模型（曲线估算、多元 Logistic、非线性、分位数）的界面对话框统计含义、算法假设及可复现语法。
- **关联文档**：标准模型已拆分为独立卡片，详见 [线性回归（OLS）](regression-linear-ols-spss.md) 与 [二元 Logistic 回归](regression-binary-logistic-spss.md)。

---

## 1. 曲线估算（Curve Estimation）

用于快速探索单个连续自变量（$X$）与一个或多个连续因变量（$Y$）之间的非线性趋势。常用于时间序列的初步趋势拟合。

### 1.1 对话框选项与统计含义

| 界面选项 | 统计含义与注意事项 |
| :--- | :--- |
| **因变量 / 自变量** | **仅限单 $X$**；允许输入多个因变量，软件将依次对同一 $X$（如时间序列变量）进行拟合。 |
| **在方程中包括常数** | 决定模型是否包含截距项（$\beta_0$）。一般建议保留，除非有强烈的理论预设（如 $X=0$ 时 $Y$ 必须为 $0$）。 |
| **模型** | 提供 11 种预设函数形态：**线性、二次、复合、增长、对数、三次、指数、逆、幂**。<br>⚠️ **核心预警**：此处的 **Logistic** 指的是时间序列中的 **S 形增长曲线**（即 $Y = 1 / (1/u + b_0 b_1^X)$），**绝对不是**用于分类因变量的 Logistic 回归。 |
| **显示 ANOVA 表** | 输出模型整体显著性 $F$ 检验。 |
| **保存** | 可保存预测值、残差、预测区间上限/下限。 |

### 1.2 方法学提示与可复现语法 (Syntax)
大部分曲线估算本质上是对 $X$ 或 $Y$ 进行变量变换（如对数转换）后的**普通最小二乘法 (OLS)** 拟合。因此，线性回归的残差独立性、同方差性及正态性假设依然适用。

```spss
* 曲线估算示例：拟合线性、二次与对数曲线.
CURVEFIT
  /VARIABLES=Y WITH X
  /CONSTANT
  /MODEL=LINEAR QUADRATIC LOGARITHMIC
  /PLOT FIT.
```

---

## 2. 多元 Logistic 回归（Multinomial Logistic Regression）

用于因变量为**无序多分类**（如出行方式：公交、地铁、自驾）的预测与统计推断。模型同时估计多个方程，以某一类别为**参考类（Reference Category）**，计算其他类别相对于参考类的发生比（Odds）。

### 2.1 核心假设（方法学必查）
- **无关选项独立性假设（IIA, Independence of Irrelevant Alternatives）**：这是该模型最核心的假设。即：增加或减少某一个分类选项，不应改变现有其他选项之间的相对发生比。若违背该假设（如选项间存在高度替代性），应改用嵌套 Logit 模型（Nested Logit）或多项 Probit 模型。
- **无完全/准完全分离（No Data Separation）**：若自变量的某种组合能完美预测因变量的某一类别（多见于样本量小或分类变量交叉存在空单元格），会导致极大似然估计不收敛，标准误（$SE$）异常膨胀。

### 2.2 界面选项解析

#### 统计与输出 (Statistics)
| 选项 | 统计含义与报告价值 |
| :--- | :--- |
| **伪 R 方 (Pseudo $R^2$)** | 输出 McFadden、Cox–Snell、Nagelkerke。<br>⚠️ **注**：仅供参考，**不具有** OLS 中 $R^2$ 的“方差解释比例”物理含义，切勿按线性回归逻辑解读，建议报告为辅助拟合指标。 |
| **模型拟合信息** | 输出 -2 对数似然值 (-2LL) 及似然比检验（卡方），用于检验模型整体（全模型 vs 仅含常数项模型）是否显著。 |
| **信息准则** | AIC、BIC，用于非嵌套模型间的比较（值越小越优）。 |
| **拟合优度** | Pearson 卡方与 Deviance 偏差统计量（检验模型拟合度，原假设为模型完美拟合，故 $p > 0.05$ 示为拟合良好）。<br>⚠️ **重要限制**：此结论**仅对分组/分箱数据成立**。对个体级（未分组）多项/Logistic 数据，大样本下 Pearson/Deviance GOF 几乎必拒绝（与 H-L 同源问题），不可作为拟合良好的证据。 |
| **参数估计值** | 核心推断输出：包含 $B$（对数发生比）、$SE$、Wald 检验、$Exp(B)$（即 OR 值）及其置信区间。 |

*(注：SPSS 旧版本可能在对话框中误植或混用 Somers' D 等单调性测度。单调性测度仅适用于**有序 Logistic 回归 (PLUM)**，多项无序分类在数学上不存在自然顺序，不可报告单调性指标。)*

#### 收敛条件与选项 (Criteria & Options)
| 选项 | 统计含义与故障排除 |
| :--- | :--- |
| **最大迭代 / 步长减半** | 极大似然估计（MLE）基于牛顿-拉夫逊或拟牛顿法。若不收敛，可尝试增加迭代次数。 |
| **检查数据点分离** | 强烈建议勾选。用于识别前文所述的完全/准分离现象，防止报告虚假的极大 $Exp(B)$。 |
| **离散标度 (Scale)** | 当数据存在**过度离散 (Overdispersion)**（即方差大于二项分布理论方差）时，可用 Pearson 卡方或偏差除以自由度来校正标准误。 |

### 2.3 可复现语法 (Syntax) 与报告清单

```spss
* 多元 Logistic 回归示例 (设参考类别为最后一类).
NOMREG Y (BASE=LAST ORDER=ASCENDING) BY FactorX WITH CovariateZ
  /CRITERIA CIN(95) DELTA(0) MXITER(100) MXSTEP(5) CHKSEP(20) LCONVERGE(0) PCONVERGE(0.000001) SINGULAR(0.00000001)
  /MODEL
  /STEPWISE=PIN(.05) POUT(0.1) MINEFFECT(0) RULE(SINGLE) ENTRYMETHOD(LR) REMOVALMETHOD(LR)
  /INTERCEPT=INCLUDE
  /PRINT=FIT PARAMETER SUMMARY LRT CPS STEP MFI.
```

#### 📝 论文报告清单：多元 Logistic 回归 (APA 标准)
在方法与结果小节中，需依次报告以下要素：
1. **数据与设计说明**：说明样本量大小、因变量的类别及分布，明确声明**参考类别（Reference Category）**。提及如何处理缺失值与共线性检查。
2. **模型整体拟合度**：报告似然比检验（Likelihood Ratio Test）的 $\chi^2$ 值、自由度 $df$ 和 $p$ 值，证明加入预测变量后的模型显著优于空模型。
3. **模型解释力（辅助）**：报告伪 $R^2$（学术界最常报告 Nagelkerke $R^2$ 或 McFadden $R^2$），并用“模型拟合度指标”而非“方差解释率”来描述它。
4. **主效应检验 (LRT)**：报告各预测变量的似然比卡方值与 $p$ 值，说明该变量的整体显著性。
5. **参数估计与发生比率 (OR)**（核心表格）：
   - 按类别对分面板呈现。
   - 必须报告：非标准化系数 $B$、标准误 $SE$、Wald $\chi^2$、$p$ 值。
   - 核心效应量：**$Exp(B)$ (即 OR 值)** 及其 **95% 置信区间 (95% CI)**。明确解释为“在控制其他变量的情况下，$X$ 每增加一个单位，该类别相对于参考类别的发生比的变化倍数”。

---

## 3. 非线性回归（Nonlinear Regression）

当因变量与自变量之间的关系无法通过简单的变量代换转化为线性模型时（如复杂的药代动力学模型、Michaelis-Menten 酶促动力学等），需使用非线性最小二乘法估算参数。

### 3.1 界面选项与算法差异

| 选项 / 模块 | 统计含义 |
| :--- | :--- |
| **模型表达式与初始值** | **非线性回归必须由用户输入数学方程，并提供参数的初始猜测值**。初始值的设定直接决定模型能否找到全局最优解而非陷入局部最优。 |
| **估算方法：L-M 算法** | **Levenberg-Marquardt (NLR)** 算法：最常用的无约束非线性最小二乘优化算法。结合了梯度下降（远距离快速收敛）与高斯-牛顿法（近距离精确收敛）的优点。 |
| **估算方法：SQP 算法** | **序列二次规划 (CNLR, Constrained NLR)**：当参数受到严格的理论约束时（例如反应速率常数必须 $k > 0$ 或限制在某区间内），必须使用此算法。 |
| **标准误差的 Bootstrap** | 非线性回归的参数标准误及置信区间依赖于渐近正态假设。当样本量较小或残差分布不理想时，强烈建议勾选 Bootstrap，通过重抽样获取更稳健的 $SE$ 和 CI。 |

### 3.2 可复现语法 (Syntax)

```spss
* 非线性回归 (NLR) 示例：拟合自定方程 y = b1 * exp(b2 * x).
* 必须先设定参数及其初始值.
MODEL PROGRAM b1=1 b2=0.1.
COMPUTE PRED_ = b1 * exp(b2 * x).
NLR y
  /OUTFILE='NLR_results.sav'
  /PRED PRED_
  /CRITERIA SSCONVERGENCE 1E-8 PCONVERGENCE 1E-8.
```

---

## 4. 分位数回归（Quantile Regression）

有别于 OLS 估计”条件均值”（$E[Y|X]$），分位数回归估计的是因变量的”条件分位数”（如中位数 $Q_{0.5}[Y|X]$）。
**优势**：对**Y 方向离群值**极度稳健（注意：仅对 Y 方向离群稳健，对 X 方向高杠杆点仍需警惕）；无需假定残差正态性；能揭示自变量对因变量分布不同位置（如高收入群体 vs 低收入群体）的异质性影响。
**系数解释**：$\beta(\tau)$ 表示 $X$ 每变化一个单位，$Y$ 的第 $\tau$ 条件分位数的变化量——**不是**条件均值的变化，解释时务必强调分位数视角（如”在收入分布的 90 分位处，教育溢价为 XX”）。

### 4.1 对话框核心选项

#### 分位数设定 (Quantiles)
| 选项 | 含义与用法 |
| :--- | :--- |
| **指定单个分位数** | 默认 $\tau = 0.50$ 即中位数回归（Least Absolute Deviations, LAD），对极值最稳健。 |
| **指定网格分位数** | 设定多个分位数（如 0.10 到 0.90，步长 0.10），用于观察系数随分位数变化的轨迹。论文中常结合「分位数回归系数图」进行报告。 |

#### 估算方法 (Estimation Methods)
| 算法选项 | 数学含义与适用场景 |
| :--- | :--- |
| **单纯形法 (Simplex)** | 基于线性规划（Linear Programming）求解。适用于中小型数据集，是传统的标准分位数优化算法。 |
| **Frisch-Newton 内点法** | 一种平滑算法。当样本量巨大或模型十分复杂时，单纯形法可能极慢，此时内点法的计算效率远超单纯形法。 |

#### 估算后推断 (Post-Estimation & Inference)
| 选项 | 统计含义 |
| :--- | :--- |
| **假定个案 IID** | 若假定误差项独立同分布，则采用标准的渐近协方差矩阵计算标准误。 |
| **Huber-White / 稳健** | 当存在异方差或聚类/面板数据特征时，必须取消 IID 假设，采用三明治方差估计量以保证推断的有效性。 |
| **带宽计算法** | 分位数回归的协方差计算依赖于对真实误差密度的估计。提供 **Bofinger** 和 **Hall-Sheather** 两种带宽规则，通常样本量较大时两者结果相差无几，按软件默认报告即可。 |

### 4.2 可复现语法 (Syntax)
*(注：SPSS v26+ 提供了原生的 `QUANTILE REGRESSION` 命令，早期版本可能需要通过 Python 扩展或在混合模型中调用)*

```spss
* 分位数回归示例：同时估计 0.25, 0.50, 0.75 三个分位数.
QUANTILE REGRESSION Y WITH X1 X2
  /CRITERIA QUANTILE=0.25 0.50 0.75
  /MODEL ALGORITHM=SIMPLEX
  /INFERENCE METHOD=BOOTSTRAP CI=95
  /PRINT=PARAMETER.
```

---

## 附录：参考文献与扩展查阅

撰写论文方法学部分或进行参数微调时，请参考 IBM SPSS 官方算法手册（对应当前使用的软件版本）：
- `CURVEFIT` (Curve Estimation)
- `NOMREG` (Multinomial Logistic Regression)
- `NLR` / `CNLR` (Nonlinear Regression)
- `QUANTILE REGRESSION` (Quantile Regression Algorithms)
