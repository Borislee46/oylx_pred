---
source: analysis-library/cards/02-linear-models/mixed-models-spss-dialogs.md
improve_fingerprint: 38136604e8be9ec556b4dff6e9eeaf85801fffc70f872a7862106ce7a6ea6f02
prompt_digest: bbf83924da02e3b0
cached: false
---

# 线性混合模型 / 广义线性混合模型（LMM / GLMM）

- **slug**: mixed-models-spss
- **菜单（典型）**：**分析 → 混合模型 → 线性…** / **广义线性…**（Analyze → Mixed Models）；新版 SPSS 或合并为 **GLMM** 向导。**具体界面以本机软件版本为准。**

## 1. 回答什么问题

在**聚类（嵌套）**或**纵向（重复测量）**数据结构中，同时估计**固定效应**（总体平均趋势）与**随机效应**（组间/个体间异质性方差），显式建模并处理**主体内相关性**。
- **LMM**：假设因变量在给定预测变量和随机效应下服从正态分布。
- **GLMM**：扩展至非正态因变量（如二项、泊松、负二项、伽玛等分布），通过连接函数（Link Function）与线性预测子建立联系，并引入随机效应。
- **零模型与组内相关系数（ICC）**：构建无预测变量的空模型计算 ICC（组间方差 / 总方差）。若 ICC 显著大于 0（一般 > 0.05 或 0.1），则证明数据存在明显的层级/聚类效应，使用混合模型比普通 GLM/OLS 更具统计学与逻辑上的必要性。

## 2. 不适用与常见滥用

- **主体/重复变量指定错误**：将导致协方差结构与自由度（df）计算完全错误，第一类错误率失控。
- **小样本下的自由度陷阱**：样本量较小或设计不平衡时，默认的「残差法（Residual）」会高估自由度。**必须**使用 **Kenward-Roger (KR)** 或 **Satterthwaite** 近似法校正固定效应的检验统计量。
- **过度参数化与收敛失败**：在数据量不足时指定过多的随机效应（如包含多个交互项的随机斜率），或将协方差类型设定为**非结构化（Unstructured, UN）**，极易导致模型无法收敛或过拟合。
- **GLMM 的分离与零膨胀**：在二项分布中出现完全分离（Complete Separation），或计数数据中 0 的比例远超泊松/负二项预期，需改用零膨胀模型（如 ZIP/ZINB）或贝叶斯正则化方法。
- **中心化（Centering）策略误用**：在多层模型中，第一层连续自变量的**总均值中心化（Grand-mean centering）**与**组均值中心化（Group-mean centering）**具有完全不同的理论解释。不加区分地输入原始变量可能导致层内效应与层间效应的混淆（Ecological Fallacy）。

## 3. 数据与设计前提

明确区分数据结构中的两个关键维度：
1. **主体变量（Subject / Level-2 ID）**：定义聚类单位（如患者、学校、多中心临床的中心）。代表 **G 侧（G-side）** 随机效应，主体间通常假定独立。
2. **重复变量（Repeated / Level-1 Index）**：定义重复测量的索引（如访视次数、时间点）。代表 **R 侧（R-side）** 协方差结构，描述同一个体内残差随时间的自相关性。

**明确随机效应的物理意义：**
- **随机截距（Random Intercept）**：允许每个主体拥有不同的基线起点（组间个体差异）。
- **随机斜率（Random Slope）**：允许自变量（如时间）对因变量的效应因主体而异（个体的变化率差异）。

## 4. 模型构建策略与检验统计量

混合模型通常需要逐步构建与比较（Model Building Strategy）：

1. **嵌套模型比较（似然比检验, LRT）**：通过比较 $-2 \times \text{Log Likelihood}$ 的差值（服从 $\chi^2$ 分布）来评估模型拟合优度。
2. **ML 与 REML 的黄金法则**：
   - 当比较**固定效应**不同但随机结构相同的嵌套模型时，**必须**使用最大似然估计（**ML**）。
   - 当比较**随机效应 / 协方差结构**不同但固定效应相同的嵌套模型时，**必须**使用约束最大似然估计（**REML**）。
   - **最终报告参数估计值**时，通常以 **REML** 结果为准（因为 ML 对方差成分的估计是有偏的，尤其在小样本下）。
3. **固定效应检验**：主要依赖 **F 检验** 或 **t 检验**，自由度（df）依实现与设定计算（如前述 KR 法）。对于 GLMM，部分软件输出伪似然（Pseudo-likelihood）统计量，其检验逻辑与 LMM 有所区别，需查阅对应软件手册。

## 5. 假设、诊断与排障（Troubleshooting）

| 假设/诊断目标 | 检查方式 / 统计量 | 违背时的典型后果 |
| :--- | :--- | :--- |
| **残差正态性（第一层）** | 残差的 Q-Q 图、直方图 | 固定效应的推断可能产生偏差，置信区间不准。 |
| **随机效应正态性（第二层）** | 提取经验贝叶斯估计量（EBE / BLUPs）绘制 Q-Q 图 | 影响总体方差成分的估计与个体预测的准确性。 |
| **协方差结构合理性** | AIC / BIC / AICC 等信息准则比较 | 结构过简导致标准误（SE）偏小（假阳性）；结构过繁导致效率降低（假阴性）。 |

**收敛失败时的排障指南（Troubleshooting）：**
1. **检查变量尺度（Scaling）**：若自变量量级差异过大（如收入为 100000，年龄为 20），极易导致 Hessian 矩阵不可逆。需对连续变量进行标准化（Z-score）。
2. **简化模型结构**：从最复杂的非结构化协方差（UN）退化为方差成分（VC）或自回归（AR1）；剔除方差极小或接近边界（0）的随机斜率项。
3. **调整算法参数**：在高级设置中增加最大迭代次数、放宽收敛容差，或（在 R/Python 中）更换优化器（Optimizer）。

## 6. 效应量与置信区间

在学术报告中，仅提供 $p$ 值和系数已不再满足规范，建议报告：
1. **固定效应估计**：线性模型的均值差（Mean Difference）、Logistic回归的优势比（OR）、泊松回归的率比（IRR），并辅以 **95% 置信区间（CI）**。
2. **随机方差成分**：报告随机截距/斜率的方差估计值及其 CI（若软件支持）。
3. **现代 $R^2$ 规范（Nakagawa $R^2$）**：
   - **边际 $R^2$（Marginal $R^2$）**：仅由模型中的**固定效应**所解释的方差比例。
   - **条件 $R^2$（Conditional $R^2$）**：由**固定效应 + 随机效应**共同解释的方差比例。

## 7. 与相关方法的取舍

- **广义估计方程（GEE）**：GEE 是群体平均（Population-Average）模型，关注总体的边际效应，不严格假设随机效应的具体分布形式，具有稳健性；而 GLMM 是主体现有（Subject-Specific）模型，允许对具体个体进行预测，解释时需带有「在给定某个体随机效应的前提下」的限定。
- **重复测量 ANOVA**：强烈依赖球形假设（Sphericity），且无法处理带有缺失值的不平衡纵向数据；MIXED 过程不对缺失值要求严格（MCAR或MAR下均可运行），且允许灵活定义方差-协方差结构。

## 8. 实现速查

- **SPSS**：`MIXED`（线性）、`GENLINMIXED`（广义）。附录提供语法示例。
- **R**：`lme4::lmer` / `glmer`（最常用）；`nlme::lme`（复杂协方差支持佳）；`mmrm`（制药及临床试验中专门用于混合模型重复测量分析的顶尖包）；`glmmTMB`（处理零膨胀/过度离散极佳）。
- **Python**：`statsmodels.regression.mixed_linear_model.MixedLM`；若需与 R 标定结果一致，推荐通过 `pymer4` 调用 R 的 `lme4` 引擎。

## 9. 报告清单（科研方法节必备）

撰写方法学与结果时，请核对是否涵盖以下信息：
- [ ] **聚类/层级设计信息**：什么是 Level-1，什么是 Level-2（如「将随访次数嵌套于患者个体中」）。ICC 是多少。
- [ ] **模型具体设定（公式）**：明确指出哪些变量是固定效应（主效应及交互项），哪些是随机效应（截距还是斜率）。
- [ ] **协方差结构**：说明选用了何种 R 侧协方差类型（如无结构 UN、一阶自回归 AR(1)）及其选择依据（如基于 AIC/BIC 的最优选择）。
- [ ] **估计方法**：明确指出使用了 ML 还是 REML（通常最终结果为 REML）。
- [ ] **自由度近似方法**：明确说明是否应用了 Kenward-Roger 或 Satterthwaite 校正。
- [ ] **软件及版本**：标注具体使用的统计软件、函数包或过程模块。

## 10. 参考文献与手册

- Pinheiro, J. C., & Bates, D. M. (2000). *Mixed-Effects Models in S and S-PLUS*. Springer. (混合模型经典教材)
- Nakagawa, S., & Schielzeth, H. (2013). A general and simple method for obtaining $R^2$ from generalized linear mixed-effects models. *Methods in Ecology and Evolution*, 4(2), 133-142. ($R^2$ 计算核心文献)
- IBM SPSS Statistics Algorithms / Documentation: 查阅对应版本的 `MIXED` 与 `GENLINMIXED` 过程手册。

---

## 附录 A：SPSS 菜单设计 — 主体与重复结构

在使用 SPSS `MIXED` 对话框时，需准确理解第一页的定义：

- **主体(S) (Subjects)**：定义聚类 ID（如患者编号）。指定后，软件知道在这些 ID 之间的残差是独立的，但在同一个 ID 内部允许存在相关性。
- **重复(E) (Repeated)**：定义测量的顺序或坐标（如访视月数 `Month`）。用于建立 **R 侧（残差）协方差矩阵**。如果不需要模拟残差随时间的序列相关性（而仅靠随机截距吸收相关），此项可不填。

**重复协方差类型(V) 的常见选择：**
| 类型 | 缩写 | 适用场景 |
| :--- | :--- | :--- |
| **对角线（方差成分）** | VC | 假设各重复测量点之间**无相关**，仅方差恒定（退化为普通残差）。 |
| **一阶自回归** | AR(1) | 假设时间上越近的测量点相关性越强，随时间间隔增加，相关性呈指数衰减。适用于**等间距**的时间序列。 |
| **复合对称** | CS | 假设所有时间点之间的相关性均相等。随机截距模型诱导的**边际**协方差结构为 CS，但**不完全等价**：随机截距约束 $\rho \ge 0$，而直接指定 CS 协方差结构允许负相关。 |
| **无结构 / 非结构化** | UN | 最灵活，不对相关性做任何假设，估算所有可能的方差和协方差。**参数极多，小样本极易不收敛**。 |

*注：若使用空间协方差（如空间高斯、空间指数），需同时提供连续的空间/时间坐标变量。*

## 附录 B：GLMM — 目标与分布连接速查

当因变量违背正态分布时，需在 GLMM 中指定“分布”与“连接函数”：

| 预设/数据类型 | 理论分布 | 默认连接函数 | 典型适用场景 |
| :--- | :--- | :--- | :--- |
| **连续对称** | 正态 (Normal) | 恒等 (Identity) | 经典的 LMM。 |
| **连续正偏态** | 伽玛 (Gamma) | 对数 (Log) | 医疗花费、反应时（大于0且呈严重右偏）。 |
| **二分类/事件发生**| 二项式 (Binomial) | Logit / Probit | 患病与否、成功/失败。 |
| **计数 (常规)** | 泊松 (Poisson) | 对数 (Log) | 罕见事件次数（均值≈方差）。 |
| **计数 (过度离散)**| 负二项式 (Negative Binomial) | 对数 (Log) | 事件发生次数的方差显著大于均值。 |
| **名义多分类** | 多项式 (Multinomial)| 广义 Logit | 互斥且无序的多个分类（如血型、偏好品牌）。 |

## 附录 C：SPSS 核心 Syntax 模板

由于 SPSS 新版 GLMM 的向导界面极其繁琐，在实务中强烈推荐直接使用语法（Syntax）运行及微调。以下提供两个可直接复用的模板：

**模板 1：基础线性混合模型（LMM）- 带有随机截距与固定交互项**
```spss
* 假设 Y 为正态连续因变量, Group 为分组, Time 为访视轮次, SubjectID 为患者编号.
MIXED Y BY Group Time WITH Age
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100)
  /FIXED=Group Time Group*Time Age | SSTYPE(3)
  /METHOD=REML
  /PRINT=SOLUTION TESTCOV
  /RANDOM=INTERCEPT | SUBJECT(SubjectID) COVTYPE(VC).
```
*(注：`BY` 后接分类因子，`WITH` 后接连续协变量。建议始终加入 `DFMETHOD` 指定自由度校正。)*

**模板 2：二分类广义线性混合模型（GLMM）- 随机截距**
```spss
* 假设 Y_Bin 为 0/1 变量.
GENLINMIXED
  /DATA_STRUCTURE SUBJECTS=SubjectID
  /MODEL_EFFECTS FIXED=Group Time
  /TARGET_OPTIONS REFERENCE=0 DISTRIBUTION=BINOMIAL LINK=LOGIT
  /RANDOM_EFFECTS SUBJECT_GROUPING=SubjectID EFFECT=INTERCEPT COVARIANCE_TYPE=VARIANCE_COMPONENTS
  /BUILD_OPTIONS TARGET_CATEGORY_ORDER=ASCENDING MAX_ITERATIONS=100 CONFIDENCE_LEVEL=95.
```
