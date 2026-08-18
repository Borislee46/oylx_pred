---
source: analysis-library/cards/08-survey/complex-samples-menu-spss-overview.md
improve_fingerprint: 7049384f65c588dad700854fdd9a50c7dbdbc2120dcde84d6328901c3497174e
prompt_digest: bbf83924da02e3b0
cached: false
---

# 复杂抽样（Complex Samples）— 菜单总览

- **slug**: complex-samples-menu-spss
- **菜单**：**分析 → 复杂抽样**（Analyze → Complex Samples）
- **前提**：必须首先通过 **选择样本（Select a Sample）** 或 **准备分析（Prepare for Analysis）** 模块生成并定义 **抽样计划文件（.csplan）**，绑定 **抽样权重（Weights）、分层（Strata）、聚类/初级抽样单元（PSU）** 以及 **有限总体校正（FPC）**；否则后续推断的 **标准误（SE）与 p 值** 将完全失效。

## 1. 回答什么问题

在非简单随机抽样（如**分层、多阶段聚类、不等概率抽样、有限总体**）设计下，对总体的**频数、描述统计量、列联表交叉、比率**，以及**回归模型（线性、Logistic、有序、Cox 生存分析）** 进行**设计一致（Design-consistent）** 的点估计，并计算正确的**标准误与置信区间**。
该模块的核心在于通过调整方差计算公式，报告**设计效应（DEFF）**、**有效样本量（Effective $N$）**，并支持正确的**子总体（Domain/Subpopulation）** 推断。

## 2. 不适用 / 易滥用

- **忽略设计直接分析**：在复杂设计数据上盲目使用普通的 `FREQUENCIES` 或 `REGRESSION` 过程。由于聚类内的组内相关性（ICC），忽略设计**通常会导致标准误（SE）被严重低估，p 值过于乐观（假阳性激增）**，除非该数据实际上是简单随机抽样（SRS）。
- **滥用普通数据筛选进行子总体分析**：**切勿使用普通的数据筛选（如 `SELECT IF` 或 `FILTER` 剔除无关数据）来分析子群体。** 物理剔除数据会导致软件丢失全局的抽样框与方差结构信息（自由度计算错误）。必须在完整数据集的基础上，使用复杂抽样过程内的 **子总体（Subpopulation/Domain）** 功能进行声明。
- **对“权重”的误解**：抽样权重不能简单理解为“案例重要性的随意放大”。复杂抽样的最终分析权重（Final Weight）通常是 **“基础抽样概率的倒数 $\times$ 无应答调整 $\times$ 事后分层校准（Post-stratification）”**，它决定了样本代表总体的规模与结构。

## 3. 数据与设计前提

- **元数据匹配**：数据集中的 **PSU（聚类）、层（Strata）、权重** 变量，以及有放回/无放回（WR/WOR）的设定，必须与原始抽样设计文档严格一致。
- **孤立初级抽样单元（Singleton PSU）警告**：在分层设计中，**每层必须至少包含两个 PSU**。若某层内只有一个 PSU，软件将无法估算该层内的方差（泰勒线性化会报错或无法计算）。遇到此情况通常需要依据统计准则合并相邻层（Collapsing strata）或对该层进行中心化处理。
- **有限总体校正（FPC）**：当抽样比例较大（通常 $> 5\%$）且为无放回抽样时，需指定 FPC 以缩减方差估计；是否启用需依抽样设计与软件的具体算法约定。

## 4. 模型或检验统计量（含自由度逻辑）

- **方差估计方法（$Var(\hat{\theta})$）**：
  - **泰勒级数线性化（Taylor Series Linearization）**：SPSS 复杂抽样模块**默认且最核心**的方差估计方法，适用于大多数显式分层聚类设计。
  - **重复复制（Replicate Weights, 如 Jackknife, BRR, Bootstrap）**：部分调查数据仅提供重复权重。SPSS 对此支持相对有限，若重抽样设计复杂，推荐切换至 R（`survey` 包）或 Stata（`svy` 体系）。
- **近似自由度（Degrees of Freedom, $df$）**：复杂抽样下的 $df$ 通常不再是“样本量减去参数个数”，而是由设计结构决定。最常见的近似 $df$ 为 **总聚类数减去总层数（$n_{PSU} - n_{Strata}$）**。这是评估检验效力（Power）的关键约束。
- **假设检验统计量**：普通检验在复杂抽样中不再适用。列联表与回归系数检验通常采用 **设计校正的 Wald 统计量** 或 **Rao-Scott 校正卡方统计量**。

## 5. 假设与诊断

| 假设 / 前提 | 检查方式 / 诊断 | 违背时的典型后果 | 推荐应对策略 |
| :--- | :--- | :--- | :--- |
| **设计与数据匹配** | 交叉核对抽样文档与 `.csplan` 设定 | 统计推断（SE、p值）完全无效 | 重新审查抽样权重生成与分层合并逻辑 |
| **无孤立 PSU** | 频数检查：确保每层内 $N_{PSU} \ge 2$ | 无法计算方差，程序报错或强制忽略该层 | 相邻层合并（Collapse strata） |
| **权重无极端离群值** | 检查权重分布的极值（Max/Min比值） | 极端权重导致方差极度膨胀，估计不稳定 | 根据统计手册进行**权重截断（Weight Trimming）** |
| **模型正确设定（回归）** | 残差分析、共线性诊断、ROC 曲线 | 系数解释偏倚、预测失效 | 引入非线性项、调整协变量（同普通回归） |

## 6. 效应量与区间

- **点估计及置信区间（CI）**：报告基于加权的正确点估计，以及基于**设计校正 SE** 计算的置信区间（如 $\hat{\theta} \pm t_{\alpha/2, df} \times SE_{design}$）。OR / HR 等效应量同理。
- **设计效应（DEFF）**：$DEFF = \frac{Var_{complex}(\hat{\theta})}{Var_{SRS}(\hat{\theta})}$。用于衡量复杂抽样带来的方差惩罚（聚类通常使 $DEFF > 1$）。
- **有效样本量（Effective $N$）**：$N_{eff} = \frac{N}{DEFF}$。反映该复杂样本提供的实际信息量等价于多大规模的简单随机样本。

## 7. 与相关方法的取舍

- **R语言 (`survey` 包) / Stata (`svy` 前缀)**：统计思想与 SPSS 完全一致。R 和 Stata 在处理**多重插补后的复杂抽样（survey + MI）**、特定的重复复制权重（Replicate Weights）及前沿因果推断模型时，生态更完善、更灵活。SPSS 复杂抽样模块与多重插补模块的整合有限，复杂设计下需缺失值处理时建议转 R（`mice` + `survey` 联合）或 Stata（`mi svy` 体系）。
- **SPSS Complex Samples**：优势在于基于向导的 `.csplan` 计划体系，对菜单用户友好，常规的描述、广义线性模型（GLM）和生存分析功能已高度一体化。

## 8. 实现速查

- **SPSS**：先通过向导生成计划文件（向导后端调用 `CSPLAN` 语法）；后续分析基于 `CSDESCRIPTIVES`, `CSTABULATE`, `CSGLM`, `CSLOGISTIC`, `CSCOXREG` 等过程命令（详见附录 B）。
- **R Ecosystem**：使用 `survey` 包（`svydesign()` 声明设计，`svymean()`, `svyglm()` 进行分析）。
- **Python Ecosystem**：`statsmodels` 和 `samplics` 包提供了部分支持。**注：** 目前 Python 在复杂调查数据上的生态远不及 R 和 Stata/SPSS 成熟，处理真实复杂设计时推荐优先选用 R 或 Stata。

## 9. 报告清单

在方法学或结果小节中，为保证研究的严谨性与可复现性，应报告：
1. **未加权样本量（Unweighted $n$）**：**必须**与加权后的总体估计值同时报告，以便读者判断基础统计功效。
2. **抽样设计简述**：说明分层（Strata）、聚类（PSU）、各级权重（Weights）的构成及有限总体校正（FPC）使用情况。
3. **软件与算法声明**：注明使用的软件模块（如 IBM SPSS Complex Samples 模块及版本）与方差估计方法（如 泰勒级数线性化）。
4. **子总体定义**：若进行了子群体分析，须明确声明使用的是域估计（Domain Analysis）而非直接剔除数据。
5. **推断质量指标**：关键参数应报告标准误（SE）、95% CI，推荐报告**设计效应（DEFF）** 和 **有效样本量（Effective $n$）**。

## 10. 参考文献与手册

- Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). John Wiley & Sons.（抽样理论经典奠基之作）
- Lumley, T. (2010). *Complex Surveys: A Guide to Analysis Using R*. John Wiley & Sons.（包含大量算法实现对比与解释）
- IBM SPSS Statistics: *Complex Samples Module User Manual*（参考具体使用的软件版本官方说明，明确其算法默认设置）。

---

## 附录 A：数据准备枢纽（Plan File）

| 菜单项 | 功能与含义 |
| :--- | :--- |
| **选择样本 (Select a Sample)** | 根据完整抽样框构建抽样计划，执行抽样并自动生成每一阶段的权重与设计变量。 |
| **准备分析 (Prepare for Analysis)** | 已有调查数据时的必经步骤。生成 **.csplan** 文件，将数据集中的具体变量绑定至对应的层、聚类、权重以及 WOR/WR 设定。 |

## 附录 B：复杂调查分析过程字典

*(注：以下过程必须以外部挂载 `.csplan` 文件为前提执行。具体回归模型的假设检验详见对应的方法卡片)*

| 菜单/过程名（Syntax） | 推断对象与功能 |
| :--- | :--- |
| **频率 (CSFREQUENCIES)** | 加权频数、总体比例分布及其设计 SE。 |
| **描述 (CSDESCRIPTIVES)** | 连续变量的总体均值、总和（Totals）、比率及其设计 SE。 |
| **交叉表 (CSTABULATE)** | 分类变量的加权列联表；输出 Rao-Scott 校正卡方或 Wald 独立性检验。 |
| **比率 (CSRATIO)** | 估计两个连续变量的总体比率（分子/分母）及其变异程度。 |
| **一般线性模型 (CSGLM)** | 连续因变量的线性回归分析（支持多因素方差分析式的分类自变量引入）。 |
| **Logistic 回归 (CSLOGISTIC)** | 二分类因变量的概率预测与加权 OR 值推断。 |
| **有序回归 (CSORDINAL)** | 有序多分类因变量分析（如累积优势模型）。 |
| **Cox 回归 (CSCOXREG)** | 复杂样本下的生存分析与 HR 值推断。 |

## 附录 C：与同名普通过程的核心区别（防雷提醒）

复杂抽样菜单下的**同名分析**（如 CSLOGISTIC）与普通分析菜单下的过程（如 LOGISTIC REGRESSION）在数学底层上**完全不等价**。
- 普通程序假设观测值相互独立（$Cov(y_i, y_j) = 0$）。
- 复杂抽样程序通过 `.csplan` 引入三维联合分布结构，正确处理由于“同一村庄/社区受访者”（聚类）带来的同质性惩罚。若错误使用普通程序，研究者会误将设计导致的数据冗余当成新的独立信息，从而得出虚假的显著性结论。
