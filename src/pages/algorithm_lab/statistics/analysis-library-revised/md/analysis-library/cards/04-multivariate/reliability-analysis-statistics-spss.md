---
source: analysis-library/cards/04-multivariate/reliability-analysis-statistics-spss.md
improve_fingerprint: c39ee79538ab5558f40b53f9d2c8dd962b3a05268283b8daca01828240ea4522
prompt_digest: bbf83924da02e3b0
cached: false
---

# 可靠性分析（量表内部一致性、ICC、评定者间一致性）

- **slug**: reliability-statistics-spss
- **aliases**: Cronbach $\alpha$、信度、组内相关系数、评定者间信度（IRR）、Fleiss Kappa
- **菜单路径**：**分析** $\rightarrow$ **刻度** $\rightarrow$ **可靠性分析** $\rightarrow$ **统计**（及主对话框模型选项）

## 1. 回答什么问题

- **Cronbach's $\alpha$（及“删除项后 $\alpha$”）**：评估一组**题项**是否在共同测量同一个潜在维度，并给出一个基于题项间平均相关性的**内部一致性（Internal Consistency）**摘要指标。
- **ICC（组内相关系数，Intraclass Correlation Coefficient）**：基于**方差成分分解**，量化**目标（被试）间差异**相对**测量误差/评定者差异**的比率；主要用于**重复测量**或**连续变量的多评定者**设计，评估评分的绝对一致性或相对一致性。
- **Fleiss Kappa**：评估**多名评定者**在**多类别（名义变量）**评定中，一致性程度是否显著超出了**随机机遇（Chance）**。
- **Hotelling $T^2$ 与 Tukey 可加性**：检验题项均值向量的相等性，以及题项与被试之间是否存在可乘交互作用（偏离可加性将影响 $\alpha$ 系数的解释）。

## 2. 不适用 / 易滥用

- **$\alpha$ 高 $\neq$ 单维结构**：这是量表分析中最严重的常见误区。多维量表强行计算 $\alpha$ 可能因题项过多而**虚高**；$\alpha$ 的前提是单维性，而非证明单维性的工具。需通过**因子分析（EFA/CFA）**提供结构效度支持。
- **反向计分遗漏导致负 $\alpha$**：若量表中包含反向题但未进行逆向重编码（Recode），直接计算会导致题项间出现负相关，进而得出**负数的 $\alpha$ 值**或极低的信度。
- **$\alpha$ 与 ICC 不可互换**：$\alpha$ 关注的是量表总分与题项内在结构的协变关系；ICC 关注的是不同评定者/测量批次下的方差成分分配。
- **题数敏感性**：$\alpha$ 公式对题项数量（$k$）高度敏感。题项极少（如 2–3 题）时 $\alpha$ 往往偏低，此时应结合「平均题项相关系数」综合解释；题数极大时 $\alpha$ 必然逼近 1，可能掩盖多维问题。
- **缺失值处理（列表删除陷阱）**：SPSS 默认使用「列表删除（Listwise Deletion）」，只要被试在任一题项漏答，整条数据即被剔除。题量大时会导致有效样本量（$N$）锐减，甚至改变样本代表性。建议在分析前进行缺失值插补（如均值插补、EM 插补）或仔细核查有效 $N$ 数。

## 3. 数据与设计前提

- **Cronbach's $\alpha$**：
  - 数据类型：通常要求李克特式（Likert-type）或连续变量。
  - 核心假设：**本质 $\tau$ 等值（Essential Tau-equivalent）**，即假设各题项的真实分数方差相等。若违背此假设（实务中极常见），$\alpha$ 仅代表真实信度的**下限**。
- **ICC**：
  - 需明确实验设计：**随机/固定效应**（评定者是随机抽样还是固定群体）、**单向/双向模型**（One-way/Two-way）。这与 Shrout & Fleiss (1979) 提出的分类矩阵严格对应。
- **Fleiss Kappa**：
  - 数据类型：**名义类别（Nominal）**变量。若分类具有内在排序（如：轻度/中度/重度），使用普通 Kappa 会丢失信息，此时应转向**加权 Kappa（Weighted Kappa）**或多面 Rasch 模型。

## 4. 模型与检验统计量

- **原始 Cronbach's $\alpha$ 与 标准化 $\alpha$**：
  - **原始 $\alpha$（基于协方差矩阵）**：$\alpha = \frac{k}{k-1}\left(1 - \frac{\sum \sigma_i^2}{\sigma_T^2}\right)$。适用于各题项方差/量纲相似的情况。
  - **标准化 $\alpha$（基于相关矩阵）**：当各题项方差差异极大（如不同量纲的分数简单相加）时，应报告基于标准化得分的 $\alpha$。
- **ICC 及其一致性类型（Consistency vs. Absolute Agreement）**：
  - **一致性（Consistency）**：忽略评定者之间的系统误差。*示例：评委A打1,2,3分，评委B打3,4,5分。两人对被试的排序完全一致，此时「一致性 ICC」极高（接近1）。*
  - **绝对一致性（Absolute Agreement）**：要求打分数值绝对相等。*在上述示例中，由于A和B的分数绝对值差异大，「绝对一致性 ICC」将非常低。*
- **Fleiss Kappa**：多评定者扩展版，公式基于实际观察到的一致率（$P_o$）与随机期望一致率（$P_e$）之差：$\kappa = \frac{P_o - P_e}{1 - P_e}$，提供渐近标准误（Asymptotic SE）。
- **Tukey 可加性（Tukey's Test for Nonadditivity）**：检验被试与题项间是否存在可乘交互作用。若存在显著交互，说明不同被试对各题项的反应模式存在系统异质性，此时用总分代表单一潜变量的做法受到质疑。

## 5. 假设与诊断

| 假设 | 检查方式 / 诊断工具 | 违背时的典型后果 |
| :--- | :--- | :--- |
| **单维性 ($\alpha$)** | 探索性/验证性因子分析 (EFA/CFA) | $\alpha$ 虚高，错误掩盖多维结构 |
| **本质 $\tau$ 等值** | CFA 模型比较（设定载荷相等检验） | $\alpha$ 低估了量表的真实内部一致性信度 |
| **无反向题干扰** | 检查“项之间：相关性”矩阵有无大量负值 | 导致极其微小或**负数的 $\alpha$** |
| **ICC 效应模型匹配** | 梳理实验设计（评定者是总体样本还是特定专家） | 选错 Shrout & Fleiss 模型，导致方差解释完全错误 |
| **Kappa 数据等级** | 确认类别是名义 (Nominal) 还是序数 (Ordinal) | 若类别为序数而误用普通 Kappa，低估真实一致性 |

## 6. 与相关方法的取舍（IRR 决策树与现代测量学趋势）

- **内部一致性：$\alpha$ vs. McDonald's $\omega$**
  - 当前心理测量学界强烈建议在 CFA 框架下使用 **McDonald's $\omega$（或分层 $\alpha$）** 替代单一的 $\alpha$。$\omega$ 放宽了严格的 $\tau$ 等值假设，对多维结构的估计更准确。若软件支持或已进行 CFA，优先报告 $\omega$。
- **评定者间信度（IRR）决策树**：
  - 评定结果为**连续数据** $\rightarrow$ **ICC（组内相关系数）**
  - 评定结果为**名义数据，2 名评定者** $\rightarrow$ **Cohen's Kappa**
  - 评定结果为**名义数据，$\ge 3$ 名评定者** $\rightarrow$ **Fleiss Kappa**
  - 评定结果为**序数数据** $\rightarrow$ **加权 Kappa (Weighted Kappa)**

## 7. 实现速查

### SPSS (RELIABILITY 命令)
SPSS 默认计算基于协方差的原始 $\alpha$。若需 ICC 或 Fleiss Kappa，在「统计」子对话框勾选（详见附录）。注意在处理前手动计算反向题。

### R 语言
推荐使用权威的 `psych` 包。
```r
library(psych)

# 1. Cronbach alpha (数据需为宽表：行=被试，列=题项)
# check.keys = TRUE 极其有用：会自动识别并处理反向计分的题项，防止负 alpha
psych::alpha(data.frame(i1, i2, i3, i4), check.keys = TRUE)

# 2. McDonald's omega
psych::omega(data.frame(i1, i2, i3, i4))

# 3. ICC (数据需为宽矩阵格式，列为不同评定者/测量次数)
psych::ICC(data.matrix(wide_raters))
```

### Python
推荐使用 `pingouin` 库，API 设计现代且严谨。
```python
import pingouin as pg
import pandas as pd

# 1. Cronbach alpha (宽表)
df = pd.DataFrame({"i1": [1,2,3], "i2": [2,2,4], "i3": [1,3,3]})
pg.cronbach_alpha(data=df)

# 2. ICC (注意：需要融化为长表格式 long_df)
# 参数明确要求指认 target (被试), rater (评委), ratings (得分)
pg.intraclass_corr(data=long_df, targets="subject", raters="rater", ratings="y")
```

## 8. 报告清单

写入论文「方法-统计分析」或「结果」小节时，应报告：

1. **基本信息**：量表名称、包含的题项数量（$k$）、处理缺失值后的**最终有效样本量 ($N$)**。
2. **$\alpha$ 报告**：
   - 报告具体的 $\alpha$ 点估计值（必要时说明是否使用了标准化 $\alpha$）。
   - **经验阈值（参考 Nunnally & Bernstein, 1994）**：通常认为 $\alpha > 0.70$ 为可接受，$\alpha > 0.80$ 为良好。若题数极少（如 $k<4$），$0.60$ 以上结合较好的平均题项相关性有时亦被接受。
   - 视需要报告“删除某项后的 $\alpha$”（用于说明为何在最终分析中剔除了某缺陷题项）。
   - **局限性声明**：若未做因子分析，可简要说明“本研究依据经典测量理论报告 $\alpha$，假设其满足单维性”。
3. **ICC 报告（若适用）**：
   - 必须报告所选的**模型类型**（如：Two-way mixed effects, 双向混合效应）、**定义方式**（Consistency 一致性 / Absolute Agreement 绝对一致性）。
   - 报告点估计值及 **95% 置信区间 (CI)**。
4. **Kappa 报告（若适用）**：报告具体的类别数、评定者数、$\kappa$ 估计值、标准误（SE）或 $p$ 值。

## 9. 参考文献与手册

- **经典教材**：Nunnally, J. C., & Bernstein, I. H. (1994). *Psychometric theory* (3rd ed.). McGraw-Hill.（$\alpha$ 系数阈值与测验理论标准引用）。
- **ICC 权威文献**：Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in assessing rater reliability. *Psychological bulletin*, 86(2), 420.
- **软件实现**：IBM SPSS Statistics Algorithms (RELIABILITY 章节)；R `psych` package documentation.

---

## 附录：SPSS「可靠性分析 $\rightarrow$ 统计」核心选项指南

为辅助理解 SPSS 繁杂/生硬的中文界面菜单，特作如下严谨对照与使用建议：

### 描述属性的统计 (Descriptives for)
| 选项 | 统计学含义与实务建议 |
| :--- | :--- |
| **项 (Item)** | 输出每个题项的均值和标准差。*建议勾选，用于检查数据录入范围与分布。* |
| **标度 (Scale)** | 输出把所有题项加总后的“总分”的均值、方差。 |
| **删除项后的标度 (Scale if item deleted)** | **核心诊断工具**。若删除某题后整体 $\alpha$ 显著跃升，提示该题拉低了内部一致性（可能是题意不清或反向计分错误）。*强烈建议勾选。* |

### 摘要与一致性度量
| 选项 | 统计学含义与实务建议 |
| :--- | :--- |
| **项之间：相关性** | 输出完整的题项间相关矩阵。*用于核查是否大量存在负相关（反向题预警）。* |
| **同类相关系数 (ICC)** | 开启 ICC 分析。勾选后需明确选择**模型**（单向随机 / 双向随机 / 双向混合）及**类型**（一致性 / 绝对一致性），详见本文第 4 节。 |
| **Fleiss Kappa** | 开启多评定者名义数据分析。SPSS 会提供**渐近显著性水平**及渐近标准误。 |

### ANOVA 表 (ANOVA Table)
*注：此处的 ANOVA 建立在把被试和题项当作因子的重复测量概念上，常规量表信度分析较少报告。*
| 选项 | 统计学含义与实务建议 |
| :--- | :--- |
| **F 检验 (F test)** | 连续变量的重复测量 F 检验。 |
| **傅莱德曼卡方 (Friedman $\chi^2$)** | **等级/序数变量**的非参数重复测量检验（K 个相关样本）。 |
| **柯克兰卡方 (Cochran $\chi^2$)** | **二分变量（0/1）**的非参数重复测量检验。 |

### 其他诊断
| 选项 | 统计学含义与实务建议 |
| :--- | :--- |
| **霍特林 $T^2$ (Hotelling's $T^2$)** | 多变量均值检验，原假设为所有题项的均值完全相等。 |
| **图基可加性检验 (Tukey's test of nonadditivity)** | 检验数据的可加性假设。若显著，说明仅仅把各题分数“相加”作为总分可能无法准确反映被试特征。 |
