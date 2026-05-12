---
source: analysis-library/cards/00-descriptive/ratio-statistics-spss-dialog.md
improve_fingerprint: df1e1aaddd7bd801b18fda03c7b33ad44950d37505f2dfa3868a4f0ea63aa7e9
prompt_digest: bbf83924da02e3b0
cached: false
---

# 比率统计（Ratio Statistics）

- **slug**: ratio-statistics-spss
- **aliases**: 分子分母比、比率研究、财产税评估分析、IAAO比率研究
- **菜单**：**分析 → 描述统计 → 比率**（Analyze → Descriptive Statistics → Ratio）
- **前提**：指定**分子**（通常为评估价值）与**分母**（通常为实际交易价格）；可选**分组变量**。

## 1. 回答什么问题

本模块并非普通的“除法计算器”，其核心算法与专有统计量专为**国际评估官协会（IAAO）的财产税和房地产批量评估标准**设计。在有明确分子与分母时，它用于评估**比值的水平（准确性）**与**均匀度（一致性）**。

- **通用场景**：描述任意有意义的比值（如支出/收入、病例/人口）的集中趋势与离散程度。
- **核心专业场景（批量评估）**：分子设定为**评估价值（Assessed Value）**，分母设定为**销售价格（Sale Price）**。回答两个核心业务问题：
  1. **整体水平是否准确？**（评估价通常是市价的百分之多少？）
  2. **评估是否公平一致？**（是否存在高价房产被低估、低价房产被高估的垂直不公平现象？同类房产的评估比率波动大吗？）

## 2. 不适用 / 易滥用

- **分母大量为 0 或缺失**：在实际交易数据中，价格为 0 通常意味着非正常交易或记录错误，直接计算会导致无意义的无穷大，须在分析前清洗或重新定义。
- **将比率的算术均值等同于「总体综合比率」**：比率的算术均值赋予了每个样本（无论价值高低）相同的权重；而代表总体的“加权均值”在比率研究中特指 $\frac{\sum \text{分子}}{\sum \text{分母}}$。两者不可混淆。
- **望文生义解读 COD/PRD**：本过程输出的 COD（离散系数）与 PRD（价格相关差异）具有特定于 IAAO 标准的统计学与业务含义，不可等同于其他学科中的同名缩写。

## 3. 数据与设计前提

比率研究的核心假设并非简单的独立同分布（i.i.d.），而是**样本对总体的代表性**：
- **正常交易假设（Arm's-length Transactions）**：样本应排除亲属间低价过户、法拍等非市场行为，否则分母（价格）不能代表公允市价。
- **样本代表性**：已售出的样本房产在特征分布上，应能代表未售出的所有房产。
- **口径一致性**：若使用加权个案，必须核对分子分母的统计口径与权重变量的逻辑自洽。

## 4. 模型与核心统计量（含公式定义）

本过程主要提供描述性统计量与基于渐近或自助法（Bootstrap）的置信区间。设分子为 $N_i$，分母为 $D_i$，单体比率为 $R_i = \frac{N_i}{D_i}$。

- **集中趋势**：
  - **中位比率（Median Ratio）**：比率分布的 50% 分位点，是 IAAO 推荐的首选集中趋势衡量指标（受极端值影响最小）。
  - **加权平均值（Weighted Mean）**：$\frac{\sum N_i}{\sum D_i}$。反映整体投资组合的价值比率。
- **一致性与公平性（离散度）**：
  - **AAD（平均绝对离差, Average Absolute Deviation）**：$\frac{1}{n} \sum |R_i - \text{Median}|$。
  - **COD（离散系数, Coefficient of Dispersion）**：$(\frac{\text{AAD}}{\text{Median}}) \times 100$。衡量**水平一致性**（横向公平）。
  - **PRD（价格相关差异, Price-Related Differential）**：$\frac{\text{Mean}}{\text{Weighted Mean}}$。衡量**垂直公平性**。

## 5. 假设与诊断

| 假设/前提 | 检查方式 / 诊断 | 违背时的典型后果 |
| :--- | :--- | :--- |
| **正常市场交易** | 审计异常值，剔除关联交易、法拍等 | 分母偏离真实价值，导致比率（$R_i$）失真 |
| **样本代表性** | 比较样本与总体在核心协变量（如房龄、面积）上的分布 | 选择性偏差，样本的中位数比率无法推论至总体 |
| **独立同分布（若报告 CI）** | 审查抽样设计或空间自相关性 | 置信区间（CI）覆盖率无效或错误估计精度 |

## 6. 效应量与行业标准阈值

撰写报告或评估模型时，通常比对 IAAO 规定的行业通用的标准阈值：

- **COD（离散系数）**：
  - **5% - 15%**：优秀至良好（较新或同质性高的房产群）。
  - **> 20%**：一致性差，评估模型需重新校准（或市场异质性极强）。
- **PRD（价格相关差异）**：
  - **理想范围**：**0.98 - 1.03**。
  - **> 1.03（累退）**：比率算术均值大于加权均值。说明**低价资产被相对高估**，高价资产被相对低估，存在严重的垂直不公平。
  - **< 0.98（累进）**：高价资产被高估。

## 7. 与相关方法的取舍

- **仅需符合 IAAO 标准的描述与合规审查**：使用本过程。
- **探究比率异质性的成因**：使用广义线性模型（GLM）、多层线性模型（HLM）或带有样条函数的回归，将比率或分子作为因变量，特征作为自变量。
- **常规数据的简单除法汇总**：如果不涉及 COD/PRD 分析，使用基础的“描述统计”或数据透视表即可。

## 8. 实现速查

- **SPSS**：`RATIO STATISTICS num_var WITH den_var /PRINT=MEDIAN MEAN WMEAN PRD COD.`
- **R (dplyr)**：
  ```R
  library(dplyr)
  df %>% 
    mutate(Ratio = Numerator / Denominator) %>%
    summarise(
      Median_Ratio = median(Ratio),
      Weighted_Mean = sum(Numerator) / sum(Denominator),
      Mean_Ratio = mean(Ratio),
      AAD = mean(abs(Ratio - Median_Ratio)),
      COD = (AAD / Median_Ratio) * 100,
      PRD = Mean_Ratio / Weighted_Mean
    )
  ```
- **Python (pandas)**：
  ```python
  ratio = df['Numerator'] / df['Denominator']
  median_ratio = ratio.median()
  weighted_mean = df['Numerator'].sum() / df['Denominator'].sum()
  
  COD = (ratio - median_ratio).abs().mean() / median_ratio * 100
  PRD = ratio.mean() / weighted_mean
  ```

## 9. 报告清单

报告中应当包含以下要素，以确保可复核性与合规性：
- **指标定义**：明确分子、分母的业务含义。
- **数据清洗说明**：缺失值、零分母记录的处理规则，以及非正常交易（Non-arm's-length）的剔除标准。
- **核心统计量**：中位数比率（水平）、COD（横向公平性）、PRD（垂直公平性）。
- **合规结论**：将得出的 COD 与 PRD 明确对照 IAAO 标准（如“PRD 为 1.05，存在累退性倾向”）。
- **不确定性（可选）**：若抽样计算，报告中位数与核心指标的置信区间（需写清置信级别与 Bootstrap 方法设置）。

## 10. 参考文献与手册

- **核心行业标准**：*International Association of Assessing Officers (IAAO). Standard on Ratio Studies.* (这是该统计方法的权威理论源头与阈值出处)。
- **软件实现细节**：IBM SPSS Statistics Algorithms / Syntax Reference: `RATIO STATISTICS`. (注意查阅具体版本的 Bootstrap CI 实现逻辑)。

---

## 附录：「统计」子对话框参数释义

*(注：此处已修正早期中文版 SPSS 中常见的 OCR 与汉化显示错误，如 AADMcOD、最大值凶 等，统一使用规范统计学术语)*

### 集中趋势 (Central Tendency)

| 选项 | 含义与用途 |
| :--- | :--- |
| **中位数 (Median)** | $R_i$ 的中位数，评估**整体水平**的首选指标。 |
| **平均值 (Mean)** | 算术平均值，受极端极值影响较大。 |
| **加权平均值 (Weighted Mean)** | $\sum \text{分子} / \sum \text{分母}$。反映价值加权后的总体综合比率。 |
| **置信区间 (Confidence Intervals)** | 提供上述集中趋势统计量的 CI（依软件版本可能需要勾选 Bootstrap 模块）。 |

### 离散 (Dispersion)

| 选项 | 含义与用途 |
| :--- | :--- |
| **AAD** | 平均绝对离差（Average Absolute Deviation）。 |
| **COD** | 离散系数（Coefficient of Dispersion），反映横向一致性。 |
| **PRD** | 价格相关差异（Price-Related Differential），反映垂直公平性。 |
| **中位数为中心的 COV** | 变异系数（Coefficient of Variation），基于中位数的相对标准差。 |
| **平均值为中心的 COV** | 变异系数，基于算术平均值的相对标准差。 |
| **标准差 / 范围 / 最小值 / 最大值** | 常规数据分布特征描述。 |

### 集中度 (Concentration)

用于评估有多大比例的个案落在了某个可接受的“合理比率区间”内。

| 选项类别 | 含义与用途 |
| :--- | :--- |
| **介于两个比例之间** | 指定绝对的**低比例界限**与**高比例界限**，计算落入该区间的个案百分比。 |
| **中位数百分比之内** | 指定一个 $\pm N\%$ 的浮动带（如落在中位数 $\pm 15\%$ 的区间内），计算符合该条件的个案百分比。支持添加多组规则。 |
