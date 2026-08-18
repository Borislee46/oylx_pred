---
source: analysis-library/cards/01-inference/compare-means-menu-spss-overview.md
improve_fingerprint: 8fabaf28c0a996f425eae3b4994653b05a7d1fbab69e10dc9e0988f695d2eac4
prompt_digest: bbf83924da02e3b0
cached: false
---

# 比较平均值（Compare Means）— 菜单族总览

> **Slug**: `compare-means-menu-spss`
> **Aliases**: 均值比较、ONEWAY、T-TEST 菜单入口、独立样本 T 检验（注：早期 OCR 扫描件中常被误识别为“独立样本工检验..山”）
> **主菜单路径**：**分析 → 比较平均值**（Analyze → Compare Means）

## 1. 核心目的与适用场景

这一菜单族集中处理基础的组间差异推断，主要回答以下问题：
1. **均值推断**：单组均值是否等于特定假设值（单样本 t）；两组独立或配对样本均值是否存在差异（独立/配对 t）；三组及以上独立样本均值是否全等（单因素 ANOVA）。
2. **比例推断**：样本比例是否等于特定值，或多组间的二项分类比例是否存在显著差异（注：*比例推断相关菜单需 SPSS 27 及更高版本支持；旧版本需通过“非参数检验”或“描述统计 → 交叉表”实现*）。
3. **描述与探索**：通过 **Means (平均值)** 过程，快速按组别分层输出描述性统计量，并可在此基础上附加基础的方差分析表或线性检验。

## 2. 数据与设计前提

在进行任何均值或比例比较前，需确认研究设计符合以下逻辑前提：
- **独立性前提（针对独立样本 t / 单因素 ANOVA）**：不同组别的数据（或同一组内的不同观测）在抽样和误差上必须是相互独立的。如果数据来自同一群体的不同时间点或存在匹配关系，必须走**配对/重复测量**逻辑。
- **变量类型**：响应变量（因变量）应为连续变量（区间或等比测度）；对于比例检验，响应变量必须明确定义为二分变量（成功/失败）。
- **协变量控制**：本菜单族**无法**处理协变量。若需控制连续混杂因素，应转向一般线性模型（GLM）。

## 3. 统计假设与诊断

均值推断的经典参数方法依赖特定分布假设。违反假设时，需报告并采取相应的稳健策略：

| 假设 | 检查对象与方法 | 违背时的典型后果 | 补救或替代方案 |
|------|----------------|------------------|----------------|
| **正态性** | **t 检验**：各组内原始数据的正态性<br>**ANOVA**：模型残差的正态性<br>*(工具：Q-Q 图、Shapiro-Wilk 检验)* | 小样本下 I 类错误率膨胀或统计功效下降；大样本下受中心极限定理保护，具有一定稳健性。 | 采用非参数检验，或基于自举法（Bootstrapping）推断。 |
| **方差齐性** | 各组响应变量的方差是否一致<br>*(工具：Levene 检验、Brown-Forsythe 检验)* | $t$ 或 $F$ 统计量的名义显著性水平偏离（尤其在各组样本量严重不均衡时）。 | 必须改用 **Welch** 修正的 $t$ 检验或 Welch ANOVA。 |
| **大样本比例近似** | $np \ge 5$ 且 $n(1-p) \ge 5$（经验法则） | 渐近正态置信区间严重失准。 | 改用精确二项检验（Exact Binomial Test）。 |

## 4. 模型与检验统计量（含自由度逻辑）

本节列出该菜单族底层调用的核心统计量及其自由度（$df$）逻辑：

*   **单样本 t 检验**：
    *   $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$，自由度 $df = n - 1$。
*   **独立样本 t 检验**：
    *   **方差齐性时（Student's t）**：使用合并方差（Pooled Variance），$df = n_1 + n_2 - 2$。
    *   **方差不齐时（Welch's t）**：不合并方差，采用 **Welch–Satterthwaite** 近似公式计算带小数的校正自由度。
*   **单因素 ANOVA**：
    *   **经典 F 检验**：$F = MS_B / MS_W$（组间均方 / 组内均方），$df_1 = k - 1$，$df_2 = N - k$。
    *   **Welch ANOVA**：根据各组方差和样本量对 $F$ 值和分母自由度进行动态惩罚与调整。
*   **比例检验**：
    *   依据软件实现版本，可能报告渐近 $z$ 统计量、Pearson $\chi^2$ 或精确二项概率。

## 5. 效应量与置信区间

仅报告 $p$ 值是不充分的，必须配合效应量（Effect Size）与置信区间（CI）评估实际意义：
- **两组均值比较**：最常用 **Cohen's $d$**（标准化均值差）或 Hedges' $g$（小样本校正）。同时报告原始**均值差的 95% 置信区间**。
- **单因素 ANOVA**：输出 $\eta^2$（Eta-squared）。*(注：在单因素方差分析的数学设定中，$\eta^2$ 与偏 $\eta^2$ [Partial $\eta^2$] 是完全等价的，无需因软件不同标签而困惑)*。
- **比例推断**：根据具体对比方式，报告**率差（Risk Difference）**、**相对危险度（RR）**或**比值比（OR）**及其 CI。

## 6. 常见误区与滥用预警

- **反复多次使用两两 t 检验替代 ANOVA**：这是极其严重的错误。对 4 个组进行 6 次两两 $t$ 检验，会导致**族错误率膨胀（Family-wise Error Rate, FWER inflation）**。必须先进行整体 ANOVA，显著后再配合适当的**事后多重比较**（如 Tukey HSD, Bonferroni）。
- **多因子或重复测量错用单因素 ANOVA**：若实验涉及“时间点”或“多重干预交叉”，强制拆分为独立单因素会破坏残差结构，导致标准误（SE）计算错误，必须使用**混合线性模型**或**重复测量 ANOVA**。
- **观察性研究中的因果倒置**：均值存在显著差异仅代表数学上的关联。除非实验设计满足随机分配与严格控制，否则不可在结论中将其描述为“A 导致了 B 的均值变化”。

## 7. 与相关统计方法的取舍

- **一般线性模型（GLM 单变量）**：当需要加入协变量（ANCOVA）、探讨多因子交互作用（Interaction），或需要更复杂的模型设定时，应放弃 `Compare Means` 菜单，转向 `GLM`。
- **非参数替代检验**：当数据严重偏态或存在无法剔除的极端离群值，且研究者更关心“位置秩（Rank）”而非“算术均值”时：
  - 两组独立样本 $\rightarrow$ **Mann-Whitney U 检验**
  - 多组独立样本 $\rightarrow$ **Kruskal-Wallis H 检验**
  - 两组配对样本 $\rightarrow$ **Wilcoxon 符号秩检验**
- **回归编码的等价性**：在经典设定下，两组独立样本 $t$ 检验，与将组别编码为哑变量（0/1）并放入普通最小二乘法（OLS）回归模型中检验斜率，结果是完全等价的。

## 8. 软件实现速查

*   **IBM SPSS**：各子对话框详见下方附录。
*   **R 语言**：
    *   基础推断：`t.test(y ~ x)`，`oneway.test(y ~ x)`（默认开启 Welch 修正），`prop.test()`。
    *   多重比较：基础包 `pairwise.t.test` 功能有限且易出错；强烈建议使用现代包 **`emmeans`** 计算边际均值并进行严谨的事后比较。
*   **Python**：
    *   基础库：`scipy.stats.ttest_ind`、`scipy.stats.ttest_rel`（配对）、`scipy.stats.f_oneway`。
    *   **推荐库**：强烈推荐使用 **`pingouin`**（如 `pingouin.ttest` / `pingouin.anova`），其输出包含了效应量、自由度、置信区间和统计效能，格式最贴近学术发表需求与 SPSS 体验。

## 9. 标准报告清单 (APA 格式导向)

在撰写学术论文“结果”部分时，应确保包含以下要素：
1. **基础描述统计**：各组的样本量（$N$）、均值（$M$）、标准差（$SD$）。
2. **假设检验陈述**：简述正态性检验结果（以图形诊断为主）。**方差齐性**：默认使用 Welch $t$ 检验 / Welch ANOVA，Levene 检验仅作为描述性诊断参考，不作为方法选择的决策依据。
3. **推断统计量**：准确报告检验统计量、自由度。
4. **P 值规范**：报告确切的 $p$ 值（如 $p = 0.032$），除非 $p < .001$（不可写为 $p = 0.000$）。
5. **效应量与置信区间**：注明置信水平与括号格式，如：“两组均值存在显著差异, $t(45) = 2.34$, $p = 0.024$, Cohen's $d = 0.52$, 95% CI [0.12, 1.34]”。

## 10. 参考文献与参考手册

- **经典教材**：Maxwell, S. E., Delaney, H. D., & Kelley, K. (2017). *Designing experiments and analyzing data: A model comparison perspective*. Routledge. （实验设计与方差分析标准参考）。
- **软件手册**：IBM SPSS Statistics Base Manual (Release 27+). 重点参考 `T-TEST`, `ONEWAY`, `MEANS` 及 `PROPORTIONS` 算法文档。

---

## 附录：比较平均值菜单子项一览

| 子项 (SPSS) | 核心功能 | 本库关联卡片索引 |
|------|------|----------|
| **平均值 (Means)** | 按分类变量分层输出 M/SD，可选附加简单 ANOVA 表与线性趋势检验。 | [compare-means-options-posthoc-proportions-spss.md](compare-means-options-posthoc-proportions-spss.md) §1 |
| **单样本 T 检验** | 检验单一连续变量的均值是否等于用户指定的检验值。 | 待补 |
| **独立样本 T 检验** | 检验分类变量（仅两水平）划分的两组独立连续数据均值差。 | [independent-samples-t.md](independent-samples-t.md) |
| **摘要独立样本 T 检验** | 无需原始数据，仅需输入 N、M、SD 即可进行独立 t 推断。 | 待补 |
| **成对样本 T 检验** | 检验配对设计（如干预前后）的数据差值均值是否显著异于 0。 | 待补 |
| **单因素 ANOVA** | 检验单一分类因子（多水平）对连续变量的影响，支持事后多重比较。 | [compare-means-options-posthoc-proportions-spss.md](compare-means-options-posthoc-proportions-spss.md) §2–3 |
| **比例菜单 (单/独立/配对)** | *（SPSS 27+ 引入）* 处理二项数据的率检验、差异估计及对应置信区间。 | [compare-means-options-posthoc-proportions-spss.md](compare-means-options-posthoc-proportions-spss.md) §4–9 |
