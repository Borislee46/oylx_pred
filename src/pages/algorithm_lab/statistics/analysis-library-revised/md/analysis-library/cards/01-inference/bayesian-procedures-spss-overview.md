---
source: analysis-library/cards/01-inference/bayesian-procedures-spss-overview.md
improve_fingerprint: 87b352c2fa16b1c230b72919f839a0b8b9841a0789e311474b4e9195083597c6
prompt_digest: bbf83924da02e3b0
cached: false
---

# SPSS 贝叶斯分析 — 过程类型总览

- **slug**: bayesian-procedures-spss-overview
- **aliases**: BAYES、贝叶斯统计菜单、SPSS Bayesian
- **菜单**：**分析 → 贝叶斯统计**（SPSS v25 及以上版本引入，v26/v27 逐步扩充）

## 1. 核心推断目标

贝叶斯统计过程不再输出传统的 $p$ 值（即假设模型为真时观测到当前或更极端数据的概率），而是基于**贝叶斯定理**，结合**似然（Likelihood，由数据提供）**与**先验（Prior，由理论或默认设定提供）**，计算参数的**后验分布（Posterior distribution）**。

此类过程主要提供以下三大推断工具：
1. **贝叶斯因子（Bayes Factor, $BF$）**：量化数据对两个竞争假设（通常为 $H_1$ 备择假设 vs $H_0$ 原假设）的支持程度比值。
2. **可信区间（Credible Interval, CrI）**：给出参数在给定概率下（如 95%）的真实后验概率范围。
3. **后验概率（Posterior Probability）**：直接回答如“参数大于 0 的概率是多少”（如 $P(\theta > 0 | \text{data})$）。

涵盖模块包括：单样本正态/二项/泊松、配对/独立样本正态推断、皮尔逊相关、线性回归、单因素 ANOVA、对数线性模型等。

## 2. 常见误区与防范（易滥用点）

- **把 CrI 解释为 CI（置信区间）**：频率学派的 95% CI 意味着“若无限次重复抽样，95% 的区间会包含真值”；而贝叶斯 95% CrI 直接表示“在此次数据和给定先验下，参数真值落入该区间的概率为 95%”。二者哲学不同，不可混用术语。
- **BF 与 $p$ 值的强制换算**：二者数学基础不同，$p$ 值仅评估对原假设的偏离，而 $BF$ 显式比较两个模型。切忌用“$BF$ 等价于显著”来行文。
- **大样本下的 Lindley 悖论**：当样本量极大时，即使效应量微小，频率学派极易拒绝 $H_0$（$p < 0.05$）；但贝叶斯因子 $BF_{01}$ 可能反而会提供支持 $H_0$ 的强烈证据。
- **先验操纵（Prior-hacking）**：为追求支持假设的 $BF$ 值，探索性地反复更换先验参数。这是极严重的学术不端，等同于频率学派的 p-hacking。
- **版本未报告**：SPSS 在版本更迭中可能微调数值积分算法或默认先验，未提供具体版本号将导致分析**完全不可复现**。

## 3. 先验族与底层算法（SPSS 实现特性）

在阅读后续具体过程卡片前，需理解 SPSS 贝叶斯模块的基础设定逻辑。

### 3.1 两种先验路径
1. **无信息/客观先验（Objective/Non-informative Priors）**：当缺乏领域前置知识时使用，旨在让“数据自己说话”。例如：
   - **Haldane 先验 / Jeffreys 先验**（常用于二项/泊松比例推断）。
   - **JZS (Jeffreys-Zellner-Siow) 先验**：SPSS 在 t 检验和 ANOVA 中计算 $BF$ 常用的默认先验，它结合了对扰动参数的无信息先验和对效应参数的柯西分布先验，在理论上具有极佳的一致性。
2. **信息/共轭先验（Informative/Conjugate Priors）**：当有明确历史文献支持时使用。SPSS 允许用户输入自定义超参数（Hyperparameters，如指定正态分布均值与方差），此时输出结果高度依赖设定。

### 3.2 算法实现差异
并非所有贝叶斯分析都使用马尔可夫链蒙特卡洛（MCMC）。SPSS 的基础模块为了计算速度和稳定性，大量采用：
- **解析解（Analytical Solutions）**：当使用共轭先验时（如正态-正态），后验有闭式解，无抽样误差。
- **数值积分（Numerical Integration）**：如 **Gauss-Lobatto 积分**（多用于一维或低维积分，如计算边缘似然度以求 $BF$）。
- **MCMC（部分模块支持）**：若后验无解析解且维度较高，才回退至 MCMC。

## 4. 评估标准与效应量

### 4.1 贝叶斯因子 (Bayes Factor)
$BF_{10} = \frac{P(\text{data}|H_1)}{P(\text{data}|H_0)}$。SPSS 界面可能输出 $BF_{10}$ 或 $BF_{01}$（互为倒数）。
科研写作中通常参考 **Kass & Raftery (1995)** 或 **Lee & Wagenmakers (2013)** 的证据等级标准：

| $BF_{10}$ 值域 | $H_1$ 相对于 $H_0$ 的证据强度解释 |
|----------------|-----------------------------------|
| $< 1$          | 支持 $H_0$ (看具体数值，等价于 $BF_{01} > 1$) |
| 1 - 3          | 弱证据 (Anecdotal/Weak) |
| 3 - 10         | 中等证据 (Moderate) |
| 10 - 30        | 强证据 (Strong) |
| 30 - 100       | 极强证据 (Very Strong) |
| $> 100$        | 决定性证据 (Decisive) |

*(注：上述阈值仅为经验法则，非绝对截断值。)*

### 4.2 后验概率推导
结合先验概率（Prior Odds，常设为 1，即假设 $H_1$ 与 $H_0$ 发生概率先验相同），可得出：
$$\text{Posterior Odds} = BF_{10} \times \text{Prior Odds}$$

## 5. 假设与诊断

无论何种推断流派，数据的生成结构（如独立性、测量尺度）必须满足模型前提。

| 假设/设定 | 检查与诊断方式 | 违背时的典型后果 |
|---|---|---|
| **似然函数设定正确** (如残差正态) | 常规残差分析、Q-Q 图、后验预测检验 | 似然计算错误，推断基础崩塌 |
| **先验稳健性** | **敏感性分析（Robustness Check）**：换用更宽/更窄先验 | $BF$ 对先验设定高度敏感，结论不可靠 |
| **算法收敛** (仅适用 MCMC 时) | $\hat{R}$ 统计量趋近 1、迹线图 (Trace plot) 混合良好 | 得到的后验分布（CrI）完全错误 |

## 6. 软件实现与同类比较

### 6.1 替代与竞品软件
- **JASP（极度推荐）**：专为贝叶斯统计打造的开源 GUI 软件，界面交互极像 SPSS，但底层集成更为现代，默认输出图表极具发表级水准，是 SPSS 用户的最佳平替。
- **R**：`BayesFactor` 包（与 SPSS 的 $BF$ 算法逻辑最接近）、`brms` / `rstanarm`（基于 Stan，支持复杂广义线性及多层模型）。
- **Python**：`PyMC`、`NumPyro`（需手动编写概率图模型代码，灵活性最高）。

### 6.2 SPSS Syntax 示例
SPSS 的贝叶斯命令以独立的 `BAYES` 根命令起首，而非传统命令的子选项。例如：
- 独立样本 t 检验：`BAYES INDEPENDENT ...`
- 单因素方差分析：`BAYES ONEWAY ...`
- 线性回归：`BAYES REGRESSION ...`
（具体语法须随时查阅对应版本的 SPSS Command Syntax Reference）

## 7. 报告清单（科研发表标准）

在科研论文的方法或结果部分报告贝叶斯分析，**必须**包含以下要素以确保可复现性：
1. **软件版本**：明确注明 IBM SPSS Statistics 的具体版本号（如 v27.0.1）。
2. **似然模型**：说明数据分布的假设（如假设数据服从正态分布）。
3. **先验分布（核心）**：
   - 若用默认：说明所用默认类型（如 JZS 先验）。
   - 若为信息先验：明确列出所有超参数的值及来源文献。
4. **底层算法与种子**：报告使用的算法（如解析法、Gauss-Lobatto 数值积分等）；若使用 MCMC 或包含随机过程，必须报告**随机数种子（Seed）**及抽样次数。
5. **推断结果**：
   - 报告后验中心位置（后验均值或中位数）。
   - 报告 95% 可信区间（CrI），并避免口误为置信区间。
   - 报告 $BF_{10}$ 或 $BF_{01}$，并给出定性解释。
6. **敏感性分析**：对于关键研究假设，建议附上基于不同先验宽度（如 JZS $r=0.5, 0.707, 1.0$）的 $BF$ 变化图/表。

## 8. 参考文献

1. **Gelman, A., et al. (2013).** *Bayesian Data Analysis (3rd ed.)*. CRC Press. （贝叶斯推断圣经，理解方法论首选）
2. **Kass, R. E., & Raftery, A. E. (1995).** Bayes factors. *Journal of the American Statistical Association*, 90(430), 773-795. （$BF$ 解释强度的经典文献）
3. **Rouder, J. N., et al. (2009).** Bayesian t tests for accepting and rejecting the null hypothesis. *Psychonomic Bulletin & Review*, 16(2), 225-237. （JZS 先验的背景文献）
4. **IBM SPSS.** *IBM SPSS Statistics Base - Bayesian Statistics* (查阅对应版本的手册以获取精确算法文档)。

---

## 附录：SPSS 贝叶斯过程卡片链一览表

*(下表作为本库路由，映射至具体对话框与参数设定的操作卡片)*

| 过程类型 | 典型研究问题 | 对应本库卡片链接 |
|---|---|---|
| **单样本正态推断** | 正态似然下单样本均值/方差推断 | [bayesian-one-sample-normal-spss-dialog.md](bayesian-one-sample-normal-spss-dialog.md) |
| **独立样本正态推断** | 比较两个独立样本组的均值差异 | [bayesian-bf-and-priors-spss-dialogs.md](bayesian-bf-and-priors-spss-dialogs.md) §1 |
| **相关样本正态推断** | 配对样本的前后测均值差模型 | 待补 |
| **单因素 ANOVA** | 单因子多组均值差异的全局推断 | [bayesian-bf-and-priors-spss-dialogs.md](bayesian-bf-and-priors-spss-dialogs.md) §6 |
| **单因素重复测量 ANOVA** | 被试内因子的效应检验 | 待补 |
| **单样本二项式推断** | 二分类数据比例 $\theta$ 的推断 (Beta 先验) | 待补 |
| **单样本泊松推断** | 计数率/发生率推断 (Gamma 先验) | 待补 |
| **皮尔逊相关性** | 连续变量相关系数 $\rho$ 的后验与 $BF$ | [bayesian-bf-and-priors-spss-dialogs.md](bayesian-bf-and-priors-spss-dialogs.md) §2–3 |
| **线性回归** | 预测因子系数与方差 $\sigma^2$ 推断 | [bayesian-bf-and-priors-spss-dialogs.md](bayesian-bf-and-priors-spss-dialogs.md) §4–5 |
| **对数线性模型** | 多维列联表频数分布推断 | [bayesian-bf-and-priors-spss-dialogs.md](bayesian-bf-and-priors-spss-dialogs.md) §7 |
