"""统计检验模块 — 全参数统计学测试库

根据 SPSS 分析菜单结构组织，复刻核心统计过程。

══════════════════════════════════════════════════════════════════════
统计决策指南 — 你的数据长什么样，该用什么检验
══════════════════════════════════════════════════════════════════════

【第一步：看你的研究问题】

A. 描述一个变量长什么样
   → descriptive/ (Frequencies, Explore, P-P/Q-Q Plot)

B. 比较两组或多组的均值/中位数
   → inference/t_test     (两组, 参数)
   → inference/anova      (多组, 参数)
   → inference/nonparametric (两组或多组, 非参数, 不假定正态)

C. 检验两个分类变量是否相关
   → categorical/crosstabs (χ² 独立性检验)

D. 检验一个连续变量和多个自变量的关系
   → inference/regression_ols (OLS 线性回归)

E. 降维: 把很多变量浓缩成少数几个因子
   → multivariate/factor_analysis (EFA, 探索性因子分析)

F. 评估量表/问卷的可信度
   → multivariate/reliability (Cronbach's α, ICC)

G. 分析"事件发生的时间"（如流失、死亡、转化）
   → survival/survival (Kaplan-Meier, Cox 回归)

H. 分析时间序列的规律
   → survival/time_series (ACF, ADF, 谱分析)

I. 聚类 / 简单分类树
   → ml/classify_cluster (K-Means, 层次聚类, 决策树)

【第二步：看你的数据类型 — 关键岔路口】

因变量是连续的吗？
├── 是 → 正态吗？方差齐吗？
│   ├── 都满足 → 参数检验 (t-test, ANOVA, OLS)
│   └── 不满足 → 非参数检验 (Mann-Whitney, Kruskal-Wallis)
│       或 → Bootstrap CI (不做分布假设)
│
├── 是分类的（是/否, A/B/C）→ categorical/ 或 proportion/
│
├── 是"时间+是否发生事件" → survival/
│
└── 是时间序列（有先后顺序, N个时间点）→ time_series/

【第三步：参数 vs 非参数 — 核心哲学】

参数检验 (t, ANOVA, OLS):
  假定数据来自某个已知分布（通常是正态），你只需估计分布的参数（μ, σ）。
  优势：如果假定成立，统计效力最高（更容易检出真实差异）。
  劣势：假定不成立时，结论不可靠。

非参数检验 (Mann-Whitney, Kruskal-Wallis, Wilcoxon):
  不对数据分布做假定。方法是对数据排序后用"秩"（rank）代替原始值做推断。
  优势：稳健，离群值影响小，适用范围广。
  劣势：统计效力略低（真实差异需要更大才能被检出）。
  关键直觉：非参数检验不是在比较均值！M-W 比较的是"随机从A组抽一个值
  比B组大的概率"，K-W 比较的是"各组的中位数是否来自同一分布"。

Bootstrap:
  通过反复有放回重抽样来估计统计量的抽样分布。不做任何分布假设。
  适用于任何统计量（均值、中位数、相关系数等）。

【第四步：多重比较问题 — 为什么需要校正】

当你做多次假设检验时（如 ANOVA 后比较所有组对），犯第一类错误
（假阳性）的概率会累积。

Bonferroni 校正：把 α 除以比较次数。保守（容易漏掉真实差异）。
Tukey HSD：专门为"所有两两比较"设计的校正，比 Bonferroni 更不保守。
Games-Howell：方差不齐时的 Tukey 替代品。
Dunnett：只需要和对照组比较时用（不是所有两两）。

【第五步：效应量 — p值不是一切】

p < 0.05 只告诉你"差异不太可能由随机产生"，不告诉你"差异有多大"。
大样本下微小的差异也能 p<0.05，但它可能毫无实际意义。

Cohen's d: 均值差异 / 合并标准差。0.2=小, 0.5=中, 0.8=大。
η² (eta-squared): ANOVA 中因子解释的方差比例。
Cramer's V: 分类变量关联强度。
Kendall's W: Friedman 检验中评定者一致性程度。

总是同时报告 p 值和效应量。

目录结构:
    descriptive/    — 描述统计 (Frequencies, Explore, P-P/Q-Q Plot, Ratio)
    inference/      — 推断统计 (t 检验, 非参数, 回归, 贝叶斯)
    multivariate/   — 多变量分析 (因子分析, 判别分析, 信度分析)
    categorical/    — 分类数据 (交叉表, 卡方检验)
    survival/       — 生存分析 (KM, Cox)
    ml/             — 机器学习分类 (聚类, 树模型)
"""

__all__ = [
    "descriptive",
    "inference",
    "multivariate",
    "categorical",
    "survival",
    "ml",
]
