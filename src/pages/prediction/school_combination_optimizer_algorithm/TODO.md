<!-- !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files. -->
# Optimizer TODO — 验证、重构、回归路线

> 当前状态：代码已从 GitLab 历史中恢复（`faa1d83^`，约 2025-11-11 版本），未被 merge 到 master。
> 下一步不是直接加回代码，而是先用实验验证核心假设，再决定回归形式。

---

## 一、为什么删掉的

| 问题 | 具体表现 |
|------|---------|
| Over-engineering | NSGA-III 453 行，但候选池通常 30-50 个学校，穷举+贪心就能逼近 |
| 验证缺失 | 没有对比过 NSGA-III vs 规则方案 vs 顾问人工方案的差距 |
| 维护成本 | optimizer 子模块 10 文件，改一个阈值要追 3 层调用链 |
| 使用率不明 | 不知道顾问是点"智能优化"还是手动勾选 |

**但核心思路没废**：多目标 trade-off、自适应阈值、相关性模拟、平衡生成——这四个才是真正的领域知识，NSGA-III 只是其中一个实现方式。

---

## 二、验证实验（回归前必须回答的问题）

### 实验 1：NSGA-III 到底比规则方案好多少？

**问题**：NSGA-III 的 Pareto 前沿解 vs 规则平衡生成（30/40/30），在 5 个 objective 上各差多少？

**方法**：
- 从 `cases.feather` 随机抽样 20 个学生背景
- 每人跑 NSGA-III + 规则 fallback，各产出 3 套方案
- 对比指标：全拒率 Δ、多样性 Δ、balance_score Δ、major_similarity Δ

**判断标准**：如果所有维度差距 < 5%，NSGA-III 不值得保留。如果某个维度差距 > 20%，定位是哪个 objective/constraint 贡献的。

**建议文件**：`experiments/optimizer/exp1_nsga3_vs_rules.py`

---

### 实验 2：自适应阈值 vs 固定 0.55/0.75

**问题**：自适应阈值（根据学生背景算分位点）和固定阈值，产出的方案差异有多大？

**方法**：
- 对比两组学生：985+GPA≥3.2 vs 双非+GPA<3.0
- 每组分别用自适应阈值和固定阈值生成方案
- 对比：冲/稳/保数量、学校档次分布、全拒率

**判断标准**：自适应阈值对高背景学生应该给出更 aggressive 的方案（更多冲刺），对普通背景应该更保守。如果两组差异不大，则自适应阈值没增量价值。

**建议文件**：`experiments/optimizer/exp2_adaptive_thresholds.py`

---

### 实验 3：Monte Carlo 相关性模拟到底纠正了多少偏差？

**问题**：独立假设（全拒率 = ∏(1-pᵢ)）vs Cholesky 相关性模拟，全拒率差多少？

**方法**：
- 取 correlation_matrix，对每个 (学校A, 学校B) 组合看相关系数分布
- 选 10 组典型的选校组合（高相关组 vs 低相关组）
- 对比独立假设 vs Sobol MC 的全拒率

**判断标准**：
- 如果差距 < 1pp → Monte Carlo 可以砍，独立假设足够
- 如果差距 1-3pp → 保留，但可以降级为简单的 multivariate normal（不用 Sobol）
- 如果差距 > 3pp → 必须保留，这是面试中的 killer feature

**建议文件**：`experiments/optimizer/exp3_monte_carlo_correlation.py`

---

### 实验 4：顾问真的在用吗？（需要等回归上线后）

**问题**：手动选择 vs 智能优化，使用率各多少？优化方案被修改的比例？

**方法**：
- 埋点：每次点击"智能优化"记录 event
- 埋点：优化方案中顾问修改了哪些学校的勾选
- 对比：优化方案的全拒率 vs 顾问修改后的全拒率

**判断标准**：如果 < 20% 用户点"智能优化"，可能需要简化入口或改为默认行为。

---

## 三、回归时的可能形式

基于实验结果，有几种回归方案：

### 方案 A：轻量规则引擎（最可能）

砍掉 NSGA-III，保留：
```
过滤链（filters.py）          → 候选池预筛选
自适应阈值（threshold_calculator.py） → 动态冲/稳/保分界
平衡生成（school_selector.py） → 规则 fallback
指标计算（metrics_calculator.py） → 方案评分
可视化（visualizer.py）        → Streamlit 展示
```

预估代码量：~500 行（从 3500 行压缩），复杂度显著降低，效果损失 < 5%。

### 方案 B：保留 NSGA-III，但作为可选深度分析

只在顾问点击"深度分析"时才跑，日常展示规则方案。适合面试叙事：*"我做了两套方案——轻量版 500 行覆盖 95% 场景，深度版用 NSGA-III 做 Pareto 探索解决剩下的 5%。"*

### 方案 C：完全重写，目标导向

不从算法选型出发，从顾问的决策流程出发：
1. 顾问先定"激进程度"（保守/平衡/激进）→ 一个 slider
2. 系统根据激进程度调自适应阈值 → 候选池自动分层
3. 展示 1-2 套方案（不是 3 套），让顾问手动微调

这种方案的面试叙事更强：*"我把多目标优化问题化简为一个用户可控的偏好参数，而不是让用户从 Pareto 前沿里自己选。"*

---

## 四、和现有项目其他模块的配合

| 模块 | 当前状态 | 和 optimizer 的关系 |
|------|---------|-------------------|
| V1 校准分析 | 已设计，未跑 | optimizer 依赖校准后的概率 |
| V2 调整链 Ablation | 已设计，未跑 | 调整链修改了概率输入，会影响 optimizer 的冲/稳/保分界 |
| S1 Rolling Backtest | 已设计，未跑 | 可以验证 optimizer 方案的时间稳定性 |
| Counterfactual | `counterfactual.py` 已有 | optimizer + counterfactual = "如果 GPA+0.2 申请组合怎么变" |

**建议**：先跑 V1+V2，确保概率可靠，再跑 optimizer 实验 1-3。否则 optimizer 评估的是"不准确的概率上的最优方案"。

---

## 五、优先级排序

| 优先级 | 任务 | 原因 |
|--------|------|------|
| P0 | 实验 3（Monte Carlo 相关性） | 最快出结果，决定这个 feature 值不值得讲 |
| P0 | 实验 1（NSGA-III vs 规则） | 决定 optimizer 回归时的架构选型 |
| P1 | 实验 2（自适应阈值） | 产出领域知识证据，和 V1 互补 |
| P2 | 实验 4（用户行为） | 需要先上线才能跑 |
