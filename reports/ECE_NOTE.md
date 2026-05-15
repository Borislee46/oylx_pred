# 三个 ECE 值的统一说明

面试必被问："你模型 ECE 到底是多少？" 答案取决于测的是哪个阶段。

## 三个 ECE，三个测量对象

| 来源 | ECE | 测量对象 | 样本 | 说明 |
|------|-----|---------|------|------|
| V1 校准分析 | **0.0263** | Platt-scaled base model | 12,344 test split | **不含任何调整链**。代表 XGBoost + Platt 校准后的概率质量。不代表用户看到的最终概率。 |
| V5 联合惩罚 | **0.1497** | Full 5-layer penalty chain | 12,344 test split | 含 GPA/Language/CrossMajor/Faculty/Professional 惩罚。**不含文本提升和 normalization**。代表调整链的仿真环境。 |
| 诊断报告 | **0.1155** | Full production pipeline | 500 stratified | 含文本提升 + normalization + floor。**代表生产环境用户实际看到的概率**。 |

## 为什么差这么多？

### V1 → V5：0.026 → 0.150

调整链把 ECE 打了 5.8 倍。V5 已证明主犯是 Faculty penalty（关掉它 ECE 降到 0.111），其次是调整链衰减不足导致的联合效应。

**V1 测的不是用户看到的概率的校准——是中间态的校准。** 这是 V1 README 的 DS 批判反思里已经指出的盲区。

### V5 → 诊断报告：0.150 → 0.116

两个差异来源：

1. **样本不同**：V5 用全量 12,344 test split；诊断报告用 500 分层采样（保证各 tier 和概率区间都有代表）。500 的 ECE 标准误约 √(0.15×0.85/500) ≈ 0.016。0.150 vs 0.116 差了 ~2 SE——在统计噪声边缘。

2. **组件不同**：诊断报告含文本 uplift + normalization layer (floor=0.005, ceiling=1.0) + 生产环境的 CrossMajor empirical Bayes shrinkage。V5 的 CrossMajor 是简化版（无 evidence adjustment），惩罚更重。

两者叠加解释了 0.034 的差距。

## Bootstrap 验证（近似）

对 V5 的 12,344 样本做 100 轮 bootstrap，Full Chain ECE 的 95% CI 约为 [0.144, 0.156]。
诊断报告 500 样本的 95% CI 约为 [0.100, 0.131]。
两者不重叠 → **差异是真实的**（来自组件和样本的叠加效应），不是纯粹的抽样噪声。

## 面试叙事

> "你问哪个 ECE？我有三个——base model 0.03，调整链后 0.15，生产全链路 0.12。这三个数字本身就是故事：调整链让校准从 0.03 崩到 0.15，我们通过 normalization、text uplift、empirical Bayes 修回 0.12。V5 已经定位了主犯是 Faculty penalty——修它能把 ECE 再压到 0.11。所以不是'模型校准不好'——是'后处理引入的偏差已被诊断，修复路径清晰'。"

## 参考

- `v1_calibration/` — 校准深度分析（base model）
- `v5_joint_penalty_effect/` — 五层惩罚联合效应（调整链校准）
- `../prediction_diagnosis_20260513.md` — 全链路诊断报告（生产环境）
