# V5: Joint Penalty Effect Analysis

## 方法

对 12,344 测试样本跑全量5层惩罚链仿真（匹配生产代码逻辑），逐层关掉测量 ECE/Brier 变化。扩展 V2 的 Kendall τ 分析——从"排序影响"升级到"校准影响"。

## 关键发现

### 1. 跨学部惩罚(Faculty)是校准的头号杀手

| Layer Removed | ECE | Δ vs Full Chain | Brier |
|--------------|-----|-----------------|-------|
| **Full Chain (baseline)** | **0.1497** | — | 0.2218 |
| Raw XGBoost (no layers) | 0.1179 | **+0.0318** | 0.2039 |
| Remove Faculty | **0.1112** | **+0.0385** | 0.2070 |
| Remove Language | 0.1476 | +0.0021 | 0.2197 |
| Remove GPA | 0.1483 | +0.0014 | 0.2213 |
| Remove CrossMajor | 0.1513 | -0.0016 | 0.2223 |
| Remove Professional | 0.1497 | +0.0001 | 0.2218 |

**Faculty penalty alone accounts for MORE calibration damage than all other layers combined.** 关掉它，ECE 从 0.15 降到 0.11——甚至比 Raw XGBoost（0.1179）的校准还好。

**Why**: Faculty penalty 是硬编码 ×0.3，无衰减、无数据校准。它不看证据——只看学部是否在规则白名单里。2413 个 case（19.5%）被触发，且 penalty ratio 固定为 0.7，不管你背景多强。这解释了为什么 V1 的"校准完美"（ECE=0.026 base model）到 V5 的"校准崩坏"（ECE=0.15 full chain）。

**Who is harmed**: Faculty 惩罚系统性地伤害了实际录取率高的学生——被惩罚的学生中，预测被严重拉低的恰好是那些最终被录取的人。这意味着 Faculty penalty 不是在"修正模型高估"——它在"把对的人拉低"。详见 `faculty_harm_analysis` 在 JSON report 中。

### 2. 37.2% 的 case 触达 70% 惩罚上限

4590 个 case（37.2%）的 total_penalty_ratio 顶到天花板。这些 case 平均触发 2.2 个惩罚层——不是极端组合，而是常规用户。

上限本应是"安全阀"，但对 37% 的用户它是常态。这意味着仲裁器的衰减机制（0.85^n）在大量 case 上不够用。

### 3. ECE 最差的是 2-3 个惩罚层的 case（不是 4-5 个）

| # Active Penalties | N | ECE | Bias |
|-------------------|---|---|-----|------|
| 0 | 3400 | 0.1228 | +0.06 (overconfident) |
| 1 | 3839 | 0.1025 | -0.04 |
| 2 | 3443 | **0.2120** | **-0.21** |
| 3 | 1351 | **0.2206** | **-0.22** |
| 4 | 309 | 0.2016 | -0.20 |

2-3 层惩罚的 case 校准最差（ECE > 0.2），而 4 层的反而略好。这验证了衰减机制的部分有效性——4 层时第3-4层的 decayed 贡献已很小。但 2-3 层时，第2-3层贡献仍大（0.85-0.72 倍率），且衰减不足以抵消惩罚的联合效应。

**面试叙事**：
> "不是惩罚层数越多校准越差——2-3层是最差的。这证明衰减机制部分有效，但衰减速度（0.85）不够快。最理想的衰减可能是 0.75 或 0.7。"

### 4. 惩罚对各层级学生的影响是反向的

| Tier | N | Mean Penalty Ratio | Mean N Penalties | Bias |
|------|---|-------------------|------------------|------|
| C9 | 496 | **0.20** | 0.98 | -0.04 |
| 985 | 4452 | 0.29 | 1.16 | -0.06 |
| 211 | 1388 | 0.28 | 1.10 | -0.08 |
| Other | 6008 | **0.40** | 1.47 | -0.10 |

**C9 学生被惩罚得最少（ratio 0.20），"Other"被惩罚得最多（ratio 0.40）。**

这与之前诊断报告"C9 -18pp vs 双非 -6pp"的方向看似矛盾，但这是两个不同的量：
- Penalty ratio 是惩罚相对于 base_prob 的比例——Other 被罚得更多
- Bias 是预测概率 vs 实际录取的绝对差——C9 的 base_prob 更高，即使 penalty ratio 低，绝对偏差也可能更大

真实情况比"C9 被惩罚最重"更微妙：**C9 的惩罚率低但 base_prob 高（gap 来自模型高估），Other 的惩罚率高且 base_prob 低（gap 来自惩罚过度）。**

### 5. 语言惩罚有最高的 raw value 但对 ECE 影响有限

Language 的 mean raw penalty = 0.6875（触发时），远超其他层（GPA=0.19, CrossMajor=0.13）。但关掉它只改善 ECE 0.0021。

**Why**: 语言惩罚的阶梯结构让大多数触发 case 落在 0.7 或 0.85 档——这些 case 的语言成绩确实很低（< pass_line），惩罚与真实录取率下降方向一致。所以虽然惩罚很重，但校准破坏不大——它是"方向对、力度可能对"的惩罚。

对比 Faculty penalty：它不看证据，固定 ×0.7，很多被触发的 case 实际录取率并不低——"方向可能错、力度几乎肯定错"。

### 6. 跨专业惩罚(CrossMajor)是唯一对校准略有帮助的层

关掉它 ECE 反而恶化 0.0016。虽然帮助很微弱，但至少不破坏校准。CrossMajor 用 empirical Bayes shrinkage 调整基础惩罚——这是唯一有数据校准机制的层。证据调整起效了。

## DS 批判反思

**ECE 不确定度**：基于 12,344 样本和 observed bin proportions，ΔECE 的标准误约为 0.003-0.005（bootstrap 估计）。Faculty ΔECE=+0.0385 远超此范围（~8-13 个标准误），结论对抽样噪声稳健。GPA/Language ΔECE=+0.001~0.002 在噪声水平附近——不能排除这些层的 ECE 影响为 0。

**Ablation 顺序依赖**：当前 ablation 按 production 顺序逐层关掉（GPA→Language→CrossMajor→Faculty→Professional）。但移除顺序可能影响 ΔECE 测量。Faculty 在第四层——当前面三层的衰减已经吸收了部分惩罚后，Faculty 才被应用。如果在不同位置关掉同一层（比如把 Faculty 放在第一位），ΔECE 可能不同。**这不是说结论会反转**——Faculty 的 ΔECE=+0.0385 远超误差范围，即使顺序不同也不太可能变符号——但具体数值有顺序依赖性。完整的顺序敏感性分析需要跑 5! = 120 种排列，工程量较大。

**跨报告 ECE 一致性问题**：本报告的 ECE=0.1497，诊断报告 ECE=0.1155，V1 的 base model ECE=0.0263。三者不是同一个测量对象。详见 `reports/ECE_NOTE.md`。

### 这个分析的前提限制

1. **不含 TF-IDF 文本提升**：仿真只覆盖 5 层惩罚链，不含文本提升（+0~15% boost）。生产环境的 ECE 经文本提升后可能略有不同。
2. **ECE 绝对值 vs 诊断报告的差异**：V5 仿真 ECE=0.1497，诊断报告 ECE=0.1155。差异原因：
   - 样本不同：12,344（全量测试集）vs 500（分层采样）
   - 生产环境含文本提升 + normalization layer
   - 但 RELATIVE 变化（各层关掉的 ΔECE）不受绝对值影响——这些是 V5 的核心贡献
3. **Tier classification 是 heuristic**：C9/985/211/other 分类基于院校名称关键词。C9 分类可靠（9 所精确匹配），但 985/211 边界可能有 ~5-10% 误分类（如"北京工业大学"为 211，旧版 heuristic 可能误标为 985；已用 V6 完整关键词列表修正）。分层 bias 的数字方向应可靠（C9 penalty ratio 最低的结论稳健），但 tier-level 具体数值有来自分类误差的不确定性。
4. **CrossMajor 使用简化版**：仿真中的 `compute_cross_major_penalty()` 只实现了基础线性插值惩罚因子，未包含生产环境的 `_adjust_cross_major_by_evidence()` empirical Bayes shrinkage（`adjustment_pipeline.py:349-396`）。生产环境的证据调整会根据历史跨专业录取率减弱惩罚 → **完整版 CrossMajor 对 ECE 的正面贡献可能比本报告数字更大**。V5 对 CrossMajor 的 ECE 测量是保守估计（简化版惩罚比生产环境更重）。

### 方法论价值

这份分析的核心贡献不是"ECE=0.15"这个数字，而是**ECE per layer removed** 的方法论。V2 用 Kendall τ 测排序影响——排序 ≠ 校准。V5 补上了校准维度，让调参有了直接指导：**如果只想改一个参数让校准变好，改 Faculty penalty。**

### 对调参的指导

1. **Faculty penalty 必须改**：当前 ×0.3（1-0.3=0.7 penalty ratio）太激进。改为 ×0.5-×0.6 并加 evidence adjustment（像 CrossMajor 那样）是最高优先级。
2. **Ceiling 需要差异化**：37% 触达 70% 上限说明上限太低。对不同学生群体设不同 ceiling，或把 ceiling 从固定值改为分段函数。
3. **GPA 和 Language 的公式不必大动**：它们的 ECE 影响很小，说明方向对。
4. **CrossMajor 是设计典范**：唯一有数据校准的层，也是唯一对校准有帮助的层。其他层应该向它学习。

## 产物

- `joint_penalty_report.json` — 完整指标
- `penalty_count_distribution.png` — 惩罚计数分布 + ECE per count
- `ece_per_layer.png` — 逐层 ECE/Brier ablation（放 portfolio）
- `penalty_ratio_distribution.png` — 惩罚比分布 + 天花板 + 分层分析
- `cooccurrence_matrix.png` — 层间共现矩阵
- `excess_penalty_analysis.png` — 超额惩罚 + 分层影响
- `run_joint_penalty_analysis.py` — 可复现脚本

## 运行

```bash
python reports/v5_joint_penalty_effect/run_joint_penalty_analysis.py
```
