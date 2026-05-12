# WritingPrint 模块审视：从方法论到产品体验

> 2026-05-05 | 核心问题：AI detection → Turnitin 代理模型，以及最后一公里 UX 断层

---

## Part A：方法论——你的模型到底是什么

### A1. 事实盘点

| 指标 | 现状 |
|---|---|
| LOO-MAE | 23.3%（代码 verdict："Modest fit. Consider adding more samples"） |
| 样本 | 50 个（含 12 个 adversarial anchor） |
| 特征 | 7 维手工特征 → Ridge(alpha=2.0) |
| 有效特征 | 仅 3 个：AI keyword ratio(+15.3)、grammar error(+9.5)、avg_sent_len(+8.7)；其余 4 维系数在 ±2 量级 |
| 13 个 turnitin_low 样本 | highlight ratio 全部被外推为 score=0 写回 manifest，收编进 turnitin_real 训练集 |

### A2. 核心问题：这不是 Reverse Engineering

**实际做的事**：用 PyMuPDF 扣取 Turnitin PDF 里的青绿色高亮矩形 → 算出 `ai_ratio` × 100 → 当 ground truth → 用 7 个手工特征拟合。这是一个 **surrogate model（代理模型）**，不是 reverse engineering。

**应该叫**："Turnitin 代理模型" 或 "Writing quality regressor calibrated to Turnitin"，**不是** "AI detection"。

### A3. 面试场景的两个致命追问

1. **你怎么 validate 这个模型是在检测 AI，而不是在检测"Turnitin 觉得像 AI 的那种 surface 特征"？**
2. **highlight ratio→score 对低分段做了零外推，把 ground truth 在尾部钉死成 0%，这会把 grammar 和 keyword 系数往哪个方向偏？**

MAE 23% 看起来还行的真实原因：12 个 adversarial anchor（88% high_fp、10% low_fp、50% bad_grammar、15% tolerable_err）把 Ridge 回归线夹住了，中间那批 turnitin_real 在 LOO 时相当于在两个 anchor 之间被插值。删掉 12 个 adversarial 重跑，MAE 预估会跳到 30%+。

### A4. 三个标准的答案

| 标准 | 结论 |
|---|---|
| 内部产品（顾问看 4 档分桶） | **够用** — 顾问只需方向感，不需精确数字 |
| 给 PS 学生的可解释建议 | **更够用** — 价值在 deterministic 的规则匹配 + LLM 改写，不依赖模型精度 |
| 面试 ML 叙事 | **不够** — 方法论的循环性还没显式承认和处理 |

### A5. 面试升级方案（比加样本更便宜）

**把 framing 从"AI detection"改成"AI detector 的可解释代理"**，加两个实验：

1. **Ablation**：去掉 adversarial 训练，只用 turnitin_real 重跑，报 MAE 并分析两个数字的差距说明什么
2. **Second oracle / inter-annotator agreement**：用 GPTZero、ZeroGPT 或 OpenAI 旧版 classifier API 给同一批文档打分，看 7-feature 模型在多大程度上是 Turnitin-specific vs detector-agnostic

叙事升级：*"我训的不是 AI detector，我训的是 Turnitin 这个 detector 的解释代理；以下是它和 Turnitin 重合度多少、和别的 detector 重合度多少、surface 特征能解释 Turnitin 输出方差的多少。"*

### A6. 隐藏风险：Vendor-Side Concept Drift

Turnitin 的 AI detection 在 2024-2026 年间改了几代，50 个样本可能横跨不同 detector 版本，但目前没有 metadata 标注检测时间或模型版本。面试时被追问会很尴尬，但主动作为一个发现讲出来——"我发现训练样本可能横跨了 Turnitin 的两到三个 detector 版本，我用 X 方法检测了 drift 并隔离了哪几个样本"——反而是加分项。

---

## Part B：产品——最后一公里 UX 断层

### B1. 最大问题：local_fixes 被丢掉了

`engine.py` 每次 analyze 都生成：
- `local_fixes`：每条带 category、sentence_idx、action、impact、difficulty
- `global_fixes`、`quick_wins`

但 `ui.py` **一个都没渲染**。当前 UI 只有：
1. 一个 ring 显示 73%
2. verdict 一段话
3. 七条 feature contribution bar
4. Humanize 全文重写按钮

**这是比 50 样本和方法论循环性都更紧迫的问题**——ML pipeline 最高价值产物在最后一公里被丢掉了。

### B2. Score Ring：精确数字的幻觉

LOO-MAE 23% 意味着真值 50% 的文章，模型可能显示 27%（绿色）也可能显示 73%（红色）。顾问看到精确数字会本能地把它当真，产生无意义争论。

**建议**：ring 中心放 verdict label（"Mixed signals" / "Significant AI"），下面小字写 "approx. 73%, ±20%"。

### B3. 分数混合性问题

一个 50% 的输出有两种完全不同的解读：
- 低分 = 干净人写
- 中分 = 人写但语法差
- 高分 = AI 生成

顾问拿到 50% 后不知道是该"语法收一下"还是"文章被 AI 改过得审"，只能自己去 feature bar 上猜。

**建议**：拆成三个正交子分数：
- **AI 模式分**：ai_keyword + template + transition + uniform
- **语法分**：grammar_error 单独
- **节奏分**：CV + burstiness + long_sent

不需要重新训模型，在现有 7 维特征上做加权聚合，规则可以先 hand-tune。

### B4. Humanize 按钮 = Goodhart's Law 活生生上演

"全文 LLM 重写"把顾问的判断外包给另一个 LLM，文章风格被磨平，学生个人声音被削弱，score ring 还会因为 AI 关键词被替换显示更低的分——**工具优化的是分数，不是文章质量**。

**建议**：改成"句子级 alternative"——顾问选中一句话，工具给出 2-3 个改写建议（保守版/大胆版），顾问决定接不接受。把"工具决定改什么"翻转成"顾问决定改什么、工具提供选项"。

### B5. 未来数据飞轮

如果开始记录"顾问点了哪条 fix 的 accept/reject"，半年后就有真正属于自己的数据集——不是 Turnitin 的高亮代理，是顾问的专业判断。这才是值得 reverse engineer 的 oracle。

---

## 行动优先级（本周）

| # | 任务 | 工时 | 价值 |
|---|---|---|---|
| 1 | 把 `local_fixes` 渲染出来——原文按句子展开，问题句子高亮，点击展开建议 | 半天 | 最大产品价值 |
| 2 | Score ring 改成 bucket-first（verdict label 为主，数字 + 误差为辅） | 1-2 小时 | 提升顾问信任度 |
| 3 | 单一分数拆成 AI 模式 / 语法 / 节奏三个子分数 | 半天 | 消除分数歧义 |
| 4 | Humanize 从全文按钮改成句子级建议 | 1 天（可延后） | 翻转工具定位 |

前三件事做完后，50 样本和 MAE 23% 对产品体验的影响会被大幅稀释——因为顾问在乎的不再是分数本身，而是具体改写建议、语法 issue、节奏分析。这些信息的可信度大部分来自规则匹配（template regex、AI 关键词词典、长度统计），不依赖那个噪声很大的 Ridge 输出。
