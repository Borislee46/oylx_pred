# DS 一年备战 TODO

> 目标：字节（或其他互联网大厂）算法 DS / 策略 DS / 搜索推荐 DS
> 当前状态：教育行业 DA，有 ML 全链路项目 + 字节搜索运营经验
> 时间预算：约 12 个月（除非老板给技术总监 title 就留下）
> 最后更新：2026-05-03

---

## Q1: 作品建设 + 验证实验（0-3 月）

**目标**：让面试官在面你之前就看到你的能力。同时产出真实数据证据，把面试回答从"我学过"升级成"我验证过"。

> 详细实验设计见 `notes/interview/project-validation-plan.md`

### 1A: 验证实验 V 类（数据已有，直接跑）

| 序号 | 任务 | 说明 | 状态 |
|------|------|------|------|
| V1 | 校准深度分析 | reliability diagram 校准前后对比 + Brier Score 分解 + ECE。关键发现：predicted rate 54.9% vs actual 33.7%，模型高估 21%，定位原因 | ⬜ |
| V2 | 五层调整链 Ablation | 每层独立关掉，计算 Kendall tau + top-10 重叠率。目标：证明每层各有分工而非 over-engineering | ⬜ |
| V3 | 阈值敏感性分析 | cost ratio FP/FN ∈ {0.2~5.0}，看 threshold 怎么变。证明"F1 最优不是业务最优" | ⬜ |
| V4 | 特征重要性 Bootstrap | 1000 轮 bootstrap，画特征排名箱线图。证明 target_major 的 top-1 地位是否稳定 | ⬜ |

### 1B: 写作 + 展示

| 序号 | 任务 | 说明 | 状态 |
|------|------|------|------|
| 1.1 | GitHub README 英文版 | 写好 pipeline 架构、技术栈、快速开始，配架构图 + V1-V4 关键图表 | ⬜ |
| 1.2 | 掘金博客 #1 | 《留学录取预测：从 XGBoost 到概率校准的完整实践》，含 V1-V3 图表 | ⬜ |
| 1.3 | 知乎专栏同步 | 掘金发布后 48h 同步知乎 | ⬜ |
| 1.4 | 项目 demo 页 | 部署脱敏版（假数据），Streamlit Cloud 免费部署 | ⬜ |
| 1.5 | GitHub README 末尾放博客 + demo 链接 | 简历上一行链接全搞定 | ⬜ |

---

## Q2: AB + 实验体系（3-6 月）

**目标**：能用统计学术语解释你过去做的所有运营决策，并用项目数据做模拟验证。

> 详细实验设计见 `notes/interview/project-validation-plan.md`

### 2A: 模拟实验 S 类（写代码模拟）

| 序号 | 任务 | 说明 | 状态 |
|------|------|------|------|
| S1 | Rolling Window Backtest | 按时间窗口滚动训练/测试，追踪 AUC/Brier/positive_rate_bias 趋势。定位模型 drift 的严重程度和 retrain 周期 | ⬜ |
| S2 | Peeking Problem 模拟 | 在自己数据上模拟：naive peeking vs 固定样本量 vs sequential testing。量化 Type I Error 膨胀幅度 | ⬜ |
| S3 | Multiple Testing 演示 | 20 个指标 + Bonferroni/BH 校正对比。证明为什么实验前要锁定主指标 | ⬜ |
| S4 | Power Analysis / MDE | 基于真实录取率 baseline ~34%，算不同样本量下的 MDE。反向算"想检测 +2pp 提升需要多少数据" | ⬜ |
| S5 | 分层采样效率演示 | 按学校层级分层 vs 简单随机，量化 variance reduction | ⬜ |
| S6 | OOD 检测系统化 | 各特征分位数 → OOD 三级（绿/黄/红）→ 对比 OOD case 准确率 vs in-distribution | ⬜ |

### 2B: 理论学习

| 序号 | 任务 | 说明 | 状态 |
|------|------|------|------|
| 2.1 | 读完 Kohavi 1-6 章 | 重点记 MDE / Power / Peeking / Multiple Testing。配合 S2-S4 实验加深理解 | ⬜ |
| 2.2 | 用术语重写你的搜索经历 | 对 libra 实验面板上的每个指标写 2 句术语描述 | ⬜ |
| 2.3 | 刷 AB 面经 ×15 题 | 重点"实验不显著怎么判断""怎么防范 p-hacking""不能做 AB 怎么办"。用 S2-S4 数据回答 | ⬜ |
| 2.4 | 掌握 Diff-in-Diff 框架 | 能把"反转实验验证长期效果"用 DiD 术语讲 | ⬜ |
| 2.5 | 掌握 RDD 基本思路 | 能把"政策门槛的效应评估"用断点回归讲 | ⬜ |

---

## Q3: ML 深度 + 博客写作（6-9 月）

**目标**：你的代码已经能打了，补的是"为什么这么选"的理论深度。同时把 V+S 实验成果写成技术博客。

| 序号 | 任务 | 说明 | 状态 |
|------|------|------|------|
| 3.1 | Calibration 专题 | 理解 Brier Score 分解（Calibration + Refinement - Uncertainty），能解释 calibration curve。直接引用 V1 数据 | ⬜ |
| 3.2 | Monotonic Constraints 专题 | 理解 XGBoost 叶子约束的底层实现，知道 LightGBM 的差异 | ⬜ |
| 3.3 | Imbalance Handling 专题 | 对比 scale_pos_weight / SMOTE / Focal Loss / 降采样 各场景优劣。引用 V3 cost-sensitive 分析 | ⬜ |
| 3.4 | Feature Engineering 原则 | 能把 log1p/capping/标准化 的选择用一句话讲清楚 WHY。引用 V4 bootstrap 特征稳定性数据 | ⬜ |
| 3.5 | SHAP 可解释性 | 能解释一个具体 case 的 SHAP waterfall plot。结合 V2 ablation 做交叉验证 | ⬜ |
| 3.6 | 掘金博客 #1 | 《留学录取预测：从 XGBoost 到概率校准的完整实践》，含 V1-V4 全部图表 | ⬜ |
| 3.7 | 掘金博客 #2 | 《没有 AB 平台怎么做实验思维训练——Peeking/多重比较/统计功效的模拟验证》，含 S2-S5 图表 | ⬜ |
| 3.8 | 实验代码开源 | `experiment-simulations/` 目录上传 GitHub，独立于主项目 | ⬜ |

---

## Q4: 投递准备（6-9 月）

**目标**：简历 + 面试叙事 + 内推全就位

| 序号 | 任务 | 说明 | 状态 |
|------|------|------|------|
| 4.1 | 简历重写 | 标题 Data Scientist/ML Engineer，摘要第一句强调 ML 全链路 | ⬜ |
| 4.2 | 两个项目叙事打磨（2+5 分钟版） | 字节搜索（证明实验+规模感）+ 留学预测（证明 ML 深度+独立闭环） | ⬜ |
| 4.3 | 行为面试故事准备（STAR 格式 ×4） | 失败经历 / 冲突与推动 / 取舍决策 / 反对上级。每个 5 行：S→T→A→R→Learn | ⬜ |
| 4.3b | 行为证据自查 | 对照 `ds-leader-hrbp-prep.md` HRBP 踩坑清单逐项自查，定位前两次挂掉原因 | ⬜ |
| 4.4 | 人脉盘点 | 确认还有哪些前同事在互联网大厂，列个单子 | ⬜ |
| 4.5 | 联系字节 PM（可选） | 把 GitHub + 博客发给 PM，约个 coffee chat | ⬜ |
| 4.6 | 牛客网刷面经（字节 DS 近半年） | 了解当前面试出题风格和难度 | ⬜ |

---

## Q5: SQL + 工程（持续进行）

**目标**：SQL 笔试不翻车，Spark 能聊

| 序号 | 任务 | 说明 | 状态 |
|------|------|------|------|
| 5.1 | LeetCode SQL hard ×15 | 每道必须看 EXPLAIN 执行计划 | ⬜ |
| 5.2 | 窗口函数专项 | ROW_NUMBER / RANK / LAG / LEAD / cumulative SUM | ⬜ |
| 5.3 | Spark 理论三件套 | Shuffle / Broadcast Join / 数据倾斜，能各用 2 句话讲清楚 | ⬜ |
| 5.4 | 用 Spark 重构你的留学数据 pipeline（可选） | 如果时间允许，用 PySpark 把 model_trainer 改写一遍 → 简历里可以写"Spark" | ⬜ |

---

## Q6: 面试冲刺（9-12 月）

**目标**：拿 offer

| 序号 | 任务 | 说明 | 状态 |
|------|------|------|------|
| 6.1 | Mock 面试 ×3（找朋友或付费平台） | 真实压力测试，录音回听找问题 | ⬜ |
| 6.2 | 前两次投非第一志愿公司 | 先拿小厂 DS 练手，面完复盘，修改回答 | ⬜ |
| 6.3 | 正式投字节搜索/推荐 DS 岗 + 其他大厂 | 面试官视角：你的项目 + 你的背景 = 搜索/策略 DS 的 top-tier 匹配 | ⬜ |
| 6.4 | 谈薪 + 背调准备 | 了解目标级别薪资范围，准备好 current salary 的合理说法 | ⬜ |

---

## 随时可以做

- 每天刷一道 LeetCode SQL medium/hard
- 记面试术语本（随时听到一个新术语就记下来 + 翻译成你的语言）
- 读一篇 ML/DS 博客（ArXiv / 技术号 / 公众号）保持知识更新
- 关注字节 DS 在 boss 直聘/脉脉上放的 JD，追踪招人方向

---

## 已经完成的（持续更新）

| 序号 | 任务 | 日期 |
|------|------|------|
| ✅ | 代码注释：7 个核心文件的 WHY 注释 | 2026-05-03 |
| ✅ | DS 面试 20 道追问清单 | 2026-05-03 |
| ✅ | 面试 Q&A 实录 + 术语表 | 2026-05-03 |
| ✅ | 补课路线图 (ds-gap-roadmap.md) | 2026-05-03 |
| ✅ | 字节搜索经历整理 | 2026-05-03 |
| ✅ | Spark 优化笔记 | 2026-05-03 |
| ✅ | 项目验证实验计划 (project-validation-plan.md) | 2026-05-05 |
| ✅ | 一年 TODO 更新（V+S 实验整合） | 2026-05-05 |

---

## 如果提前上岸（老板给技术总监）

如果公司给了技术总监 + 合理的 scope（不只是 title 虚名可以管更多事），也不用死磕互联网。真正的加分是：

- 管理经验（带人 > 个人贡献者）
- 技术选型权（选技术栈 > 用别人定的栈）
- CTO / VP 级别的推荐信

这些在你有 1-2 年技术总监经验后再去互联网，起点直接是 senior DS / DS lead，比现在以 junior/mid DS 身份进字节快得多。

**底线**：给了技术总监但 scope 没变 → 别被 title 骗了，该走还是走。
