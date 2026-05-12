# 大厂 DS 面试模拟：基于留学预测项目的深度追问

## Round 1: ML 基础 — 往深了挖

### Q1: 单调约束的底层
"你说 XGBoost monotone_constraints 保证单调性。如果我不让你用 XGBoost——
- **LightGBM** 怎么加单调约束？它的实现和 XGBoost 有区别吗？
- **神经网络** 怎么保证单调性？"

> 要点：LightGBM 也支持 `monotone_constraints`，但作用于 leaf-wise 而非 level-wise，行为有差异。NN 加单调性需要约束权重非负（`tf.nn.relu` + 非负权重初始化），或用 monotonic neural network 架构。这说明你理解"约束是需求，库只是工具"。

### Q2: 校准的验证
"你说用了 sigmoid calibration。给我看 calibration curve（校准前后对比）。
- Brier Score 校准前 vs 校准后差多少？
- 如果 isotonic 在校准集上 Brier Score 更好，你敢换吗？为什么？"

> 要点：校准曲线 + reliability diagram 是标准答案。Isotonic 在小数据上容易过拟合（校准集 Brier 好但测试集差），能说出这个 tradeoff 就过关。进阶：Expected Calibration Error (ECE) 和 Maximum Calibration Error (MCE) 的区别。

### Q3: Ablation Study
"你的概率调整链有 5 层。怎么向面试官证明每一层都是必要的？
- 如果去掉 GPA 惩罚层，top-10 推荐的变化率是多少？
- 有没有做过 ablation——每一层单独移除看对最终排序的影响？"

> 要点：这是考察"你是否有实验思维"。即使没做过 ablation，能说出设计 ablation 的方法论也是加分。关键指标：Kendall's tau（排序一致性变化）。

### Q4: 样本不均衡的边界
"scale_pos_weight 在正负比 1:5 时好用。如果正负比变成 **1:100** 呢？
- scale_pos_weight = 100，梯度比例太悬殊，XGBoost 训练还稳定吗？
- 在什么极端情况下 SMOTE 反而比 scale_pos_weight 更好？"

> 要点：1:100 时 scale_pos_weight 导致正例梯度爆炸，训练震荡。SMOTE 在特征以数值为主时较好，但你的系统有大量分类特征（院校/专业），SMOTE 合成的新样本可能在语义上不合理（如"香港大学 + 文科专业"的合成中间点无意义）。能说出这个维度就超过 90% 候选人。

### Q5: 特征变换的可解释性
"你对经历计数做了 log1p 变换。SHAP value 输出的是 log1p(x) 对 log-odds 的影响。
- 向业务方解释 SHAP 时，log1p 变换怎么翻译成业务语言？
- 如果业务方问 '多一段实习到底加多少概率'，你怎么回答？"

> 要点：log1p 的非线性使得边际效应不均匀——从 0→1 段经历和从 4→5 段的增量概率不同。用 SHAP partial dependence plot 展示曲线而非报单个数字。这是"技术到业务的翻译能力"。

---

## Round 2: 系统设计 & 工程化

### Q6: 延迟与扩展性
"用户点提交到看到预测结果，p99 延迟是多少？瓶颈在哪一步？
- 如果日活从 10 个顾问变成 1000 个，你的 Streamlit 单进程扛得住吗？
- Agent 调用 DeepSeek API 的 timeout 是 15s，如果 API 挂了整个页面 hang 住怎么办？"

> 要点：pipeline 延迟分解（特征构建 vs 模型推理 vs Agent调用）。Streamlit 是单进程，1000 DAU 需要改成 FastAPI + 异步推理 + 缓存。Agent 应该有 fallback（缓存结果/降级到规则解读）。

### Q7: 模型迭代 & Drift
"新学年开学了，新录取数据进来了。你怎么知道模型该 retrain？
- 有没有做 concept drift detection？
- 多久 retrain 一次？手动还是自动？
- retrain 后发现新模型 F1 比旧模型低 3%，你上不上线？"

> 要点：Population Stability Index (PSI) 监控特征分布漂移。F1 下降但 calibration 可能变好——上线决策不是单指标。如果新数据的录取模式确实变了（如政策变化），旧模型可能更不准。这是"工程判断力"。

### Q8: 技术选型的 Tradeoff
"为什么选 Streamlit 而不是 FastAPI + React？如果给你 2 周重构前后端分离，你会吗？
- Streamlit 的 session_state 在并发场景下有什么问题？"

> 要点：Streamlit 适合内部工具快速迭代，但 session_state 是线程不安全的，多用户共享 session 会出问题。知道这个限制说明你理解技术边界。

### Q9: 线上诊断
"用户投诉：'我的 GPA 3.8 但系统只给我 20% 概率'。
- 你怎么排查是模型问题还是数据问题还是概率调整链问题？
- 你的 `_adjustment_trace` 存在哪？能复现这个用户的完整调整链路吗？"

> 要点：能说出逐层排查的思路：trace 回溯 → 检查该 case 的 gpa_penalty/lang_penalty/跨专业惩罚 → 对比相似 case → 检查是否校准 drift。这就是 production debugging 能力。

---

## Round 3: 实验 & 因果（DS 面试最易挂的一轮）

### Q10: A/B 测试设计
"你们上线这个预测系统后，怎么用 A/B 测试证明它对业务有价值？
- 实验组和控制组分别是什么？
- 核心指标是什么？
- 最大的干扰因素是什么？"

> 要点：控制组 = 顾问人工评估；实验组 = 系统辅助。指标 = 录取率、用户满意度、顾问效率（每个 case 耗时）。最大干扰：顾问看到预测后会改变行为（如不推荐低概率学校），这会产生 self-fulfilling prophecy —— 你预测低所以不申，不申所以录取少，但无法知道"申了会不会录"。

### Q11: 自我实现预言
"如果系统预测某学生申港大只有 15% 概率，顾问就不推港大。
最后确实没录——但你怎么知道是'本来就不可能'还是'系统误导了决策'？
- 这在因果推断里叫什么问题？
- 怎么用实验设计解决？"

> 要点：这就是 selective labels problem / feedback loop。解决方案：exploration arm——随机给一部分用户不展示低概率结果（或故意推荐一些"系统不看好"的组合），观察实际录取率。这涉及 ethics（不能给用户明知很差的建议），需要权衡。

### Q12: 伦理与产品
"产品经理说：'把低 GPA 学生的预测概率调高 10%，申请量就上来了'。
- 你怎么回应？
- 如果 VP 也支持这个方向，你怎么在不造假的前提下做到？"

> 要点：不能造假。但可以：1) 推荐"彩票校"（reach schools）并明确标注"挑战档"；2) 强调 offer 中最有希望的方向；3) 展示"如果 GPA 提 0.2 能开哪些新选择"——给 hope 但不给 fake hope。这是"DS 的职业道德"问题，面试官很喜欢问。

---

## Round 4: 业务理解 & 产品思维

### Q13: North Star Metric
"这个系统的成功怎么衡量？准确率？用户满意度？申请转化率？
- 如果三个指标冲突了，你优先哪个？"

> 要点：没有标准答案，但要有清晰的逻辑链。建议："准确率是基础（没有准确率就没有信任），但在准确率达到一定水位后，用户行为指标（是否按推荐申请、是否满意推荐列表）优先级上升。最终 north star 应该是'顾问采纳系统推荐 + 用户录取'的联合指标。"

### Q14: 单特征挑战
"如果只允许用一个特征，你选哪个？用数据怎么证明你的选择？"

> 要点：考的是特征重要性 + 业务直觉 + 验证方法。GPA 可能是最常见的答案，但你的系统里 `language_score` 和 `background_university` 也可以讨论。关键不是答案，而是你能说出"在测试集上逐个单特征训练，比较 ROC-AUC"的验证方法。

### Q15: 用户体验设计
"一个学生看到系统预测只有 12%，很沮丧。
- 你怎么设计产品让他不'只看到数字'？"

> 要点：分层展示——顶层是档位（冲刺/匹配/保底），中层是概率区间（10-25%），底层才是精确数字。配合提升建议（"语言提 0.5 分可升至 25%"）把焦虑转化为行动。这是 DS 的产品 sense。

---

## Round 5: 边缘情况 & 失败模式

### Q16: 高置信度错误
"系统预测某用户 85% 录取概率，Tier-1 保底校，结果被拒了。
- 你怎么排查根因？
- 你的 Pipeline 有冷启动问题吗——新学校新专业有 0 个案例时怎么办？"

> 要点：排查路径：check model prediction → check each adjustment layer → check calibration → check data freshness。新学校问题：用相似学校的先验（学校层级近似），或依赖更多文本特征而非 case 匹配。冷启动策略应该在 README 里写明。

### Q17: 分布外检测
"进来一个用户：GPA 4.0、IELTS 9.0、3篇SCI、清华本科。
- 训练数据里几乎没这种人。你的模型会怎么预测？
- 你怎么保护系统不对 OOD 样本给出荒谬结果？"

> 要点：OOD 检测可以用 Mahalanobis distance 或模型不确定度（ensemble variance）。XGBoost 单模型对 OOD 不敏感——它只是把输入放进最接近的叶子。但概率调整链的 quadratic penalty 会识别到 "gpa > mean+2std" → 不触发惩罚，行为合理。能说出这个观察比直接答"要做 OOD 检测"更好。

### Q18: 政策冲击
"假设港大突然宣布扩招 200%，录取标准大幅降低。
- 你的系统多久能反映这个变化？
- 在这期间的预测会怎么样——偏高还是偏低？"

> 要点：模型基于历史数据 → 依旧按旧标准预测 → 概率偏低（保守估计）。等到新数据回流 → retrain → 通常需 1 个申请季。过渡期可以加一个 policy adjustment layer（人工校正系数），但这个系数怎么定？—— 需要 domain expert 输入，不是纯 ML 问题。

---

## Round 6: 统计基础

### Q19: Brier Score 分解
"Brier Score = Calibration + Refinement - Uncertainty。解释一下：
- 你的模型在这三项上分别表现如何？
- Refinement term 低代表什么？对你这个场景是好事还是坏事？"

> 要点：Refinement = 模型"区分度"（resolution），低 refinenment = 所有预测概率接近均值 = 模型没有区分能力。高 refinement = 预测概率分散，区分度高。录取场景下 refinement 通常不高（大家都挤在 20-60%），这合理——录取本身就不确定。能解释"为什么不高"而不只是"低了不好"。

### Q20: Threshold 的业务含义
"你的 threshold=0.24 是 F1 最优。但 F1 假设 precision 和 recall 权重相等。
- 在你们的业务场景下——'推荐一个不该申的学校'（FP）和'漏掉一个该申的学校'（FN），代价一样吗？
- 如果不一样，threshold 应该往哪边调？"

> 要点：FP（推荐了不该申的）→ 浪费申请费 + 被拒打击信心；FN（漏掉能申的）→ 错过机会。两者代价不一样。应该问业务方"你们更怕哪个？"而不是从 F1 直接推导。这考的是"DS 不做数学的奴隶"的思维。

---

## Bonus: 大厂 DS 面试官的心理

面试官不是在找"答对的人"，而是在找"让我想和他做同事的人"。以下行为自动加分：

1. **说不知道时补充"但我的猜测是..."** —— 显示思维过程而非放弃
2. **反问我** —— "这个场景下你们的 precision/recall tradeoff 是怎么定的？" 表示你在想产品
3. **承认代码的不完美** —— "这个 penalty=0.85 其实是我拍脑袋的，我知道应该做 sensitivity analysis 但我没来得及" 比假装它是最优解更可信
4. **把你的项目当成真的产品聊** —— 不是"我做了一个 ML model" 而是 "我帮顾问做选校决策"
