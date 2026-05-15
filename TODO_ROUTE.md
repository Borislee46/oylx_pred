# TODO 路由表 — AI Native

> 给AI看的任务索引。用户说关键词时，AI应自动定位到对应TODO、读取关联文件。

## AI 行为准则 — 每次干活都要遵循

当用户要做任何 TODO 相关的设计、实现、验证时，**必须**按 Senior DS 的思路去工作，不只是把功能做出来：

1. **Why before How**：先诊断根因，再设计方案。不要上来就写代码。追问"这个问题为什么重要？真正的原因是什么？"
2. **Verification before claim**：说"修好了"之前，先设计验证方案——怎么证明修好了？对照实验怎么设计？metric怎么选？
3. **Quantify uncertainty**：任何结论都带上不确定性量化——置信区间、敏感性分析、限制条件。不说"概率是0.7"，说"在n=5的情况下，概率0.70但置信区间[0.3, 0.9]"。
4. **Trade-off explicit**：每个设计决策写清楚trade-off——选了方案A意味着放弃了什么？什么情况下方案A会失效？
5. **Interview-ready narrative**：做完每件事，能用3句话讲清楚——问题是什么、为什么这样做、怎么验证的。
6. **Know when NOT to do something**：不是所有问题都需要ML/因果推断/deep learning。能说清楚"这个问题做描述性分析就够了"同样是senior能力。
7. **Plain language final step**：分析完了，必须能用大白话讲给业务听。面试官不会被"heteroscedasticity""empirical Bayes shrinkage"这些词加分——加分的是"我给我妈讲一遍她也能懂"。吃透→讲简单，这是senior DS的最后一公里。

---

## TODO-1 · Corner Case 不确定性量化

```yaml
status: experiment_done
priority: critical
ds_domain: Uncertainty Quantification
triggers: [corner case, 小样本, 稀疏, 不置信, ECE, 校准, 天花板, 预测失败]
effort: medium
depends_on: []
blocks: [TODO-2, TODO-3]
```

**当前状态**：实施阶段。1-4 项已完成（Faculty 3-tier + ceiling per-layer + sim threshold + UI 不置信标记），5 待做，6 不做。

**实施进度**：
- 🔴✅ Faculty penalty 重构 → 3-tier severity (light 0.7 / medium 0.5 / heavy 0.3)
- 🔴✅ 天花板差异化 → per-layer ceiling (1:70%, 2:55%, 3+:45%)
- 🟡✅ 相似度阈值 0.89→0.87
- 🟡✅ UI 不置信标记 → 专业名按 n 着色（红/橙/默认）
- 🟢✅ 预测失败兜底（6% case）→ 三层 fallback：pre-flight 检查 + 人口统计 cascading + UI 标识
- ⚪ 文本 uplift 升级 → **已验证不做（2026-05-14 深挖）**。三路特征在 n=5,000 上全部 ΔAUC<0.01：Surface +0.0084、TF-IDF centroid -0.0043、**E5 -0.0017**（5090 已跑完）。文本"含金量"在统计上不可测量或根本不存在。详见 `reports/text_quality_analysis.json`

**V5-V9 诊断结论 (2026-05-13)**：
- V5: Faculty penalty 是头号校准杀手（关掉 ECE 0.15→0.11），37.2% case 触顶
- V6: 外部数据 -67pp = DGP mismatch + 相似度缓存命中仅 23%
- V7: 缺失 GPA 与更高录取率相关（+13pp，correlational），DEC-010 需重审
- V8: **文本 uplift 是 near-zero-effect 组件**（mean uplift=0.36%, ΔECE=-0.0006），不深挖
- **V8+ (2026-05-14)**: 深挖了 TF-IDF vs E5 vs surface features，全部 ΔAUC<0.01。发现 TF-IDF centroid 衡量"质量"、E5 衡量"相关性"——两个不同构念，但都不预测录取。E5 5090 已跑完（n=5,000, ΔAUC=-0.0017）
- V9: **相似度 0.89 阈值切反了**（0.88-0.89 admit 29.8% > 0.89-0.90 admit 29.0%），<0.80 死区

**要解决的问题**：模型在小样本区域统一输出~0.70，6%完全预测失败。根因在5层惩罚链的乘法叠加+仲裁器70%上限。

**DS核心价值**：不确定性量化——分层诊断定位非对称偏差（C9 -18pp vs 双非 -6pp），设计对照实验区分"LLM误读"vs"模型外推失败"。

**实施路线**（按 ROI 排序，基于 V5-V9 实证）：
1. 🔴 **Faculty penalty 重构**（最大 ROI）→ `src/pages/prediction/flow/adjustment_pipeline.py`
2. 🔴 **天花板差异化**（37% case 触顶）→ `config.py` MAX_TOTAL_PENALTY_RATIO
3. 🟡 **相似度阈值 0.89→0.87**（V9 发现切反了）→ `config.py` MIN_SIMILARITY_THRESHOLD
4. 🟡 **UI 不置信标记**（改动最小，最快止血）→ `src/pages/prediction/result_display/`
5. 🟢 **预测失败兜底**（6% case）→ `src/pages/prediction/result_modifier/`
6. ⚪ **文本 uplift 升级**（V8 证明 near-zero-effect，不做了）

**相关文件**：
- 分析：`reports/prediction_diagnosis_20260513.md`（主参考）、`DECISIONS.md#DEC-002/003/004/007`
- 代码：`result_modifier/arbitrator.py`、`adjustment_pipeline.py`、`config.py`
- 测试：`test_sparsity_stress.py`、`test_corner_cases.py`、`test_calibration_report.py`、`test_ai_blind_eval.py`

**面试叙事**：模型在小样本区域有系统性偏差，分层诊断定位到惩罚链的非对称效应，设计不确定性量化框架管理稀疏预测风险。

**Senior DS 会怎么评价**：这件事的亮点不是"修了一个bug"，而是**诊断过程**。分层拆解（C9 -18pp vs 双非 -6pp）说明你知道偏差从哪来的、对不同用户群影响不一样。面试官会追问"你怎么确定是调整链的问题不是base model的问题"，你能拿出ablation trace才是senior级回答。还有一个陷阱："ECE 0.1155就一定需要修吗？用户觉得概率合理啊"——要能说清楚calibration metric和user perception的gap，以及稀疏区域低估虽然用户不投诉但长期损害产品信任。

---

## TODO-2 · Monte Carlo 全拒率联合概率

```yaml
status: not_started
priority: medium
ds_domain: Dependence Modeling / Probabilistic Graphical Models
triggers: [全拒, Monte Carlo, 联合概率, 组合录取, 相关性, 多目标, copula]
effort: high
depends_on: [TODO-1]
blocks: []
```

**当前状态**：单预测输出独立概率，多目标查询无交叉污染（已测试），但未做联合建模。

**要解决的问题**：学生申N所学校，"全拒"概率不是 ∏(1-p_i)。录取事件非独立——同一个学生的GPA在所有学校共享。

**DS核心价值**：相关性建模——从历史残差估计条件依赖 P(B被拒|A被拒)，用copula或empirical Bayes shrinkage。独立假设会严重低估全拒率。

**实施路线**：
1. 读现有 `generate_correlation_matrix.py` + `school_combination_optimizer_algorithm/`（EXPERIMENTAL）
2. 设计相关性估计方案（残差copula vs empirical Bayes）
3. Monte Carlo模拟框架 + 历史多申请backtest验证
4. 产品化：输出"申5所至少中1所概率" + 全拒风险告警

**相关文件**：
- 代码：`scripts/generate_correlation_matrix.py`、`school_combination_optimizer_algorithm/`
- 测试：`test_corner_cases.py::test_multiple_targets_probability_ordering`

**面试叙事**：独立假设低估全拒风险。用残差相关性估计学校间条件依赖，Monte Carlo给出诚实的联合概率——概率图模型思维。

**Senior DS 会怎么评价**：面试官听到Monte Carlo会先点头再追问："你只有23个target school，相关性矩阵有多少自由度？数据够估计吗？" 所以不能只说"用copula"，要说清楚在数据有限时的shrinkage策略。另一个加分点：联合概率对产品决策有实际价值——"全拒概率>90%"这个信号可以触发顾问主动介入，把DS工作和产品指标直接挂钩。这是senior和junior的分水岭：junior做模拟，senior解释为什么模拟结果值得信赖。

---

## TODO-3 · 外部数据 Domain Adaptation

```yaml
status: data_cleaned
priority: medium
ds_domain: Distribution Shift / Transfer Learning
triggers: [外部数据, ApplySquare, Compass, 泛化, 分布漂移, domain adaptation, transfer learning]
effort: medium
depends_on: [TODO-1]
blocks: []
```

**当前状态**：ApplySquare（507行）/ Compass（17K行）已清洗，偏差-67pp已确认（铁证），待设计使用方案。

**要解决的问题**：外部数据标签分布严重不同——ApplySquare是爬虫数据（录取了才发帖，self-selection bias），Compass是机构数据（只记录成功case，censoring）。直接merge训练会污染内部分布。

**DS核心价值**：Distribution shift诊断——feature-level vs label-level分拆，DGP分析（爬虫self-selection vs 机构censoring），post-hoc calibration reference而非训练数据。本质是covariate shift下的importance weighting。

**实施路线**：
1. 外部数据分布漂移定性报告（feature分布对比 + label分布对比）
2. 用外部数据做out-of-sample校准验证（不是训练）
3. 探索importance weighting或domain adaptation方法
4. 设计外部数据持续接入pipeline

**相关文件**：
- 数据：`data/external/applysquare.feather`、`compass.feather`
- 脚本：`scripts/clean_external_data.py`
- 测试：`test_external_data_validation.py`

**面试叙事**：外部数据不能直接训练——数据生成过程不同导致-67pp label偏差。分离feature-level和label-level漂移，用外部数据做校准参考而非训练数据——更严谨的domain adaptation。

**Senior DS 会怎么评价**：这个-67pp偏差反而是亮点——几乎所有面试者都会说"我用了外部数据扩充训练集"，但很少有人能做"为什么外部数据不能用"的诊断。两个数据源的DGP分析（爬虫self-selection vs 机构censoring）展示了data literacy。面试官会追问："如果只差-9pp（训练集内部）但外部差-67pp，你认为根因在feature还是label？"——这时候能分拆回答而不说"都有"就是senior信号。关键不是最终用了什么方法，而是**为什么选择不直接merge的决策逻辑**。

---

## TODO-4 · 学校视角 / 因果推断

```yaml
status: done
priority: medium_high
ds_domain: Causal Inference / Counterfactual
triggers: [学校视角, school_view, 申请人画像, 低年级, 因果推断, 反事实, counterfactual, ITE, 差距分析, What-If]
effort: high
depends_on: []
blocks: []
```

**当前状态**：Phase 1 已完成并落地，已删除 demo。Phase 2-4 增强方向待研判。

### Phase 1 — 描述性分析 + 反事实模拟 ✅

**页面路由**：`pages/school_view.py`，独立页面，不依赖 hk.py 预测流程。

**三个 Tab**：

| Tab | 功能 | 实现 |
|-----|------|------|
| **学校画像** | 目标学校的录取者 P50/P25/P75（GPA/语言/科研/实习/论文/奖项），学生在录取者中的百分位排名，差距排序（最短板Top3） | `school_profiles.py`，MIN_SAMPLES=10 |
| **What-If 模拟** | 5个预设场景（GPA+0.2/+0.4、语言+0.05、科研+1、实习+1），对比基线 → 按学校聚合 → ROI排序 | `what_if_simulator.py`，调 XGBoost predict_batch |
| **学校对比** | 多校并排 `st.metric` 卡片，每个 feature 一行，显示 P50 + 学生百分位 + 样本量 | `page.py` `_render_comparison_tab` |

**数据流**：
```
表单（GPA/语言/经历+目标学校）
  ├─→ SchoolProfileCalculator → P50/P25/P75 + 百分位 + 差距排序
  └─→ WhatIfSimulator → XGBoost predict_batch → 5场景概率 → ROI表
```

**What-If 场景（硬编码）**：baseline / GPA+0.2 / GPA+0.4 / 语言+0.05 / 科研+1段 / 实习+1段。只支持单因子扰动，不支持组合（如同时 GPA+语言）。

**关键设计决策**：
- `MIN_SAMPLES_FOR_PROFILE = 10`：少于 10 个录取样本的学校不展示画像
- `TOP_MAJORS_PER_SCHOOL = 5`：What-If 每个学校只跑录取人数最多的前 5 个专业
- `get_top_majors()` 方法有实现也有测试，但 UI 层没调用——学校画像 Tab 不展示热门专业列表

**文件**：
- 路由：`pages/school_view.py`
- 模块：`src/pages/school_view/`（6 文件：data_loader / school_profiles / school_form / what_if_simulator / page / __init__）
- 测试：`tests/test_school_view.py`（23 tests，含单调性验证）

**已知局限**：
- 场景硬编码，用户不能自定义扰动幅度或组合
- 点估计无置信区间——"GPA +0.2 → 概率 +6%"，用户不知道 ±多少
- 无图表（纯表格和 metric 卡片）
- 表单无 session 持久化，切换页面后重置
- 不检查目标学校是否等于本科学校

### Phase 2-4 — 增强方向（待研判）

**要做的事不是因果推断**。当前数据 47,835 个四维组合、中位 1 样本，严格做 ITE / DAG / counterfactual 置信区间会宽到没有实操意义。产品侧也不需要——Phase 1 的 XGBoost predict_batch 给方向性参考已经够用。

取而代之，三个增强 Phase 1 的方向：

| # | 方向 | 做什么 | 用户看到什么 | 难度 |
|---|------|--------|-------------|------|
| A | Bootstrap 置信区间 | 对每个 What-If 场景做 bootstrap resampling，给 80% 置信区间 | "GPA +0.2 → 概率 +3%~+9%" | 低，纯计算，不动模型 |
| B | 自定义扰动 | 放开硬编码场景，拖滑块调幅度 + 支持多因子组合 | "GPA +0.3 同时语言 +0.5" | 中，改 form + simulator |
| C | 帕累托直觉 | 不跑因果模型，直接算每单位投入的边际收益 | "花 3 个月刷 GPA 比刷语言收益大 2 倍" | 低，纯计算 |

A 和 C 本质是数据转换，不动 XGBoost。B 改 UI 和 simulator 的 SCENARIOS 结构。三者加起来不涉及因果推断，但用户体感提升明显。

**因果推断部分留给面试**：写一份 `reports/identification_argument.md`，说明如果数据够（每个 combo ≥30 样本）会怎么做——DAG 设计、confounding 识别、ITE 估计方法选型、为什么当前数据下不做。面试时这段比强行上 DML 高级。

**文件**：
- 现有：`docs/pathfinder_v2_blueprint.md`、`src/pages/school_view/`
- 待写：`reports/identification_argument.md`

---

## TODO-5 · 用户调研 → 产品优化路线图

```yaml
status: surveys_ready
priority: high
domain: Product Research / Stakeholder Alignment
triggers: [用户调研, 问卷, 销售反馈, 选校反馈, CEO汇报, 优化排期, 产品路线图]
effort: medium
depends_on: []
blocks: [TODO-1, TODO-4]
```

**当前状态**：两份调研问卷已完成，等待发放。后期文书已审计，前期销售和中期选校尚未收集过系统反馈——而他们是最高频用户。

**为什么做这件事**：
- 已知 DS 层面的问题（ECE=0.1155、系统性低估 -9pp、C9 被低估 18pp），但不知道用户在不在乎
- 后期审计过了，但后期不是核心用户——选校和销售才是
- 优化排期需要业务输入，不能纯靠 DS 指标驱动
- 最终要向 CEO 汇报，需要一线的真实声音

**调研设计**：
- **销售端** `reports/signals_survey_sales.md`（10 选择 + 4 开放）：签单帮助 → 客户反应 → 界面冲击力 → 概率对学生心态的影响 → 竞品
- **选校端** `reports/signals_survey_selection.md`（9 选择 + 5 开放）：概率准确度 → 方案影响 → AI 解读使用 → 被替代顾虑 → 参与优化的意愿

**设计上刻意埋的钩子**：
- 销售 5.4："你私下有没有把概率往上说"——如果答案是"经常"，说明概率展示有系统性信任问题
- 销售 4.1："界面看起来高级吗"——如果没有一个人选"很高级"，UI 重做优先级直接拉满
- 选校 7.1："系统会不会替代选校顾问"——高风险问题，CEO 汇报时交叉分析最有价值
- 选校 7.3："顾问最不可替代的价值"——反向验证系统定位，如果答案跟系统功能重叠就是真实威胁

**下一步**：
1. 发放问卷（问卷星/腾讯问卷/纸质，看团队习惯）
2. 回收分析：按部门分 → 选择题统计 → 开放题归类 → 交叉分析（高频 vs 低频用户 / 不同使用时长）
3. 产出 CEO 汇报材料：3-5 页，重点放"用户说概率偏低的占比""最想改的 Top 3""顾问是否担心被替代"
4. 根据调研结果更新 TODO-1~4 的优先级和 scope

**产品风险假设（需要调研验证或推翻）**：
1. 概率系统性低估在用户端被感知到了 → 如果是，TODO-1 优先级提到最高
2. 销售需要"展示版"而不是"工具版" → 如果是，需要拆出一个 presentation mode
3. 选校顾问担心被替代 → 如果是，定位上要从"替代"转向"增强"
4. AI 解读没人看 → 如果是，要么砍掉要么重做

**相关文件**：
- 问卷：`reports/signals_survey_sales.md`、`reports/signals_survey_selection.md`
- 诊断：`reports/prediction_diagnosis_20260513.md`
- 决策记录：`DECISIONS.md`
- 产品洞察：memory `product_insight_actionable_probability.md`

**面试叙事**：不只是做了个模型——在模型上线后，用结构化问卷去验证"DS 指标上的问题在用户端是否被感知"，然后根据业务反馈排优化优先级。技术指标和用户感知之间的 gap 分析是 senior 和 junior 的分水岭——junior 盯着 ECE 说"必须修"，senior 先问"用户在乎吗"。

---

## TODO-6 · 5090 GPU 批量任务（回家跑）

```yaml
status: done
priority: high
domain: GPU Computing / Model Diagnostics
triggers: [5090, GPU, SHAP, E5预计算]
effort: ~1 hour total
```

**前提**：在家 5090 24GB 上跑，不是办公电脑。

### 6-1 · SHAP 全量模型解释 ✅ 已完成 (2026-05-14)

5090 上 28 秒（3.7s SHAP + 22.8s interaction），10K 样本。

**核心发现**：
- background_major 绝对主导（mean |SHAP|=1.20），匹配特征 >> 个人硬指标
- 最强交互：background_university × target_university (0.175)
- gpa (0.53) 的个体影响力只有专业匹配的一半
- language_score (0.11) 和 paper_count (0.09) 在尾部
- **意外发现**：NaN GPA 案例 SHAP 过度推高（均被拒），印证 DEC-010 median imputation 信息损失

**产出**：`reports/v10_shap_explanation/`（7 图 + README + JSON + 可复现脚本）+ `scripts/compute_shap.py`（SHAP 值生成）

**面试叙事**："我用 SHAP TreeExplainer 在 10K 样本上算了全量解释 + 交互矩阵，发现分类匹配特征主导预测，个人硬指标影响力只有一半。最强交互在学校档次匹配上——光看 feature_importances_ 看不到这个。"

### 6-2 · Bootstrap 置信区间 → 转为在线功能

**不能预计算**：Bootstrap CI 依赖每个学生的具体输入（GPA/语言/学校/专业组合），无法提前跑完。应作为 TODO-4 What-If 的在线增强功能——用户填表后几秒出 CI，不做离线批量。

### 6-3 · XGBoost gpu_hist 重训 → 跳过

**决策**：不做。原因——
1. `gpu_hist` 浮点精度和并行归约策略与 CPU `hist` 不同，同参数产出不同模型（非确定性）
2. Optuna 搜索轨迹也会变，最终模型可能与当前差异大
3. 改完要全量重验证（ECE + Brier + 89 tests + 5 层调整链 + SHAP），时间成本远超 GPU 节省的时间
4. ECE=0.1155 是调整链主导的，换 tree_method 不解决根本问题
5. 面试价值低——"换了 GPU 训练"不算亮点

### 6-4 · 全量 E5 相似度 → 已覆盖

E5 全量分析已在 `scripts/analyze_text_quality.py --e5` 中完成（n=5,000），结论是文本不能为硬指标提供增量预测力（ΔAUC -0.0017）。全量 61K 的 E5 嵌入预计算场景（相似学生检索、专业聚类）与当前核心任务无直接关联，暂缓。

### 5090 预跑总结

真正值得在 5090 上预跑的就两件，都已完成：
1. **SHAP 全量解释** — 面试必问的解释性，填了项目最大空白
2. **E5 文本质量深挖** — 三路特征（surface/TF-IDF/E5）全部 ΔAUC<0.01，为放弃文本 uplift 升级提供铁证

---

## 依赖与执行顺序

```
TODO-1 (UQ) ──┬──→ TODO-2 (Dependence)  # 单预测可靠→联合才可靠
              ├──→ TODO-3 (Distribution)  # 调参→用外部校准验证
              └──  TODO-4 (School View) ✅  # Phase 1 已落地

TODO-5 (调研) ──→ [TODO-2~3 优先级重排]    # 业务反馈决定下一步做什么

TODO-6 (5090 GPU) ✅ 收工                 # SHAP + E5 文本深挖
```

## AI 使用指令

当用户消息包含某个 TODO 的 `triggers` 关键词时：
1. 定位到对应 TODO，读取"相关文件"中的关键文件
2. 遵循上述6条行为准则——Why before How、验证先于声称、量化不确定性、显式trade-off、面试验收、懂得不做什么
3. 参考 `Senior DS 会怎么评价` 了解面试官可能的追问，在设计阶段就预判这些问题
4. 每次工作结束后，更新该 TODO 的 `status` 字段

```bash
# 快速上下文加载
cat TODO_ROUTE.md                          # 任务全景
cat reports/prediction_diagnosis_20260513.md # 诊断详情

# 质量验证
pytest tests/data_quality/ -v               # 全量62 tests
pytest tests/data_quality/test_sparsity_stress.py -v -s  # 稀疏压测
```

---

## Session Handoff (2026-05-14 下午)

**本轮完成**：
- **TODO-6 5090 预跑全部收工**：
  - E5 文本质量深挖：三路特征（surface/TF-IDF/E5）全量 5,000 样本，全部 ΔAUC<0.01，文本 uplift 不做最终定论
  - SHAP 全量模型解释：V10 正式报告出炉 `reports/v10_shap_explanation/`（7 图 + README + 可复现脚本）
- **V10 SHAP 报告核心发现**：
  - SHAP vs XGBoost Gain Spearman ρ=0.479——两者排名不一致，SHAP 有价值
  - 分类匹配特征占 69% 决策权重，个人硬指标 31%
  - 最强交互：target_uni × bg_uni (0.187)，且与调整链惩罚热点重叠
  - NaN GPA 案例被 SHAP 抓到过度推高→全被拒，问责 DEC-010
- TODO-6 的 6-2（Bootstrap CI）判定不能预计算，转为在线功能；6-3（gpu_hist）判定跳过
- **清理**：`scripts/analyze_text_quality.py` 的 N_ANALYSIS=5000/batch_size=256 是 5090 专属，办公电脑上跑会慢，但保留不 revert（用于标注"在家跑"）

**下个 session 起点**：
- TODO-5 问卷发放
- 或 TODO-2/3（TODO-1 已收工，依赖已解除）

---

## Session Handoff (2026-05-14 上午)

**本轮完成**：
- **TODO-5 调研问卷**：销售端 + 选校端两份问卷已完成，写入 `reports/signals_survey_sales.md` 和 `reports/signals_survey_selection.md`
  - 销售端：签单帮助 / 客户反应 / 界面冲击力 / 概率心态 / 竞品对比
  - 选校端：概率准确度 / 方案影响 / AI 解读使用 / 被替代顾虑 / 参与优化意愿
- 问卷设计遵循"业务语言、短、具体"原则，每份 5-8 分钟可填完
- TODO_ROUTE.md 新增 TODO-5 及 4 条产品风险假设

---

## Session Handoff (2026-05-13)

**本轮完成**：
- TODO-1 实施 5/6 项：Faculty 3-tier + per-layer ceiling + sim threshold + UI 不置信标记 + **预测失败兜底**
- 兜底方案：DataCompleteness 四级分类 → cascading population fallback (Wilson CI) → 调整链可用层 → UI 明确标识"历史统计估算"
- 新增 `fallback.py` (130 行) + `test_fallback.py` (27 tests)，全量 89 tests 零回归
- **TODO-4 Phase 1**：`pages/school_view.py` 独立页面上线 — 学校画像 + What-If模拟 + 学校对比，24 tests 通过
- ⚠️ TODO-4 页面**未在浏览器中验证**，标记为未检查

**下个 session 起点**：
- TODO-1 全部收工 ✅
- TODO-4 浏览器验证 → 修bug → 拼入 hk.py
- 或 TODO-2/3（依赖 TODO-1，现已可启动）

---

## Session Handoff (2026-05-14 工程注解)

**本轮完成**：`pages/hk.py` 及其全链路依赖的工程级注解（22 文件）。

| 阶段 | 文件 | 覆盖层 |
|------|------|--------|
| Phase 1 | `pages/hk.py` | 页面路由 + 状态机 + fragment trampoline + timeline |
| Phase 2 | `handler_config.py` `results_handler.py` `session_manager.py` | Session/Config/Key 基础设施 |
| Phase 3 | `input_form.py` `ui_elements.py` `content_display.py` `result_section.py` `submission_logger.py` `ghost_input/__init__.py` `page_components/__init__.py` | 表单 + UI 组件 + Ghost 输入 + 流式解释 + 双层缓存 |
| Phase 4 | `ui/handler.py` `page_data_loader.py` `cross_faculty_guard.py` `core/utils.py` `progress_reporter.py` `ui_messages.py` | Handler + 模型加载 + 跨学部守卫 + 进度报告 |
| Phase 5 | `flow/pipeline.py` `flow/run_prediction.py` | 预测管道 (prep→match→infer→deliver) + XGBoost 执行器 |
| Phase 6 | `flow/processor.py` `result_modifier/adjustment_pipeline.py` | 候选池构建 + 三路分拣 + Agent 平衡 + 5 层调整链 |

**剩余待注解**（算法深入层）：
- `result_modifier/probability_adjuster.py` — GPA/语言惩罚数学（numba jit）
- `result_modifier/arbitrator.py` — 衰减仲裁器
- `result_modifier/fallback.py` — Wilson CI 级联兜底
- `result_modifier/faculty_filters.py` — 学部规则 + 三级 severity
- `flow/result_processor.py` — 单条结果处理器

**注解风格约定**：模块级 docstring 说架构意图，函数级说调用时机+副作用+返回语义，关键模式（trampoline/fragment/双层缓存/evidence Bayes）展开 WHY。
