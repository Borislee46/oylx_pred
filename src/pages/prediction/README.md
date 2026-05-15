# Prediction 模块技术文档

## 1. 模块概述

`prediction` 是 Signals 留学选校系统的核心预测页面，完整覆盖：LeadIn 背景提取 → 三步式步骤条引导 → 表单填写/核验 → XGBoost 并行推理 → 多阶段概率调整 → 合并去重 → AI 流式解读（ExplainAgent）→ 数据对比。用户填写背景信息与目标选择后，系统输出相似专业、跨专业、用户指定三类推荐结果。

---

## DS 视角

预测流水线本质上是把原始输入送进一个 9 步变换链，每一步都在叠加新假设：XGBoost 假设 IID、CalibratedClassifierCV 假设训练集的标签分布代表真实世界、5 层调整链假设每种偏差可以用乘法因子独立修正。每一步单独看都有道理，但联合行为从未被当成一个整体来设计——这是典型的"不确定性叠加"，每个环节在解决自己那个问题时引入的误差，到链末端已经被放大到不可忽视。

一个关键数据事实：99.4% 的（院校 × 专业）组合只有 4 条或更少样本。这意味着模型几乎永远在做外推而非内插——它看到一个"港中文 + MSc Marketing"组合时，几乎不可能是从同组合的历史案例学到的，而是靠院校特征、专业相似度、GPA/语言等特征的泛化能力在猜。XGBoost 对稀疏组合的预测方差本身就不低，再叠上 5 层惩罚链，结果的不确定性会很大。

调整链的设计方式是逐层叠加的——今天加 GPA 惩罚、下个月加跨专业惩罚、再下个月加跨学部惩罚，每层上线时只验证了该层单独的表现（修复了当时的一个具体 bad case），但从未有人跑过全链路的系统性验证，直到 2026 年 5 月才第一次测 ECE。这意味着很多参数是在没有全局校准约束的情况下选定的，只保证了单层"看起来合理"。

## 2. 目录结构

```
prediction/
├── README.md                       # 本文档
├── input_form.py                   # 表单入口（create_input_form）
├── page_data_loader.py             # @st.cache_resource 全局模型单例
├── handler_config.py               # Session key 常量、FormSubmissionContext
├── results_handler.py              # 合并去重、状态重置
├── ai_report.py                    # AI 选校报告：匹配度环+梯度条+产品卡+解读渲染
├── ai_report_sections.py           # 流式渲染组件（overview/insight/school_notes/products）
├── ai_report_styles.py             # AI 报告全部 CSS（字体/动画/响应式）
├── ai_report_catalog.py            # 产品目录（PRODUCTS dict）
├── ai_school_stats.py              # Phase 2 数据对比：分位数排名+院校特征描述
├── api/
│   ├── __init__.py                 # predict, validate_and_normalize
│   └── json_api.py                 # 非 Streamlit 后端 API（表单校验→预测→调整）
├── config/
│   └── ui_messages.py              # 流水线提示文案
├── core/
│   ├── types.py                    # PredictionInput dataclass
│   ├── utils.py                    # 通用工具
│   └── exceptions.py               # MissingInputError
├── input_form_components/          # 表单子组件（详见子模块 README）
├── prediction_preparation/         # 输入校验与归一化
│   ├── __init__.py
│   ├── preparer.py                 # prepare_input_data, validate_and_clean_input
│   └── form_normalizer.py          # normalize_form_data_for_prediction
├── prediction_execution/           # 并行推理执行器
├── modeling/                       # XGBoost 模型封装
├── flow/                           # 预测流水线（详见 flow/README.md）
│   ├── pipeline.py                 # 主编排入口
│   ├── run_prediction.py           # 单次预测执行
│   ├── processor.py                # 组合生成 + 结果处理
│   ├── progress_reporter.py        # 进度上报
│   └── result_processor.py         # 相似度 + 语言惩罚
├── result_modifier/                # 5层概率调整链（详见子模块 README）
│   ├── adjustment_pipeline.py      # 管道主入口
│   ├── probability_adjuster.py     # GPA/语言偏差惩罚
│   ├── arbitrator.py               # 多因子仲裁（衰减合并）
│   ├── counterfactual.py           # 反事实扰动分析（"如果背景调整..."）
│   ├── filters.py                  # 同专业/跨专业推荐过滤
│   ├── faculty_filters.py          # 学部范围过滤
│   ├── text_boost_provider.py      # 背提文本加成接口
│   └── ...
├── result_display/                 # 结果表格 + trace 可视化 + delta 对比
│   ├── __init__.py
│   ├── results_display.py          # 表格配置与渲染（含 ±% delta 列）
│   ├── delta_calculator.py         # 跨提交概率对比（(uni,major) key 匹配）
│   ├── hero_summary.py             # 顶部摘要横幅：校徽墙 + 梯度分布条
│   ├── trace_display.py            # 概率调整链瀑布图 + 反事实
│   └── trace_assets.py             # trace CSS + 文案
├── usage_stats.py                  # 轻量使用计数 → cache/usage_stats.json（为 hot_paths 提供数据）
├── page_components/
│   ├── content_display.py          # 内容区调度：AI 解读 + Phase2 数据对比
│   ├── result_section.py           # 预测结果区（三类推荐表格 + hero_summary）
│   ├── ui_elements.py              # LeadIn 面板、步骤条、思想气泡
│   ├── submission_logger.py        # 表单提交日志
│   ├── display_helpers.py          # 展示辅助
│   └── pdf_generation/             # PDF 报告生成器（基于 reportlab）
├── ghost_input/                    # Cursor 风格灰字补全（详见子模块 README）
│   ├── __init__.py                  # Streamlit 组件声明 + 院校白名单注入
│   └── frontend/index.html          # 前端：缓存/规则/LLM 三层补全 + 分段逐出
├── data_sort_config/               # 列配置与排序
├── ui/
│   └── handler.py                  # handle_form_submission + 跨学部确认
├── school_combination_optimizer_algorithm/  # NSGA-III 选校优化器（实验性）
│   └── optimizer/                  # 优化器子模块（10 文件，2026-05 新增）
└── result_modifier/providers/      # logit_uplift / text_boost（详见子模块 README）
    ├── logit_uplift_provider.py     # TextBoostProvider 主实现
    └── logit_uplift/               # TF-IDF + 信号评分 + Δ 计算
```

---

## 3. 端到端流程

```
顾问自由文本（LeadIn Panel）
    │
    ▼
LeadInAgent.run(StudentContext)
    ├── NLU：碎片文本 → extracted_background
    ├── quick_assessment：方向性评估
    └── suggested_questions：追问建议
    │
    ▼
apply_lead_in_to_form(form_bridge.py)
    ├── 模糊匹配院校/专业 → st.session_state widget
    ├── 设 GPA / 语言 / 目标 / 经历 widget
    └── 写 lead_in_form_summary → expander 标题
    │
    ▼
[Expander: 核验/修改表单]
    ├── FormUIComponents 渲染各区块
    ├── FormValidator 校验
    └── normalize_form_data_for_prediction
    │
    ▼
用户点预测按钮
    │
    ├── st.rerun() 先重绘页面 → 步骤条 Step3 点亮 "active"
    │     └── 二次渲染：hk_ui_phase="running" → _run_prediction()
    │
    ├── 跨学部检测 (quick_cross_faculty_check)
    │     └── 若跨学部且未确认 → hk_ui_phase="awaiting_confirm"
    │
    ├── handle_form_submission (ui/handler.py)
    │     ├── prepare_input_data
    │     ├── has_meaningful_experience_text (背提 LLM 校验)
    │     └── run_prediction_with_guard
    │
    ▼
run_prediction_pipeline_with_progress (flow/pipeline.py)
    ├── 加载模型、校验 cases_df 指纹
    ├── run_single_prediction
    │     ├── generate_prediction_combinations
    │     ├── PredictionExecutor.execute_parallel
    │     └── process_prediction_results
    │           ├── SingleResultProcessor (相似度 + 语言惩罚)
    │           ├── get_similar_major_recommendations
    │           ├── get_cross_major_recommendations
    │           └── BoundaryCaseAgent 平衡微调
    ├── ProbabilityAdjustmentPipeline.adjust_batch
    │     (GPA惩罚→语言惩罚→跨专业×0.5→跨学部×0.3→职业学位→TF-IDF提升)
    ├── combine_and_deduplicate_results
    └── 写入 session: prediction_results + has_predicted=True
    │
    ▼
[预测结果表格] (display_results_section)
    │
    ▼
[AI 选校解读] 按钮 → 点击触发 ExplainAgent
    ├── render_static_frame(): 匹配度环 + 选校梯度条 + 产品匹配卡
    ├── ExplainAgent.stream(ctx):
    │     ├── classify_profile() → 选择 System Prompt (strong_elite/medium_mixed/weak_gaps/cross_major)
    │     ├── _build_explain_prompt(): 学生背景 + 预测结果 + 产品
    │     └── DeepSeek 流式输出 JSON (overview/strengths/concerns/summary/school_notes/products)
    ├── render_ai_section_streaming(): 逐段揭示（顾问解读→优势→需关注→总结→院校简析→推荐说明）
    │     └── 卡片末尾 "AI解读中..." 脉冲动画
    └── Phase 2: SchoolFeatureStats 数据对比
    │     └── GPA/语言成绩在每所目标校录取者中的分位数排名
    │
    ▼
[Trace 可视化] — 结果表格中展开单条结果
    ├── 瀑布图：逐层展示概率调整链的各因子贡献（51_trace.css 动画）
    ├── 反事实：GPA/语言/实习扰动后重跑调整链
    └── 校准指标：Brier / AUC / 阳性率偏差

[ApplicationAgent] 申请策略生成（待集成到 UI）
    └── 基于全量背景 + 预测结果 → 行动建议 + 时间线
```

---

### 3.1 关键假设 & 批判性问题

以下不是 bug，而是设计决策。每个在当时都有理由，但从 DS 视角需要追问：

- **GPA 惩罚用二次函数（系数 0.15），语言惩罚用阶梯函数——为什么不同？** 二次函数意味着"GPA 越低惩罚加速越快"，阶梯函数意味着"IELTS 6.0 和 6.5 之间没区别，但 5.5 到 6.0 有一条硬线"。两种函数形式选择有 DS 层面的统一逻辑吗？
- **5 层调整链的联合效应是否验证过？** DEC-007 仲裁器加进来之前，有没有人跑过全链路乘法叠加的 worst-case 分析？五层最坏情况折到 0.7×0.5×0.3 = 0.105——这个极端衰减是有意的还是意外的？
- **跨专业阈值 similarity < 0.89 从哪来的？** 是数据驱动还是经验拍板？similarity=0.88 和 0.90 两条结果经历完全不同的惩罚路径，这个 cliff effect 有没有做过敏感性分析？
- **缺失 GPA/语言用中位数填充——缺失性本身可能是信号。** 不填 GPA 的学生更可能是 GPA 低才不填。用中位数等于默认给他们一个"平均生"的 GPA。有没有对比过缺失 GPA 学生和已知低 GPA 学生的录取率？
- **没有 held-out test set。** CalibratedClassifierCV 的校准、ECE 的测量、所有调整链参数的选择，全在训练数据上完成。在未见过的组合上——占 99.4%——参数表现是未知的。

### 3.2 DS Known Issues

- **ECE=0.1155 严重失校准**：模型输出"40%"实际只对应约 29% 录取率——一个未校准的概率比没有概率更有害，制造虚假精确感
- **偏差不对称**：C9 被低估 18pp，双非只被低估 6pp——乘法惩罚对高基础概率区间影响更大，从公平性角度产生反向偏差
- **外部 ApplySquare -67pp**：外部数据相似度匹配差 → 更多惩罚触发 → 乘性衰减放大——调整链在未见过的数据上问题加速恶化。Compass 17K 行外部验证已纳入 `data_quality/` 测试套件
- **模型天花板 0.72**：仲裁器 70% 上限人为压制了所有输出——没有任何学生能拿到 >72% 的概率，不管背景多强
- **Trace 只解释调整链，不解释模型**：用户看到的是"因为惩罚了你的 GPA"，而不是"历史上类似背景的学生有 X% 被录取"。`boundary_explainer_design.md` 设计了边界解释器但未实现
- **6% 完全预测失败**：无 GPA + 冷门背景 → 静默失败，`fallback.py` 已提供基于 Wilson CI 的级联兜底（TODO-1 Item 5），但尚未集成到 UI

## 4. 核心组件

### 4.1 页面入口（pages/hk.py）

薄路由层，负责：
- **步骤条状态机**：`_step_class()` 级联逻辑，Step1/2/3 的 active/done 由 `has_lead_in`、`form_expander`（`st.expander` 原生 key）、`hk_ui_phase` 三者驱动
- **预测触发分离**：form submit → 设 `hk_ui_phase="running"` → `st.rerun()` → 二次渲染时执行 `_run_prediction()`，确保步骤条在预测阻塞前已更新到 Step3 active
- **进度思想气泡**：`render_thought_bubble_with_wait_pulse()` 在 `st.status` 内逐条展示 pipeline 日志

### 4.2 input_form

**create_input_form** → `(is_new_submission, input_data, all_unis, all_majors, original_form)`

- 表单校验通过 → `_process_successful_submission` → `normalize_form_data_for_prediction` → `st.rerun()`
- 校验失败 → toast 提示 → `reset_prediction_results`

### 4.3 page_data_loader

**machine_learning_model**：`@st.cache_resource` 全局单例，持有：
- `prediction_model`（XGBoost + CalibratedClassifierCV sigmoid prefit）
- `loaded_feature_names`、`cases_df`、`cases_df_fingerprint`
- `background_universities`、`target_base_df`、`university_country_map`
- `boundary_agent`（BoundaryCaseAgent 实例）

预热策略（2026-05）：`resource_loader()` 内已调用 `load_bg_target_similarity_cache()`，相似度缓存 + 相关性矩阵（`cache/correlation_matrix.feather`）+ pair weight 矩阵在页面首次加载时即写入 `@st.cache_data`，上午/下午高峰第一个用户的首次预测无需额外等待。

### 4.4 ui/handler

**handle_form_submission**：接收 `FormSubmissionContext`：
- `session_manager`、`page_state`、`input_data_from_form`
- `all_universities_target`、`all_majors_target`、`original_form_data`
- 成功：写入 `prediction_results` + `has_predicted=True`
- 失败：`reset_prediction_results`

### 4.5 flow（预测流水线）

详见 `flow/README.md`。关键路径：
1. `_execute_prediction_pipeline`：加载→校验→组合生成→并行推理→概率调整→去重
2. 概率调整链：GPA 惩罚 → 语言惩罚（含 L3.5 新档位） → 跨专业 ×0.5 → 跨学部 ×0.3 → 职业学位降级 → TF-IDF 文本提升
3. `_adjustment_trace` 记录每条结果的调整历史，ExplainAgent 用其区分 penalty/boost
4. **热门组合快速路径**（2026-05）：`config/hot_paths.json` 中的热门专业子串直接命中，跳过语义相似度查表 + fuzzy 匹配
5. **增量计算**（2026-05）：目标院校/专业未变时，复用上次组合列表跳过 `generate_prediction_combinations()`，仅重跑 XGBoost 推理 + 调整链

### 4.6 page_components

**content_display.py**：
- `display_content()`：调度结果区 + AI 解读区 + Trace 可视化
- `_render_ai_explanation()`：缓存检查 → 按钮门控（disabled 模式）→ 静态框架 → 流式生成 → 磁盘持久化
- `_stream_explain_content()`：流式循环 + 局部 JSON 提取（括号计数）+ 25ms/6字双阈值节流 + 同步降级
- `_render_unified_school_cards()`：统合渲染学校卡片（概率 + AI 备注 + 百分位数据）

**ui_elements.py**：
- `render_lead_in_panel()`：步骤条 + LeadIn 输入区 + 已提取信息展示
- `render_thought_bubble()` / `render_thought_bubble_with_wait_pulse()`：预测进度气泡
- 步骤条级联规则：
  ```
  Step1 ← has_lead_in | form_open | predicting → done（否则 active）
  Step2 ← predicting → done | form_open → active
  Step3 ← predicting → active
  has_predicted → 全部 done
  ```

### 4.7 AI 选校报告（ai_report.py / ai_report_sections.py / ai_report_styles.py）

**render_static_frame()**：返回 `(html, products)` 元组
- 匹配度环形图（`_compute_match`：top3 均值 + P90-P10 离散度 + 背景健康分）
- 选校梯度条（较稳/适中/冲刺，动态分位点 `_p(probs, 33/66)`）
- 产品匹配卡（`_build_products`：根据 GPA/语言/经历/跨专业自动推荐）

**render_ai_section()**：最终渲染完整 AI 解读
- `streaming=True` 时 overview 带 `ar-streaming` 类，末尾追加 "AI解读中..." 脉冲动画
- `streaming=False` 时纯静态展示

**render_ai_section_streaming()**：流式逐段揭示
- 字段级别的 `seen_fields` 追踪，首现时有 `ar-section-enter` 入场动画
- 末尾始终追加 "AI解读中..." + 三点脉冲（复用 `hk-thought-wait-d1/2/3` CSS 动画）

**字号体系**：hero 1.25rem | body 0.8rem | label 0.68rem（3 档）

### 4.8 ai_school_stats.py（Phase 2 数据对比）

**SchoolFeatureStats**：
- 从 `cases_df` 按 `target_university` 分组计算 `NUMERIC_FEATURES` (gpa/language_score/research_count/internship_count/award_count/paper_count) 的分位数（P5/P10/P25/P50/P75/P90/P95）
- `get_percentile(university, feature, value)` → 分位数
- `get_rank_label(university, feature, value)` → "偏低"/"中等"/"较高"
- `generate_per_school_texts(student_feat, top_results, top_n=3)` → 对比文本列表

### 4.9 results_handler

**combine_and_deduplicate_results**：优先级 `user_specified > cross_major > similarity`，同优先级取概率更高者。

**reset_prediction_results**：清空所有预测相关 session state。

### 4.10 result_display（表格 + Trace 可视化 + Delta 对比）

**results_display.py**：
- `get_column_config()`：动态列配置（概率、相似度、院校排序、专业详情）
- 三类推荐表格：`TOP_SIM_RESULT_UI_CONFIG`、`TOP_CROSS_RESULT_UI_CONFIG`、`UNIVERSITY_SORT_ORDER`
- **Delta 对比列**（2026-05）：复用上次预测结果的 `(university, major)` 匹配，在 dataframe 中追加 `±%` 列（pinned，紧挨概率条右侧），绿涨红跌蓝 NEW——纯展示层改动，不碰 pipeline 核心

**delta_calculator.py**（2026-05 新增）：
- `should_show_delta()`：判断是否有上次结果、目标是否重叠。海投模式（目标为空）默认全量比较。`result_section.py` 入口处 `_normalize_target_list()` 同时处理 list 和逗号分隔 string 两种 input_data 格式
- `calculate_delta()`：`(university, major)` key 匹配，返回 `"+3.2%"` / `"-1.5%"` / `"NEW"` / `"—"`。diff < 0.5pp 视为不变
- 所有 `float()` 转换经 `_safe_float()` 防护，`None`/非数字 → 0.0，不会因脏数据崩溃
- 每次提交前在 `handle_form_submission` 中快照旧 `prediction_results` + `input_data` 到 `previous_*` session keys。快照在 `persist_input_state()` **之前**执行，保证拿到旧 input_data 而非已被覆盖的新值

**trace_display.py**：
- `render_trace_for_results()`：概率调整链可视化——对 top 3 结果展示瀑布图（GPA→语言→跨专业→跨学部→职业学位→文本提升各层贡献），含 baseline 历史录取率对比虚线
- 反事实（counterfactual）：GPA/语言/实习维度小幅扰动后重跑核心调整链，回答"如果背景再好/差一点"
- CSS 动画（`trace_assets.py`）：30 秒讲完一个 case 的完整链路故事
- 标题标注「开发者 Trace (beta)」

### 4.11 api/json_api（非 Streamlit 后端 API）

**`predict()`**：独立于 Streamlit 的预测 API：
- 表单校验（FormValidator）→ 归一化（normalize_form_data_for_prediction）→ 输入准备（prepare_input_data）→ 执行预测（run_single_prediction）→ 概率调整（ProbabilityAdjustmentPipeline）
- 支持跨学部检测（quick_cross_faculty_check）、新专业识别（is_new_major）、录取组合缓存
- 每次调用生成 `request_id`（UUID），记录总耗时

**`validate_and_normalize()`**：仅校验 + 归一化，不执行预测。

### 4.12 usage_stats（使用统计，2026-05 新增）

轻量级预测组合计数器，为下版本 `hot_paths.json` 提供数据支撑。

**设计要点**：
- **触发**：每次预测成功后 `increment(unified_results)`，对每条结果的 `(university, major)` 计数 +1
- **排除**：`lijiapeng8@xdf.cn`（开发/调试）不计数；写入失败记录 warning 不阻塞预测
- **存储**：`cache/usage_stats.json`，格式 `{"香港中文大学|MSc Accounting": 42, ...}`，compact JSON（无空格换行）
- **裁剪**：超过 2000 条唯一组合时，按计数降序保留前 2000，防止文件膨胀
- **线程安全**：`threading.Lock()` 保护读写，Streamlit 多 session 并发安全
- **消费**：admin 页面「使用统计」tab → 查看 Top 50 + 下载 JSON → 手动更新 `config/hot_paths.json` → git commit → 随下一版本部署生效

---

## 5. AI 解读数据流

```
用户点 "AI 选校解读"
    │
    ├── cache_key = _build_explain_cache_key(v=3, profile, compact_results)
    │     └── MD5 hash，命中则直接渲染（跳过 LLM 调用）
    │
    ├── render_static_frame() → (html, products)
    │     ├── _compute_match(): 匹配度 0-100
    │     ├── _build_products(): 产品推荐列表
    │     └── st.html() 渲染卡片 + st.session_state["_ar_match"] / "_ar_products"
    │
    ├── StudentContext(stage="match", ..., prediction_results, matched_products)
    │
    ├── classify_profile(sim, cross, unified)
    │     ├── cross >= 40% → "cross_major"
    │     ├── avg_prob >= 0.55 & penalty ≤ 1 → "strong_elite"
    │     ├── avg_prob >= 0.30 & penalty ≤ 3 → "medium_mixed"
    │     └── 否则 → "weak_gaps"
    │
    ├── ExplainAgent._prepare(ctx) → system_prompt (profile-specific) + data_prompt
    │
    ├── ExplainAgent.stream(ctx)
    │     ├── _call_api_streaming(max_tokens=700, timeout=20s)
    │     ├── _try_extract_partial(buffer): 括号计数 JSON 提取器
    │     └── render_ai_section_streaming(merged): 每 25ms/6字刷新 UI
    │
    ├── 流式完成 → parse_stream_result()
    │     ├── 四级 JSON 修复: direct → regex → json_repair lib → API repair
    │     └── schema: {"overview","strengths","concerns","summary","school_notes","products"}
    │
    ├── 成功 → 磁盘持久化 + session_state 缓存
    │     ├── _render_unified_school_cards(): 院校卡片（模糊匹配学校名）
    │     └── render_ai_section(result): 最终静态渲染
    │
    └── 失败 → 流式无输出时自动降级到 ExplainAgent.run()（同步路径，timeout=20s）

[Tail] Trace 可视化（与 AI 解读并列，结果表格内展开）
    ├── counterfactual.baseline_admit_rate: 历史该专业平均录取率
    ├── counterfactual.compute_counterfactuals(): GPA/语言/实习扰动 → 重跑调整链
    └── render_trace_for_results(): 瀑布图 + 反事实 + 校准指标
```

---

## 6. Session State 关键键值

| Key | 类型 | 说明 |
|-----|------|------|
| `has_predicted` | bool | 预测是否已完成（驱动步骤条和结果区） |
| `prediction_results` | PredictionResultModel | sim/cross/user/unified 四组结果 |
| `hk_ui_phase` | str | idle / running / awaiting_confirm / done / error |
| `form_expander` | bool | `st.expander(key="form_expander", on_change="rerun")` 原生开关状态 |
| `lead_in_ctx` | StudentContext | LeadIn 全链路上下文（含 extracted_background） |
| `lead_in_form_summary` | str | LeadIn 摘要文本，非空时 expander 展开 |
| `explain_cache` | dict | AI 解读缓存（MD5 key → result dict），磁盘持久化 |
| `explain_generating` | bool | 是否正在生成 AI 解读（控制按钮 disabled 状态） |
| `_ar_match` | float | 最近一次匹配度分数 |
| `_ar_products` | list[dict] | 最近一次产品推荐列表 |
| `_pending_submission_data` | dict | st.rerun 前暂存的提交数据（预测触发分离） |
| `previous_prediction_results` | PredictionResultModel | 上一次预测结果（delta 对比用） |
| `previous_input_data` | dict | 上一次提交的表单数据（delta 对比用） |
| `cached_combinations` | list | 增量计算：上次预测的 (uni, major) 组合缓存 |
| `prediction_submit_lock` | bool | 提交锁，防止重复提交 |
| `cross_faculty_confirmed` | bool | 跨学部确认状态 |
| `explain_cache` | dict | AI 解读磁盘缓存（MD5 → result） |

---

## 7. 子模块文档

| 子模块 | 文档路径 |
|-------|----------|
| input_form_components | `input_form_components/README.md` |
| flow | `flow/README.md` |
| result_modifier | `result_modifier/README.md` |
| result_modifier/providers | `result_modifier/providers/README.md` |
| result_display | `result_display/README.md` |
| ghost_input | `ghost_input/README.md` |
| school_combination_optimizer_algorithm | `school_combination_optimizer_algorithm/README.md` |
| page_components/pdf_generation | `page_components/pdf_generation/README.md` |

---

## 8. 依赖

- `streamlit==1.56.0`：UI 框架（`st.expander` on_change 跟踪、`st.status` 阻塞容器）
- `pandas` / `numpy`：数据处理
- `xgboost-cpu`：模型推理
- `numba`：JIT 加速数值计算（result_modifier 调整链）
- `rapidfuzz`：模糊匹配（院校/专业）
- `json-repair`：LLM JSON 输出快速修复（<5ms，替代 API 兜底）
- `SessionManager`：强类型 session state 封装
- `ExplainAgent` / `LeadInAgent` / `ApplicationAgent`：LLM Agent（DeepSeek）
- `BoundaryCaseAgent`：边界案例决策
- `SchoolFeatureStats`：院校历史数据分位数统计
- `school_alias_resolver`：院校别名解析（985→北京大学，港3→港校列表）
- `config/hot_paths.json`：热门组合快速路径配置
- `cache/correlation_matrix.feather`：Monte Carlo 相关性矩阵（选校优化器）
