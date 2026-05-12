# Prediction Flow 技术文档

## 1. 模块概述

`flow` 是预测页面的执行流程模块，负责从输入数据到最终推荐结果的完整流水线：组合生成、模型推理、结果处理、过滤排序、概率调整、Agent 微调等，并通过 `ProgressReporter` 向 UI 反馈进度。

## 2. 目录结构

```
flow/
├── __init__.py
├── pipeline.py         # 主流水线入口
├── run_prediction.py    # 单次预测执行
├── processor.py        # 组合生成 + 结果处理
├── result_processor.py # 单条结果处理（相似度、语言惩罚等）
└── progress_reporter.py # 进度回调
```

## 3. 核心流程

```
run_prediction_pipeline / run_prediction_pipeline_with_progress
        │
        ▼
_execute_prediction_pipeline
        │
        ├── 1. 加载模型 (cached_get_prediction_model)
        ├── 2. 校验 cases_df 指纹
        ├── 3. validate_and_clean_input
        ├── 4. load_bg_target_similarity_cache
        ├── 5. 构建 ProbabilityAdjuster
        ├── 6. run_single_prediction
        │       ├── generate_prediction_combinations → (university, major) 组合
        │       ├── prepare_model_inputs
        │       ├── PredictionExecutor.execute_parallel
        │       └── process_prediction_results
        │               ├── SingleResultProcessor.process (逐条)
        │               ├── get_similar_major_recommendations
        │               ├── get_cross_major_recommendations
        │               └── _apply_agent_balance_adjustment_flat
        ├── 7. ProbabilityAdjustmentPipeline.adjust_batch (相似/跨专业/用户指定)
        ├── 8. combine_and_deduplicate_results
        └── 9. 返回 PredictionResultModel
```

## 4. 组件说明

### 4.1 pipeline

**run_prediction_pipeline**：`@st.cache_data(ttl=600)` 缓存版本，无进度回调，用于后台或测试。

**run_prediction_pipeline_with_progress**：带 `progress_cb` 的版本，用于 UI 实时展示进度。

**输入**：
- `input_data`：含 `_all_universities_target`、`_all_majors_target`、`_cross_faculty_confirmed`、`_has_valid_experience` 等
- `model_name`、`cases_df_fingerprint`、`loaded_feature_names`
- `all_universities_fingerprint`、`all_majors_fingerprint`（用于缓存 key）
- `background_faculty`、`admitted_combinations`、`page_state`

**输出**：`PredictionResultModel`（`similarity_results`、`cross_major_results`、`user_specified_results`、`unified_results`、`meta`）

**增量参数**（2026-05）：`cached_combinations: list[tuple[str, str]] | None` — 非 None 时跳过组合生成，直接复用传入的 (university, major) 列表。由 `handle_form_submission` 在目标未变时自动注入。

**错误**：`meta.error` 可能为 `model_load_failed`、`cases_df_fingerprint_mismatch`、`no_valid_combinations`、`model_unavailable`、`missing_features`、`execution_failed`、`empty_results`。

### 4.2 run_prediction

**run_single_prediction**：单次预测执行。

1. 校验/补全 `PredictionInput`
2. `generate_prediction_combinations` 生成 (university, major) 组合（若传入 `cached_combinations` 则跳过此步）
3. `prepare_model_inputs` 构建模型输入
4. `PredictionExecutor.execute_parallel` 并行推理
5. `process_prediction_results` 处理原始输出，得到 sim_rec、cross_rec、user_results

**Session 内增量计算**（2026-05）：当同一 session 内目标院校/专业未变化时，`handle_form_submission` 从上次 `unified_results` 提取 `(university, major)` 组合作为 `cached_combinations` 传入 pipeline，跳过 `generate_prediction_combinations()`（含 E5 相似度查表 + fuzzy 匹配），仅重跑 XGBoost 推理 + 调整链。目标有变化时自动 fallback 到全量重算。

**设计取舍**：增量触发条件是「上次结果中的院校集合 == 当前 `all_universities_target`」。在 bulk 模式（未选具体学校）下，`all_universities_target` 是全量 26 所，而上轮结果只命中 8-10 所——二者不相等，增量不触发。这是保守策略：宁可多算，不拿错误组合。精确选校 + 调 GPA/语言的场景才是增量的主战场，且 popular-path（A5）已给 bulk 模式提了速。

### 4.3 processor

**generate_prediction_combinations**：
- 目标专业：用户指定则用 `target_majors`；否则从 `all_majors_target` 中筛选，语义相似度 ≥ 0.6 或 fuzzy 匹配 > 90 的纳入
- 目标院校：`target_universities` 或 `all_universities_target`
- 过滤：仅保留 `_data_manager.valid_combinations` 中的 (u, m)
- **热门组合快速路径**（2026-05）：`_is_major_match()` 对命中 `config/hot_paths.json` 中 `hot_major_substrings` 的专业直接返回 True，跳过语义相似度查表 + fuzzy 匹配。配置由 `_load_hot_paths()` 在模块加载时读取，文件不存在或损坏时 fallback 到硬编码默认值（港三 × SMART/ACCT/IT）。`hot_schools` 字段已预留，当前仅 `hot_major_substrings` 生效——学校级快速路径待 usage_stats 积累足够数据后启用

**process_prediction_results**：
- 构建 `SingleResultProcessor`，逐条处理
- `_get_user_specified_results` 提取用户指定组合结果，数量多时截断到 `USER_SPECIFIED_LARGE_RANGE_TOP_N`
- `get_similar_major_recommendations` 同专业推荐
- `get_cross_major_recommendations` 跨专业推荐
- `_apply_agent_balance_adjustment_flat`：当 sim 与 cross 数量差超过阈值时，调用 Agent 微调平衡

**_apply_agent_balance_adjustment_flat**：
- `diff = len(cross_rec) - len(sim_rec)`，阈值 `max(AGENT_MIN_BALANCE_DIFF_MIN, AGENT_MIN_BALANCE_DIFF_RATIO * max_len)`
- 若 `abs(diff) < threshold` 或 cross 过少，不调整
- 否则调用 `adjust_similarity_results_with_agent`（RelaxStrategy/TightenStrategy），从 sim_rec 补充或移除
- 去重：sim_rec 中已有的 (u, m) 从 cross_rec 中移除

### 4.4 result_processor

**SingleResultProcessor**：单条模型输出 → 可展示结果。

- **Part-time 过滤**：专业名或学习模式含 "part time" 不含 "full" 的丢弃
- **语言惩罚**：从 row 取 `_lang_reqs`，`LanguageRequirementPenalty.calculate_penalty` 计算，`probability *= penalty`
- **学部**：`faculty`、`major_cn` 从 details_df 补充，`_is_in_faculty_scope` 判断是否在 allowed_faculties
- **强匹配分**：`_strong_match_score` = rapidfuzz 对背景专业与目标专业（中英文）的最大相似度
- **相似度**：`get_cached_major_similarity` 取基础值，`adjust_similarity_score` 应用规则与模糊偏置

### 4.5 progress_reporter

**ProgressReporter**：节流进度回调。

- `emit(text, force=False)`：`text` 可为 str 或 list（随机选一条），`force=True` 跳过节流
- `min_interval` 默认 0.5s，避免 UI 刷新过于频繁

## 5. 数据流

**PredictionInput**：`background_university`、`background_major`、`target_universities`、`target_majors`、`gpa`、`language_score`、`language_type`、`internship_count`、`experience_details` 等。

**PredictionResultModel**：
- `similarity_results`：同专业推荐
- `cross_major_results`：跨专业推荐
- `user_specified_results`：用户指定组合结果
- `unified_results`：合并去重后的最终列表
- `meta`：错误码、用户提示等

## 6. 依赖

- `PredictionModel`、`PredictionExecutor`：模型与执行
- `prediction_preparation`：输入校验、组合准备
- `result_modifier`：过滤、调整、Agent
- `results_handler`：`combine_and_deduplicate_results`
- `page_data_loader`：`machine_learning_model`、`cached_get_prediction_model`
- `BoundaryCaseAgent`：边界案例决策
