# Prediction Flow 技术文档

仓库级契约与 JSON 入口说明见 [prediction_api.md](../../../../docs/prediction_api.md)（§3–4）。本文聚焦 `flow/` 目录内函数、缓存与 `meta.error` 等实现细节；上层包结构见 [../README.md](../README.md)。

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

**错误**：`meta.error` 可能为 `model_load_failed`、`cases_df_fingerprint_mismatch`、`no_valid_combinations`、`model_unavailable`、`missing_features`、`execution_failed`、`empty_results`。

### 4.2 run_prediction

**run_single_prediction**：单次预测执行。

1. 校验/补全 `PredictionInput`
2. `generate_prediction_combinations` 生成 (university, major) 组合
3. `prepare_model_inputs` 构建模型输入
4. `PredictionExecutor.execute_parallel` 并行推理
5. `process_prediction_results` 处理原始输出，得到 sim_rec、cross_rec、user_results

### 4.3 processor

**generate_prediction_combinations**：
- 目标专业：用户指定则用 `target_majors`；否则从 `all_majors_target` 中筛选，语义相似度 ≥ 0.6 或 fuzzy 匹配 > 90 的纳入
- 目标院校：`target_universities` 或 `all_universities_target`
- 过滤：仅保留 `_data_manager.valid_combinations` 中的 (u, m)

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
