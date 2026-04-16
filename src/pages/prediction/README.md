# Prediction 模块技术文档

## 1. 模块概述

`prediction` 是录取概率预测页面的完整实现，涵盖表单输入、数据校验、模型推理、结果修饰、跨学部确认、结果展示等全流程。用户填写背景信息与目标选择后，系统基于 XGBoost 模型与历史案例，输出相似专业、跨专业、用户指定三类推荐结果，并应用 GPA/语言惩罚、背提文本加成、Agent 边界微调等多维度调整。

## 2. 目录结构

```
prediction/
├── README.md                    # 本文档
├── input_form.py                # 表单入口与提交逻辑
├── page_data_loader.py          # 模型与资源加载
├── handler_config.py            # 会话 key、FormSubmissionContext
├── results_handler.py           # 结果合并去重、状态重置
├── config/
│   └── ui_messages.py           # 流水线提示文案
├── core/
│   ├── types.py                # PredictionInput
│   ├── utils.py                # 通用工具
│   └── exceptions.py            # MissingInputError
├── input_form_components/      # 表单组件（详见 input_form_components/README.md）
├── prediction_preparation/     # 输入校验与准备
│   ├── preparer.py
│   └── form_normalizer.py
├── prediction_execution/       # 模型并行执行
│   └── executor.py
├── modeling/                    # XGBoost 模型封装
│   └── model.py
├── flow/                        # 预测流水线（详见 flow/README.md）
├── result_modifier/             # 结果修饰（详见 result_modifier/README.md）
├── result_display/              # 结果展示
│   └── results_display.py
├── page_components/             # 页面区块
│   ├── content_display.py       # 内容区调度
│   ├── result_section.py        # 结果区渲染
│   ├── ui_elements.py
│   └── submission_logger.py
├── data_sort_config/            # 结果排序与列配置
│   └── config.py
└── ui/
    └── handler.py               # 提交处理与跨学部确认
```

## 3. 端到端流程

```
用户填写表单 (input_form.py)
        │
        ├── FormUIComponents 渲染各区块
        ├── FormValidator 校验
        └── normalize_form_data_for_prediction
        │
        ▼
提交 (submit_button)
        │
        ├── 跨学部检测 (quick_cross_faculty_check)
        │       └── 若跨学部且未确认 → cross_faculty_confirm_dialog
        │
        ├── handle_form_submission (ui/handler.py)
        │       ├── prepare_input_data
        │       ├── has_meaningful_experience_text (背提 LLM 校验)
        │       └── run_prediction_with_guard
        │
        ▼
run_prediction_pipeline_with_progress (flow/pipeline.py)
        │
        ├── 加载模型、校验指纹
        ├── run_single_prediction
        │       ├── generate_prediction_combinations
        │       ├── PredictionExecutor.execute_parallel
        │       └── process_prediction_results
        │               ├── SingleResultProcessor (相似度、语言惩罚)
        │               ├── get_similar_major_recommendations
        │               ├── get_cross_major_recommendations
        │               └── Agent 平衡微调
        │
        ├── ProbabilityAdjustmentPipeline.adjust_batch (GPA/语言/跨专业/学部/背提)
        ├── combine_and_deduplicate_results
        └── 写入 session_manager.prediction_results
        │
        ▼
display_content → display_results_section → ResultsDisplay
```

## 4. 核心组件

### 4.1 input_form

**create_input_form**：创建完整表单，返回 `(submitted, input_data, all_unis, all_majors, original_form)` 或 `(False, input_data, ...)`。

- 校验通过后：`_process_successful_submission` → `normalize_form_data_for_prediction`，将 `_input_form_pending_submission` 写入 session，`st.rerun()` 后由调用方取走
- 校验失败：toast 提示，`reset_prediction_results`，`st.rerun()`

### 4.2 page_data_loader

**machine_learning_model**：`@st.cache_resource` 全局单例，持有：
- `prediction_model`：XGBoost 模型
- `loaded_feature_names`：特征列表
- `cases_df`、`cases_df_fingerprint`
- `background_universities`、`target_base_df`、`university_country_map`
- `boundary_agent`：BoundaryCaseAgent

**cached_get_prediction_model**：`@st.cache_resource` 缓存模型加载。

### 4.3 ui/handler

**handle_form_submission**：接收 `FormSubmissionContext`，执行跨学部检测、输入准备、`run_prediction_with_guard`，成功时写入 `prediction_results`、`has_predicted=True`。

**run_prediction_with_guard**：组装 `_all_universities_target`、`_all_majors_target`、`_cross_faculty_confirmed`、`_has_valid_experience`，调用 `run_prediction_pipeline_with_progress`，失败时 `reset_prediction_results`。

### 4.4 prediction_preparation

- **validate_and_clean_input**：清洗为 `PredictionInput`
- **prepare_input_data**：补充 `school_level`、`faculty`
- **prepare_model_inputs**：按 `expected_features` 构建模型输入，返回缺失特征列表
- **get_user_specified_combinations**：用户指定 (university, major) 组合
- **compute_list_fingerprint** / **compute_df_fingerprint**：用于缓存 key

### 4.5 prediction_execution

**PredictionExecutor**：组合数 ≤ `PREDICTION_SINGLE_THREAD_THRESHOLD`（默认 2048）时单线程；否则按 chunk 并行，支持 `ThreadPoolExecutor` 或 `ProcessPoolExecutor`。

### 4.6 results_handler

**combine_and_deduplicate_results**：按 (university, major) 去重，优先级 `user_specified > cross_major > similarity`，同优先级取概率更高者。

**reset_prediction_results**：清空 `prediction_results`、`prediction_submit_lock` 等。

### 4.7 result_display

**ResultsDisplay**：将 sim/cross/user_specified 三类结果按配置列展示，支持 `TOP_SIM_RESULT_UI_CONFIG`、`TOP_CROSS_RESULT_UI_CONFIG`。

### 4.8 page_components

**display_content**：当 `has_predicted` 时，从 session 取 `input_data`、`prediction_results`，调用 `display_results_section`。若 `form_data_changed` 且未提交，显示“输入已更改”提示。

## 5. 数据流

**表单原始数据** → `normalize_form_data_for_prediction` → **PredictionInput**（含 `gpa`、`language_score` 归一化）→ **prepare_input_data** → **run_prediction_pipeline** → **PredictionResultModel**（similarity_results、cross_major_results、user_specified_results、unified_results、meta）。

## 6. 子模块文档

| 子模块 | 文档路径 |
|-------|----------|
| input_form_components | `input_form_components/README.md` |
| flow | `flow/README.md` |
| result_modifier | `result_modifier/README.md` |
| result_modifier/providers | `result_modifier/providers/README.md` |

## 7. 依赖

- `streamlit`：UI
- `pandas`、`numpy`：数据处理
- `xgboost`：模型
- `rapidfuzz`：模糊匹配
- `SessionManager`：会话状态
- `school_level_service`：院校等级、海外判定
- `BoundaryCaseAgent`：边界案例决策
