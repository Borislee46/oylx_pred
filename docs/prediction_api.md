# 预测流水线 API

| 项 | 说明 |
|----|------|
| 源码路径 | `src/pages/prediction/` |
| UI 管线 | `flow/pipeline.py`：`run_prediction_pipeline`、`run_prediction_pipeline_with_progress` |
| JSON 入口 | `api/json_api.py`（与 Streamlit **同进程**，模块 docstring 标明非生产用途；独立部署需另起服务层） |

实现细节与调用链（`run_prediction_pipeline_with_progress`、`meta.error` 枚举等）见源码旁 [src/pages/prediction/README.md](../src/pages/prediction/README.md)、[src/pages/prediction/flow/README.md](../src/pages/prediction/flow/README.md)。

## 1. 输入类型 `PredictionInput`

定义位置：`src/pages/prediction/core/types.py`（`TypedDict`，`total=False`）。

由 `prediction_preparation/preparer.py::validate_and_clean_input` 从运行期字典清洗得到。常用键包括：

- `background_university`、`background_major`（字符串）
- `target_universities`、`target_majors`（字符串列表）
- `gpa`、`language_score`（可选浮点；`language_score` 为归一化到 [0,1] 后的值）
- `language_type`、`internship_count`、`research_count`、`award_count`、`paper_count`
- `experience_details`（`dict[str, str]`）
- `school_level` 等可由 `prepare_input_data` 注入

运行期 dict 中可能存在 `faculty`、`background_major_original` 等辅助字段，它们**不属于** `PredictionInput` 类型本身，供后续步骤使用。

## 2. 结果项（运行期字典）

单条推荐在流水线各阶段逐步补全，**无单一 Pydantic schema**。常见字段：

| 字段 | 说明 |
|------|------|
| `university`、`major` | 院校与专业 |
| `probability` | 模型或后处理后的概率 |
| `similarity` | 专业相似度（准备阶段注入） |
| `faculty` | 学部（处理阶段 metadata） |
| `is_new_major` | 经 `adjustment_pipeline` 注入 |

合并去重使用内部 `_source` / `_priority`，交付前会剥离。

## 3. JSON API（`api/json_api.py`）

### 3.1 `validate_and_normalize(payload, cases_df=None, school_base_df=None)`

将表单型 payload 校验并归一化为模型可用字段（GPA/语言、海外院校默认语言等）。失败返回：

```json
{"ok": false, "errors": [{"field": "...", "message": "...", "severity": "error"}], "warnings": []}
```

### 3.2 `predict(payload, confirm_cross_faculty=False)`

**请求**：含 `background_university`、`background_major`、`gpa_raw`、`language_score_raw` 等；跨院系相关：`selected_major_categories`、`selected_target_majors`；候选池可选：`all_universities_target`、`all_majors_target`。

**响应形态**：

1. 校验失败：同 `validate_and_normalize` 错误结构。
2. 需跨院系确认（未确认且守卫未自动放行）：`ok: true`，`needs_confirmation: true`，`confirmation` 含 `background_faculty`、`target_faculties`、`agent_approved` 等，`result` 可为 `null`。
3. 成功：`result` 内含 `similarity_results`、`cross_major_results`、`user_specified_results`、`unified_results`、`meta`（如 `combination_count`）。

### 3.3 `unified_results` 合并规则

按 (院校, 专业) 去重时：

1. **优先级 3**：用户指定（`user_specified_results`）
2. **优先级 2**：跨专业推荐（`cross_major_results`）— 仅当条目带 `admitted == 1` 时参与合并（与 `result_modifier/filters.py` 行为一致）
3. **优先级 1**：相似专业推荐（`similarity_results`）

同键多条时，高优先级覆盖低优先级；同优先级取 `probability` 较大者。

## 4. Streamlit 管线步骤（逻辑顺序）

1. **跨院系守卫**：`cross_faculty_guard.py`（与 JSON 路径的 `confirm_cross_faculty` 协同）。
2. **准备**：`prediction_preparation/preparer.py`、`form_normalizer.py`。
3. **召回与推理**：`flow/run_prediction.py::run_single_prediction`（含 E5/Fuzzy 等召回逻辑，以源码为准）；`prediction_execution/executor.py` 批量 `predict_proba`。
4. **处理**：`flow/processor.py`（元数据、目标专业语言门槛惩罚、相似度偏置、`BoundaryCaseAgent` 等）。
5. **后处理**：`result_modifier/adjustment_pipeline.py`（GPA/语言、跨专业/学部、文本 uplift 等）。
6. **合并**：`results_handler.py::combine_and_deduplicate_results`。

---

相关文档：[input_form_components_api.md](input_form_components_api.md)、[result_modifier_api.md](result_modifier_api.md)。

维护：与 `src/pages/prediction/` 同步更新。
