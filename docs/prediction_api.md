# 预测模块 API

**路径**: `src/pages/prediction`

此模块存在两条入口：
- **Streamlit 页面预测管线**：`src/pages/prediction/flow/pipeline.py::run_prediction_pipeline`
- **JSON 形态预测入口**（用于前后端解耦实验）：`src/pages/prediction/api/json_api.py::predict`

## 1. 内部输入类型 (`PredictionInput`)

代码定义：`src/pages/prediction/core/types.py`（由 `src/pages/prediction/prediction_preparation/preparer.py::validate_and_clean_input` 产出）

- `background_university: str`
- `background_major: str`
- `target_universities: list[str]`
- `target_majors: list[str]`
- `gpa: float`（可空）
- `language_score: float`（可空，归一化到 0~1）
- `language_type: str`（可选）
- `internship_count / research_count / award_count / paper_count: int`
- `school_level: str`（可选；由 `src/pages/prediction/prediction_preparation/preparer.py::prepare_input_data` 注入）
- `experience_details: dict[str, str]`

**注意**：`faculty`、`background_major_original` 等字段会存在于“运行期输入 dict”中，但不属于 `PredictionInput` 数据类。

## 2. 结果项结构（运行期字典）

结果不是严格 schema，会在不同阶段补齐字段：

- **基本字段**：`university: str`, `major: str`, `probability: float`
- **预处理注入**：`similarity: float`, `faculty: str`
- **页面管线额外注入**：`is_new_major: bool`

内部合并去重会使用 `_source/_priority` 元数据，但最终会剔除。

## 3. JSON 入口 (`src/pages/prediction/api/json_api.py`)

### 3.1 校验与归一化

`validate_and_normalize(payload, cases_df=None, school_base_df=None) -> dict`

- **作用**：把“表单式 payload”校验并归一化成模型可用字段（含 GPA/语言归一化、海外默认语言加成、标化加成等）。
- **校验失败返回**：`{"ok": false, "errors": [...], "warnings": []}`。
- `errors` 元素结构：`{"field": str, "message": str, "severity": "error"|"warning"}`。

### 3.2 预测

`predict(payload, confirm_cross_faculty=False) -> dict`

**请求 payload 字段**：
- **表单数据**：`background_university`, `background_major`, `gpa_raw`, `language_score_raw` 等。
- **跨院系检查**：`selected_major_categories`, `selected_target_majors`。
- **候选池**：`all_universities_target`, `all_majors_target`（可选）。

**返回结构（3 种典型形态）**：

1.  **校验失败**
    ```json
    {
      "ok": false,
      "errors": [{"field": "...", "message": "...", "severity": "error"}],
      "warnings": []
    }
    ```

2.  **需要跨院系确认（不是错误）**
    当检测到跨院系、且 Agent 未自动放行、且 `confirm_cross_faculty=false`：
    ```json
    {
      "ok": true,
      "needs_confirmation": true,
      "confirmation": {
        "background_faculty": "...",
        "target_faculties": ["..."],
        "agent_approved": false
      },
      "normalized_input": {...},
      "result": null
    }
    ```

3.  **预测成功**
    ```json
    {
      "ok": true,
      "needs_confirmation": false,
      "warnings": ["..."],
      "normalized_input": {...},
      "result": {
        "similarity_results": [...],
        "cross_major_results": [...],
        "user_specified_results": [...],
        "unified_results": [...],
        "meta": {"combination_count": ...}
      }
    }
    ```

### 3.3 关于 `unified_results` 的重要细节

`unified_results` 通常只会合并相似推荐与用户指定结果。跨专业推荐生成（`result_modifier/filters.py`）产生的结果通常不会写入 `admitted` 字段，因此不会被自动合并。

## 4. Streamlit 页面预测管线

**入口**：`src/pages/prediction/flow/pipeline.py::run_prediction_pipeline`

页面管线负责串联资源加载、并行推理、推荐生成与后处理：

1.  **准备输入**：`src/pages/prediction/prediction_preparation/preparer.py`
2.  **并行推理**：`src/pages/prediction/flow/run_prediction.py::run_single_prediction` (内部调用 `prediction_execution.executor.PredictionExecutor`)
3.  **结果调整**：`src/pages/prediction/result_modifier/adjustment_pipeline.py`（概率调整 + 文本加成 + `is_new_major`）
4.  **合并去重**：`src/pages/prediction/results_handler.py::combine_and_deduplicate_results`
5.  **输出**：`PredictionResultModel` (定义在 `src/utils/session_manager.py`)
