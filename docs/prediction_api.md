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
- **结果自动注入**：`is_new_major: bool` (所有预测路径均会通过 `adjustment_pipeline` 注入)
- **学部标记**：`faculty: str` (由 `processor.py` 的 metadata 附加阶段注入)

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

### 3.3 关于 `unified_results` 的合并策略

`unified_results` 会合并以下来源的结果，并根据优先级进行去重（院校+专业）：
1.  **优先级 3**：用户指定结果 (`user_specified_results`) —— 最高优先级。
2.  **优先级 2**：跨专业推荐 (`cross_major_results`) —— 仅当样本标记为 `admitted == 1` 时参与合并。
3.  **优先级 1**：相似专业推荐 (`similarity_results`) —— 基础推荐。

当同一 (院校, 专业) 组合在多个来源中出现时，高优先级来源将覆盖低优先级来源；若优先级相同，则录取概率（`probability`）更高者获胜。

> **注意**：跨专业推荐生成器（`result_modifier/filters.py`）现在会自动为符合历史录取的推荐项标记 `admitted: 1`，因此它们会出现在 `unified_results` 中。

## 4. Streamlit 页面预测管线 (Flow Control)

**入口**：`src/pages/prediction/flow/pipeline.py::run_prediction_pipeline_with_progress`

页面管线负责串联资源加载、并行推理、推荐生成与后处理：

1.  **风险预警**: 通过 `cross_faculty_guard.py` 识别潜在的跨学院申请风险。
2.  **准备输入 (Preparation)**：`src/pages/prediction/prediction_preparation/preparer.py`（含 `form_normalizer.py` 归一化）。
3.  **核心推理 (Recall & Execution)**：
    - 在 `flow/run_prediction.py::run_single_prediction` 中进行 **混合召回 (Recall)**（E5 语义 + Fuzz 字符匹配）。
    - 之后调用 `prediction_execution.executor.PredictionExecutor` 进行 **精排推理**。
4.  **结果平衡与初筛 (Processing)**：
    - 在 `flow/processor.py` 中通过 `SingleResultProcessor` 完成元数据注入、**目标特定语言惩罚**、相似度偏置修正。
    - 利用 `BoundaryCaseAgent` 对相似推荐与跨专业推荐进行数量平衡。
5.  **批量修正 (Modification)**：通过 `src/pages/prediction/result_modifier/adjustment_pipeline.py` 进行 GPA/语言、跨专业、跨学部及文本加成的统一修正。
6.  **结果交付**：`src/pages/prediction/results_handler.py::combine_and_deduplicate_results`。

### 关于 `unified_results` 的合并优先级
合并过程基于优先级覆盖逻辑：
- **优先级 3**：用户指定结果 (`user_specified`) —— 最高优先级。
- **优先级 2**：跨专业推荐 (`cross_major`) —— 仅当样本标记为 `admitted == 1` 时参与合并。
- **优先级 1**：相似专业推荐 (`similarity`) —— 基础推荐。

当 (院校, 专业) 组合在多个来源中出现时，高优先级来源将覆盖低优先级来源；若优先级相同，则录取概率（`probability`）更高者获胜。
