# 表单组件与校验 API 文档

**路径**: `src/pages/prediction/input_form_components`

本模块涵盖表单校验、GPA/语言分数转换、配置常量与表单状态管理，供页面与服务侧复用/对接。

## 组成模块

- **表单校验器**：`form_validator.py`
- **校验错误类型**：`validation_errors.py`
- **核心工具**：`src/pages/prediction/core/utils.py` (含语言成绩标准化)
- **GPA 转换**：`gpa_converter.py`
- **语言分数换算**：`language_score_converter.py`
- **语言分数校验**：`language_score_validator.py`
- **语言分数处理**：`language_score_processor.py`
- **配置常量**：`form_config.py`
- **表单状态管理**：`form_state.py`
- **目标筛选服务**：`target_options_service.py`
- **UI 组件（拆分）**：
  - 背景信息：`background_ui.py`
  - 申请信息（目标筛选 UI 层）：`target_ui.py`
  - GPA：`gpa_ui.py`
  - 标化成绩（GRE/GMAT，选填）：`standardized_test_ui.py`
  - 语言成绩：`language_ui.py`
  - 其他经历：`experience_ui.py`
  - 提交按钮：`submit_ui.py`
- **UI 组合器**：`form_ui.py`（提供 `FormUIComponents` 聚合渲染入口）
- **UI 辅助工具**：`widget_helpers.py` (v2.3 新增)
- **跨学院提示拦截**：`cross_faculty_guard.py` (v2.5 新增)

---

## 1. 表单校验器 (`form_validator.py`)

**类**: `FormValidator`

### 主要方法

- `validate_standardized_test_score(exam_type, score) -> (bool, str | None, float | None)`
  - 校验标化成绩输入（当前仅支持 `STANDARDIZED_TEST_TYPES=["GRE","GMAT"]`）。
  - **约定**：
    - `score` 为空：返回 `(True, None, None)`（表示选填且未填写）。
    - `score` 非数字：返回 `(False, "<错误信息>", None)`。
    - `score` 非整数（小数）：返回 `(False, "<错误信息>", None)`。
    - 范围校验：基于 `GRE_SCORE_RANGE` (260-340) 与 `GMAT_SCORE_RANGE` (200-800)。
    - 返回的 `parsed_score` 为 `float`，但其值始终为整数。

- `normalize_gpa(raw_gpa, scale_key, background_university=None, gpa_converter: GPAConverter | None=None) -> float | None`
  - 归一化 GPA 到 4.0 制。
  - **逻辑**：
    1. 优先调用 `GPAConverter.convert_gpa_by_rules` 匹配特定院校或国家的转换规则。
    2. 若规则未命中，则按 `GPA_SCALES[scale_key].max` 进行线性缩放（`raw / max * 4.0`）。
    3. 结果保留两位小数，并截断在 `[0.0, 4.0]` 区间。
  - `raw_gpa` 为空或不可解析时返回 `None`。

- `validate_form_data(form_data: dict, gpa_converter: GPAConverter | None=None) -> List[ValidationError]`
  - 全量业务校验，返回 `ValidationError` 列表。
  - **核心校验项**：
    - `background_university`、`background_major_original`、`background_major` 必填且有效。
    - `gpa_raw` 不能为空或 0。
    - `exam_score`（标化成绩）若填写，则调用 `validate_standardized_test_score` 校验。
    - `language_score_input_error`（UI 层标记）若为 `True` 则报错。
    - `language_type` 为雅思时，若分数 > 0，校验 0.5 步长。
    - `language_score_raw` 为 0 且非海外院校时报错（海外院校允许为 0/空）。
    - `research_count` 等经历数量字段非空（0-99）。
    - `experience_details` 与 `count` 的一致性：若数量为 0 但填写了详情，则提示错误。

---

## 2. 核心工具 (`src/pages/prediction/core/utils.py`)

- `normalize_language_score(score, language_type) -> float | Any`
  - 将分数标准化到 [0,1]（托福/120，雅思/9）。异常时原样返回。

- `denormalize_language_score(normalized_score, language_type, round_to_half=False) -> float | Any`
  - 将 [0,1] 区间分数反归一到具体考试分数。`round_to_half=True` 时雅思按 0.5 步长。

---

## 3. GPA 转换 (`gpa_converter.py`)

**类**: `GPAConverter`

### 主要方法

- `__init__(school_base_df)`
  - 接收学校基础表，构建 `school_country_map`（院校→国家映射）。

- `get_university_country(university_name) -> str | None`
  - 返回学校所属国家（从 `school_country_map` 获取）。

- `load_gpa_conversion_rules(config_path: str, file_mtime: float) -> dict | None`
  - 从 `config/gpa_conversion_rules.json` 读取转换规则。使用 `@st.cache_data` 缓存。

- `convert_gpa_by_rules(raw_gpa: float, scale_key: str, background_university: str | None=None, country: str | None=None) -> float | None`
  - 规则匹配逻辑：
    1. 查找“院校规则”（`conversion_rules`），要求 `trigger_scale` 与 `scale_key` 一致。
    2. 查找“国家规则”（`country_rules`），要求 `trigger_scale` 与 `scale_key` 一致。
  - 命中规则后调用 `_apply_conversion_rule`。

- `_apply_conversion_rule(raw_value: float, rule: dict) -> float`
  - 1. 遍历 `ranges`：匹配 `min <= raw_value < max`，返回 `target_gpa` 或按 `target_min/max` 插值。
  - 2. 兜底逻辑：应用 `fallback_multiplier`。支持 `is_percentage`（按百分比缩放）。
  - 结果进行 `[0,4]` 截断并保留两位小数。

### 规则文件关键结构（节选）

```json
{
  "conversion_rules": {
    "某大学": {"trigger_scale": "100", "ranges": [{"min": 85, "max": 90, "target_min": 3.3, "target_max": 3.6}]}
  },
  "country_rules": {
    "中国": {"trigger_scale": "100", "is_percentage": true, "fallback_multiplier": 1.0}
  }
}
```

---

## 3. 语言分数模块

### 3.1 语言分数换算 (`language_score_converter.py`)

**类**: `LanguageScoreConverter`

- `toefl_to_ielts(toefl_score) -> float | None`
  - 将托福分数转换为对应的雅思分数，基于区间映射表。
- `ielts_to_toefl(ielts_score) -> float | None`
  - 将雅思分数转换为对应的托福分数，基于最邻近匹配。

**内置映射表**：
- `TOEFL_TO_IELTS_MAP`：托福分数区间到雅思分数的映射
- `IELTS_TO_TOEFL_MAP`：雅思分数到托福分数的映射

### 3.2 语言分数校验 (`language_score_validator.py`)

**类**: `LanguageScoreValidator`

- `validate_ielts_step(score: float) -> bool`
  - 校验雅思分数是否为 0.5 的倍数（容差 `1e-9`）。

- `validate_score_range(score: float, language_type: str) -> Tuple[bool, Optional[str]]`
  - 校验分数是否在 `LANGUAGE_SCORE_RANGES` 范围内，雅思额外校验步长。

- `validate_and_parse_score(score_text: str, language_type: str) -> Tuple[Optional[float], Optional[str], bool]`
  - 解析分数文本。
  - 返回：`(解析后的分数, 错误消息, 是否有输入错误)`。
  - 输入错误 `has_input_error` 用于 UI 层锁定提交按钮。

### 3.3 语言分数处理 (`language_score_processor.py`)

- `apply_overseas_language_boost(school_name: str, language_type: str) -> float`
  - 根据海外院校等级应用语言成绩加成。
  - 非海外院校返回默认分数（`DEFAULT_LANGUAGE_SCORES`）。
  - 海外院校根据学校等级（`OVERSEAS_SCHOOL_LEVELS`）应用对应倍数（`LANGUAGE_BOOST_MULTIPLIERS`），结果不超过该语言类型的最大值。

---

## 4. 配置常量 (`form_config.py`)

- `GPA_SCALES`：分制上限、步长与格式（如 `{"4.0": {"max": 4.0, "step": 0.1, "format": "%.2f"}}`）
- `DEFAULT_GPA_SCALE`：默认分制（`"4.0"`）
- `LANGUAGE_TYPES`：`["雅思", "托福"]`
- `LANGUAGE_SCORE_RANGES`：分数范围、步长与显示格式
  - 雅思：`{"min": 0.0, "max": 9.0, "step": 0.5, "format": "%.1f"}`
  - 托福：`{"min": 0, "max": 120, "step": 1, "format": "%d"}`
- `DEFAULT_LANGUAGE_SCORES`：默认语言分数（`{"雅思": 6.5, "托福": 90}`）
- **标化成绩（GRE/GMAT）**：
  - `STANDARDIZED_TEST_TYPES`：`["GRE", "GMAT"]`
  - `GRE_SCORE_RANGE`：`{"min": 260, "max": 340, "step": 1, "format": "%d"}`
  - `GMAT_SCORE_RANGE`：`{"min": 200, "max": 800, "step": 10, "format": "%d"}`
  - 加成参数（用于上游 `calculate_gpa_bonus` 计算）：`*_BONUS_THRESHOLD`、`*_SIGMOID_MIDPOINT`、`*_SIGMOID_STEEPNESS`、`*_MAX_BONUS`
- `TARGET_COUNTRY_UNIVERSITY_MAP`：目标国家/地区到院校列表的映射
- `UNIVERSITY_SORT_ORDER`：院校排序顺序（按国家顺序排列）
- `TARGET_COUNTRIES`：目标国家/地区列表
- **告警阈值**：
  - `GPA_WARNING_THRESHOLDS`：各分制的 GPA 低分告警阈值
  - `LANGUAGE_WARNING_THRESHOLDS`：语言成绩低分告警阈值（`{"雅思": 5.5, "托福": 72}`）
- **海外院校相关**：
  - `OVERSEAS_SCHOOL_LEVELS`：海外院校等级列表（`["1-50", "51-100", "101-200", "201-300", "301-500", "500+"]`）
  - `LANGUAGE_BOOST_MULTIPLIERS`：各等级的语言成绩加成倍数

---

## 4.1 校验错误类型 (`validation_errors.py`)

**类**: `ValidationError`（数据类）

- `field: str`：错误字段名
- `message: str`：错误消息（中文）
- `severity: Literal["error", "warning"]`：错误严重程度（默认 `"error"`）

**方法**：
- `__str__() -> str`：返回错误消息
- `to_dict() -> dict`：转换为字典格式

---

## 5. 表单状态管理 (`form_state.py`)

**类**: `FormStateManager`

### 主要方法

- `initialize_session_state(session_manager) -> None`
  - 初始化会话态默认值，关键键包括：
    - `selected_target_countries / universities / majors / major_categories` (空列表)
    - `gpa_scale` (`"4.0"`)、`gpa_raw_input` (`None`)
    - `language_type` (`"雅思"`)、`language_score_input` (`None`)
    - `submitted` (`False`)、`form_data_changed` (`False`)
    - 经历类初始值：`research_count_initial` (0) 等。

- `update_form_snapshot_hash_after_prediction(session_manager)`
  - 预测后更新快照 Hash，防止由于自动保存逻辑导致的重复计算状态。

- `on_form_change(session_manager, change_type: str | None=None) -> None`
  - 通用变更入口，重置提交相关位：
    - `submitted = False`, `form_data_changed = True`, `cross_faculty_confirmed = False` 等。
    - **自动保存节流**：
      - `change_type="text"`：4.0 秒。
      - `change_type="select"` 或默认：1.5 秒。
      - 默认：2.0 秒。

- **针对字段的专用处理**：
  - `on_target_country_change(session_manager)`：同步更新 `selected_target_countries`，并触发 `_handle_target_cascade_filter`（根据新选国家筛选已选院校中不再合法的项）。
  - `on_major_category_change`、`on_target_university_change`、`on_target_major_change`：更新选择并触发变更。
  - `on_language_type_change`：雅思/托福互转（调用 `LanguageScoreConverter`），支持缓存 `lang_conversion_cache`。
  - `gpa_scale_changed`：GPA 分制互转（基于线性缩放），支持缓存 `gpa_conversion_cache`。
  - `on_submit_click(session_manager)`：标记提交、触发自动保存。

### 状态键约定（常见）

- **选择类**：`selected_target_countries`、`selected_target_universities`、`selected_target_majors`、`selected_major_categories`
- **分数类**：`gpa_raw_input`、`gpa_scale`、`language_type`、`language_score_input`
- **标化类**：`standardized_test_type`、`current_exam_score`
- **文本类**：`background_university_selectbox`、`background_major_selectbox`、`experience_details.*`
- **控制类**：`submitted`、`form_data_changed`、`prediction_submit_lock`、`last_submission_logged`、`last_gpa_warning_key`、`last_lang_warning_key`、`last_ielts_step_warning_key`

---

## 6. 最小集成示例

```python
from src.pages.prediction.input_form_components.form_validator import FormValidator
from src.pages.prediction.input_form_components.gpa_converter import GPAConverter
from src.pages.prediction.input_form_components.validation_errors import ValidationError

def validate_and_prepare(form_data, school_base_df):
    gpa_conv = GPAConverter(school_base_df)
    errors = FormValidator.validate_form_data(form_data, gpa_converter=gpa_conv)
    if errors:
        error_messages = [str(err) for err in errors]  # 或使用 err.message
        return None, error_messages

    gpa_4 = FormValidator.normalize_gpa(
        form_data['gpa_raw'], form_data['gpa_scale'], form_data.get('background_university'), gpa_conv
    )
    lang_norm = FormValidator.normalize_language_score(form_data.get('language_score_raw'), form_data.get('language_type'))
    return {"gpa_4": gpa_4, "language_norm": lang_norm}, []
```

---

## 7. UI 组件 (background / target / gpa / language / experience / submit)

- **背景信息** (`background_ui.render_background_section`)
  - 返回：`(background_university, background_major_original, background_major)`
  - **v2.3 重构**：使用 `widget_helpers.SelectBoxHelper` 渲染，将选项生成逻辑与 UI 渲染解耦。
  - 背景院校候选来自 `cases_df` 与 `load_school_base_data()` 的并集，并按案例频次降序排序。
  - 背景专业选项来自 `cases_df`，显示原始专业名（`background_major_original`），内部映射为标准专业名（`background_major`）。
  - 背景院校变更时会清空语言成绩输入（海外院校语言成绩为选填）。

- **申请信息（目标筛选 UI 层）** (`target_ui.render_target_section`)
  - 返回：`(prediction_universities, prediction_majors, all_unis, all_majors)`
  - **v2.3 重构**：使用 `widget_helpers.SelectBoxHelper` 渲染多选框，简化 UI 代码。
  - **v2.3 优化**：**支持仅选择专业**——当用户未选择学校，但选择了国家/专业大类/专业时，`prediction_universities` 会被自动扩展为所有符合条件的院校列表。
  - 核心筛选与映射调用 `target_options_service`。

- **GPA** (`gpa_ui.render_gpa_section`)
  - 返回：`gpa_raw`
  - 说明：分制切换、输入与低分警示；键：`gpa_scale_widget_key`、`gpa_raw_input_widget`。

- **标化成绩** (`standardized_test_ui.render_standardized_test_section`)
  - 返回：`(exam_type, exam_score)`
  - 说明：
    - 使用 `st.segmented_control` 选择 `GRE/GMAT`（key：`standardized_test_type_widget`）。
    - 分数为文本输入（每种考试类型分别使用 key：`standardized_test_score_text_{exam_type}`）。
    - 解析成功后写入 `session_manager.set(current_exam_score=parsed_score)`；解析失败会 toast 提示并将成绩置为 `None`。

- **语言成绩** (`language_ui.render_language_section`)
  - 返回：`(language_type, language_score_raw)`
  - **逻辑说明**：
    - 类型切换（雅思/托福）：使用 `st.segmented_control`。
    - **海外院校背景**：
      - `st.text_input` 输入。
      - 标签显示“（选填）”。
      - 支持为空，调用 `LanguageScoreValidator.validate_and_parse_score` 解析。
    - **非海外院校背景**：
      - `st.number_input` 数字输入。
      - 若无历史值，默认填入 `DEFAULT_LANGUAGE_SCORES`。
    - **告警**：分数低于 `LANGUAGE_WARNING_THRESHOLDS` 或雅思非 0.5 步长时，显示 `st.toast` 告警。

- **其他经历** (`experience_ui.render_experience_section`)
  - 返回：`(research_count, award_count, internship_count, paper_count, experience_details)`
  - 说明：四类经历（科研/获奖/实习/论文）的数量和详情输入。

- **提交按钮** (`submit_ui.render_submit_button`)
  - 返回：`bool`（是否点击）
  - 说明：禁用提示与提交态控制。

---

## 8. 目标筛选服务 (`target_options_service.py`)

- `build_target_base_df(cases_df, details_df) -> tuple[pd.DataFrame, Dict[str, str]]`
  - 合并案例表与详情表，生成用于联动筛选的 `base_df`。
  - 返回：`(base_df, university_country_map)`。

- `compute_selection_cache_key(...) -> tuple`
  - 生成用于选项缓存的键（由已选国家、院校、大类、专业的有序元组构成）。

- `compute_options(base_df, countries, universities, categories, majors) -> tuple`
  - 计算四级联动后的可选列表。
  - 院校排序基于 `UNIVERSITY_SORT_ORDER`。

- `expand_aggregated_majors_for_prediction(...) -> List[str]`
  - 将 UI 选中的“聚合专业名”映射回模型所需的“原始专业名”列表。

### 行为约定（重要）

- UI 多选使用聚合名（`target_major_agg`）；传入后续流程的字段仍是 `target_majors`，内容为“原始专业名列表”。
- 未选择聚合专业时，`target_majors` 应为空。

### 自动扩展逻辑

- 当未选择院校，但选择了国家/专业大类/聚合专业之一时：`prediction_universities` 自动扩展为当前筛选条件下的所有可选院校。
- 当未选择聚合专业，但已选择国家/院校/专业大类之一时：按筛选条件给出聚合专业候选。
- 选项生成与缓存：基于 `compute_selection_cache_key` 复用 `target_options_cache`。

---

## 9. UI 辅助工具 (`widget_helpers.py`) (v2.3 新增)

**类**: `SelectBoxHelper(session_manager, form_state_manager, logger)`

封装单选框和多选框的通用渲染逻辑，包括选项生成、缓存、历史默认值恢复。

- `render_cached_selectbox(...) -> Any`
  - 封装**单选框**通用逻辑。
  - 自动从 `user_history_data` 恢复历史默认值。

- `render_multiselect(...) -> list`
  - 封装**多选框**的通用渲染逻辑。
  - `placeholder` 默认为 `"不填默认全选"`。

### 最小示例

```python
from functools import partial
from src.pages.prediction.input_form_components.widget_helpers import SelectBoxHelper

# 假设已有 session_manager, form_state_manager, logger
helper = SelectBoxHelper(session_manager, form_state_manager, logger)

# 单选院校
university = helper.render_cached_selectbox(
    label="背景院校",
    widget_key="background_university_selectbox",
    cache_key="background_universities_cache",
    history_key="background_university",
    options_generator_func=lambda: sorted(cases_df["background_university"].dropna().astype(str).unique().tolist()),
    on_change_callback=partial(form_state_manager.on_form_change, session_manager, change_type="select"),
)
```

---

## 10. UI 组合器 (`form_ui.py`)

**类**: `FormUIComponents(session_manager: SessionManager)`

- **作用**：聚合并渲染背景/目标/GPA/标化/语言/经历/提交七大区块，封装日志器、状态写入与提示。
- **说明**：内部使用上述分拆组件方法；日志器来源于 `utils.logger.setup_logger("page3", "prediction")`。

---

## 11. 最小页面渲染顺序示例（整合）

```python
import streamlit as st
from src.utils.session_manager import SessionManager
from src.pages.prediction.input_form_components.form_ui import FormUIComponents

sm = SessionManager()
ui = FormUIComponents(sm)

# 背景信息
bg_university, bg_major_original, bg_major = ui.render_background_section(cases_df)

# GPA / 标化 / 语言
_ = ui.render_gpa_section()
exam_type, exam_score = ui.render_standardized_test_section()
lang_type, lang_score = ui.render_language_section()

# 申请信息（目标筛选）
pred_unis, pred_majors, all_unis, all_majors = ui.render_target_section(cases_df)

# 其他经历
research, award, internship, paper, exp_details = ui.render_experience_section()

# 提交按钮
if ui.render_submit_button(disabled_status=False):
    st.toast("已提交，正在计算…")
```

---

## 12. 跨学院提示 (`cross_faculty_guard.py`)

- **作用**：当用户所选目标专业可能跨出其背景专业所属学院时，弹出确认对话框以二次确认，避免误触发跨学院预测。
- **依赖数据**：
  - `cases_df` 中背景专业到学院的映射（列 `background_major`、`faculty`）。
  - 专业详情表中列：`学校`、`专业英文名称`、`专业英文名称_聚合`（可选）、`专业大类`。

### 主要方法

- `check_cross_faculty_situation(...)`
  - 精确匹配模式（基于“院校+原始专业”）。

- `quick_cross_faculty_check(background_major, selected_categories, selected_majors, cases_df=None) -> (has_cross, background_faculty, target_faculties, agent_approved)`
  - 快速检查模式（基于“背景专业 + 已选专业大类/聚合专业”）。
  - `agent_approved`：智能确认标志位（当前实现中默认为 `False`）。

- `cross_faculty_confirm_dialog(session_manager, background_faculty, target_faculties) -> None`
  - 弹出 `st.dialog` 二次确认框。

### 最小接入示例

```python
import streamlit as st
from src.pages.prediction.input_form_components.cross_faculty_guard import (
    quick_cross_faculty_check,
    cross_faculty_confirm_dialog,
)

# 假设已有 session_manager, cases_df
has_cross, bg_faculty, target_faculties, agent_approved = quick_cross_faculty_check(
    background_major=session_manager.get("background_major"),
    selected_categories=session_manager.get("selected_major_categories"),
    selected_majors=session_manager.get("selected_target_majors"),
    cases_df=cases_df,
)

if has_cross and not agent_approved and not session_manager.get("cross_faculty_confirmed"):
    cross_faculty_confirm_dialog(session_manager, bg_faculty, target_faculties)
    st.stop()  # 等待用户确认；确认后会 rerun
```

---

## 13. 其他说明

### 海外院校语言成绩处理
- 系统通过 `school_level_service.is_overseas_school()` 判断背景院校是否为海外院校。
- 海外院校的语言成绩为选填项，使用文本输入框，允许为空或为 0。
- 非海外院校的语言成绩为必填项，使用数字输入框，默认值为 `DEFAULT_LANGUAGE_SCORES`。
- 海外院校的语言成绩会根据学校等级应用加成（`apply_overseas_language_boost`）。

### 表单自动保存机制
- 表单变更时更新快照 hash（带节流）：
  - 文本输入：4 秒节流
  - 选择操作：1.5 秒节流
  - 默认：2 秒节流
- 使用快照哈希去重，避免频繁重复更新。
- 仅在会话态里维护，不做落盘/数据库持久化。

### 目标选择自动扩展逻辑
- 当用户未选择院校，但选择了国家/专业大类/专业之一时，`prediction_universities` 自动扩展。
- 当用户未选择聚合专业，但已选择国家/院校/专业大类之一时，按筛选条件给出聚合专业候选。

---

> **维护人**: lijiapeng8@xdf.cn
> **版本**: v2.5
