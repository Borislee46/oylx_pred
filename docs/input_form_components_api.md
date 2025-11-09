## 表单组件与校验 API 文档（src/pages/prediction/input_form_components）

本模块涵盖表单校验、GPA/语言分数转换、配置常量与表单状态管理，供页面与服务侧复用/对接。

### 组成模块
- 表单校验器：`form_validator.py`
- 校验错误类型：`validation_errors.py`
- GPA 转换：`gpa_converter.py`
- 语言分数换算：`language_score_converter.py`
- 语言分数校验：`language_score_validator.py`
- 语言分数处理：`language_score_processor.py`
- 配置常量：`form_config.py`
- 表单状态管理：`form_state.py`
- 目标筛选服务：`target_options_service.py`
- UI 组件（拆分）：
  - 背景信息：`background_ui.py`
  - 申请信息（目标筛选 UI 层）：`target_ui.py`
  - GPA：`gpa_ui.py`
  - 语言成绩：`language_ui.py`
  - 其他经历：`experience_ui.py`
  - 提交按钮：`submit_ui.py`
- **UI 组合器**：`form_ui.py`（提供 `FormUIComponents` 聚合渲染入口）
- **UI 辅助工具**：`widget_helpers.py` (v2.3 新增)
- **跨学院提示拦截**：`cross_faculty_guard.py` (v2.5 新增)

---

### 一、表单校验器（form_validator.py）
类：`FormValidator`

- `normalize_language_score(score, language_type) -> float | Any`
  - 将分数标准化到 [0,1]（托福/雅思）。异常时原样返回。

- `denormalize_language_score(normalized_score, language_type, round_to_half=False) -> float | Any`
  - 将[0,1]区间分数反归一到具体考试分数。`round_to_half=True` 时雅思按 0.5 步长。

- `normalize_gpa(raw_gpa, scale_key, background_university=None, gpa_converter: GPAConverter | None=None) -> float | None`
  - 优先按 `GPAConverter.convert_gpa_by_rules` 的学校/国家规则换算，否则按 `GPA_SCALES[scale_key].max` 线性缩放到 4.0 制并保留两位小数。
  - `raw_gpa` 为空或不可解析、`scale_key` 缺失时返回 `None`。

- `validate_form_data(form_data: dict, gpa_converter: GPAConverter | None=None) -> List[ValidationError]`
  - 业务校验，返回 `ValidationError` 对象列表，包含：
    - 背景院校/专业必填，且映射有效
    - GPA 不能为空/不为 0，分制有效
    - 语言分数校验（通过 `LanguageScoreValidator`）：雅思分数必须是 0.5 的倍数；非海外院校语言分数不为 0
    - 经历数量字段非空，经历详情与数量一致性检查

---

### 二、GPA 转换（gpa_converter.py）
类：`GPAConverter`

- `__init__(school_base_df)`
  - 接收学校基础表（含列 `学校名称`、`国家`）以构建院校→国家映射。

- `get_university_country(university_name) -> str | None`
  - 返回学校所属国家，内置简单缓存。

- `load_gpa_conversion_rules(config_path: str, file_mtime: float) -> dict | None`（缓存）
  - 从 `config/gpa_conversion_rules.json` 读取转换规则，进行结构校验。

- `convert_gpa_by_rules(raw_gpa: float, scale_key: str, background_university: str | None=None, country: str | None=None) -> float | None`
  - 优先匹配“院校规则”（`conversion_rules`），其次“国家规则”（`country_rules`）；规则命中 `trigger_scale` 时应用 `_apply_conversion_rule`。

- `_apply_conversion_rule(raw_value: float, rule: dict) -> float`
  - 支持区间 `ranges[{min,max,target_gpa | target_min,target_max}]` 与兜底公式（比例/百分制），并进行 [0,4] 截断与四舍五入。

规则文件关键结构（节选）：
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

### 三、语言分数换算（language_score_converter.py）
类：`LanguageScoreConverter`

- `toefl_to_ielts(toefl_score) -> float | None`
  - 将托福分数转换为对应的雅思分数，基于区间映射表。
- `ielts_to_toefl(ielts_score) -> float | None`
  - 将雅思分数转换为对应的托福分数，基于最邻近匹配。

内置映射表：
- `TOEFL_TO_IELTS_MAP`：托福分数区间到雅思分数的映射
- `IELTS_TO_TOEFL_MAP`：雅思分数到托福分数的映射

---

### 三-1、语言分数校验（language_score_validator.py）
类：`LanguageScoreValidator`

- `validate_ielts_step(score: float) -> bool`
  - 校验雅思分数是否为 0.5 的倍数（使用浮点数精度容差 `FLOAT_EPSILON = 1e-9`）。

- `validate_score_range(score: float, language_type: str) -> Tuple[bool, Optional[str]]`
  - 校验语言分数是否在有效范围内，并检查雅思步长。
  - 返回：`(是否有效, 错误消息)`。

- `validate_and_parse_score(score_text: str, language_type: str) -> Tuple[Optional[float], Optional[str], bool]`
  - 解析并校验语言分数文本输入。
  - 返回：`(解析后的分数, 错误消息, 是否有输入错误)`。
  - 空输入返回 `(None, None, False)`。

---

### 三-2、语言分数处理（language_score_processor.py）

- `apply_overseas_language_boost(school_name: str, language_type: str) -> float`
  - 根据海外院校等级应用语言成绩加成。
  - 非海外院校返回默认分数（`DEFAULT_LANGUAGE_SCORES`）。
  - 海外院校根据学校等级（`OVERSEAS_SCHOOL_LEVELS`）应用对应倍数（`LANGUAGE_BOOST_MULTIPLIERS`），结果不超过该语言类型的最大值。

---

### 四、配置常量（form_config.py）
- `GPA_SCALES`：分制上限、步长与格式（如 `{"4.0": {"max": 4.0, "step": 0.1, "format": "%.2f"}}`）
- `DEFAULT_GPA_SCALE`：默认分制（`"4.0"`）
- `LANGUAGE_TYPES`：`["雅思", "托福"]`
- `LANGUAGE_SCORE_RANGES`：分数范围、步长与显示格式
  - 雅思：`{"min": 0.0, "max": 9.0, "step": 0.5, "format": "%.1f"}`
  - 托福：`{"min": 0, "max": 120, "step": 1, "format": "%d"}`
- `DEFAULT_LANGUAGE_SCORES`：默认语言分数（`{"雅思": 6.5, "托福": 90}`）
- `TARGET_COUNTRY_UNIVERSITY_MAP`：目标国家/地区到院校列表的映射
- `UNIVERSITY_SORT_ORDER`：院校排序顺序（按国家顺序排列）
- `TARGET_COUNTRIES`：目标国家/地区列表
- 告警阈值：
  - `GPA_WARNING_THRESHOLDS`：各分制的 GPA 低分告警阈值
  - `LANGUAGE_WARNING_THRESHOLDS`：语言成绩低分告警阈值（`{"雅思": 5.5, "托福": 65}`）
- 海外院校相关：
  - `OVERSEAS_SCHOOL_LEVELS`：海外院校等级列表（`["1-50", "51-100", "101-200", "201-300", "301-500", "500+"]`）
  - `LANGUAGE_BOOST_MULTIPLIERS`：各等级的语言成绩加成倍数

---

### 四-1、校验错误类型（validation_errors.py）
类：`ValidationError`（数据类）

- `field: str`：错误字段名
- `message: str`：错误消息（中文）
- `severity: Literal["error", "warning"]`：错误严重程度（默认 `"error"`）

方法：
- `__str__() -> str`：返回错误消息
- `to_dict() -> dict`：转换为字典格式

---

### 五、表单状态管理（form_state.py）
类：`FormStateManager`

- `initialize_session_state(session_manager) -> None`
  - 初始化会话态（尝试从用户历史记录恢复），并进行提示；默认键包括：
    - `selected_target_*`、`gpa_scale`、`language_type`、`*_input`、`submitted`、`form_data_changed`、`prediction_submit_lock` 等。

- `save_current_form_data(session_manager, form_data: dict) -> bool`
  - 将当前表单快照持久化（用户级）。

- `on_form_change(session_manager, change_type: str | None=None) -> None`
  - 通用变更入口：重置提交相关标志；按变更类型节流自动保存（文本 4s，其它 1.5s）。

- 针对字段的专用处理：
  - `on_target_country_change(session_manager)`：同步目标国家→筛选院校，并触发变更。
  - `on_major_category_change(session_manager)`、`on_target_university_change(session_manager)`、`on_target_major_change(session_manager)`：更新选择并触发变更。
  - `on_language_type_change(session_manager)`：互转雅思/托福分数（带缓存与回填）。
  - `gpa_scale_changed(session_manager)`：互转 GPA 分制（带缓存与回填）。
  - `on_submit_click(session_manager)`：标记提交、触发自动保存。

私有方法（内部使用）：
- `_clear_widget_state(widget_key: str)`：清除指定 widget 的状态。
- `_snapshot_hash(snapshot: dict) -> str`：计算表单快照的哈希值（用于去重保存）。
- `_auto_save_form_data(session_manager, throttle_seconds: float)`：自动保存表单数据（带节流）。
- `_on_target_selection_change(session_manager, session_state_key, widget_key, log_message_template)`：目标选择变更的通用处理（包括国家变更时的院校自动筛选）。

状态键约定（常见）：
- 选择类：`selected_target_countries`、`selected_target_universities`、`selected_target_majors`、`selected_major_categories`
- 分数类：`gpa_raw_input`、`gpa_scale`、`language_type`、`language_score_input`
- 文本类：`background_university_selectbox`、`background_major_selectbox`、`experience_details.*`
- 控制类：`submitted`、`form_data_changed`、`prediction_submit_lock`、`restore_notice_shown`、`last_gpa_warning_key`、`last_lang_warning_key`、`last_ielts_step_warning_key`

缓存与会话键（与实现一致）：
- `target_section_cache`: {`base_df`, `university_country_map`}（首次进入目标区块时加载并缓存）。
- `target_options_cache`: {selection_key → {`country`, `university`, `category`, `major`}}（基于四级筛选键的选项缓存）。
- `background_universities_cache`: 背景院校候选集（并集来源）。
- `background_majors_cache`: {`majors_display`, `major_map`}（原始→标准专业名映射与显示列表）。
- `lang_conversion_cache`: 语言分数互转缓存（雅思↔托福）。
- `gpa_conversion_cache`: GPA 分制互转缓存（旧/新分制+当前值作为键）。

---

### 六、最小集成示例
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

### 七、UI 组件（background/target/gpa/language/experience/submit）

- 背景信息（`background_ui.render_background_section(session_manager, form_state_manager, cases_df, logger)`）
  - 返回：`(background_university, background_major_original, background_major)`
  - 说明：
    - **v2.3 重构** - 使用 `widget_helpers.SelectBoxHelper` 渲染，将选项生成逻辑与 UI 渲染解耦。
    - 背景院校候选来自 `cases_df` 与 `load_school_base_data()` 的并集，并按案例频次降序排序。
    - 背景专业选项来自 `cases_df`，显示原始专业名（`background_major_original`），内部映射为标准专业名（`background_major`）。
    - 背景院校变更时会清空语言成绩输入（海外院校语言成绩为选填）。

- 申请信息（目标筛选 UI 层，`target_ui.render_target_section(session_manager, form_state_manager, cases_df, logger)`）
  - 返回：`(prediction_universities, prediction_majors, all_unis, all_majors)`
  - 说明：
    - **v2.3 重构** - 使用 `widget_helpers.SelectBoxHelper` 渲染多选框，简化 UI 代码。
    - **v2.3 优化** - **支持仅选择专业**：当用户未选择学校，但选择了国家/专业大类/专业时，`prediction_universities` 会被自动扩展为所有符合条件的院校列表。
    - 核心筛选与映射调用 `target_options_service`。

- GPA（`gpa_ui.render_gpa_section(session_manager, form_state_manager, logger)`）
  - 返回：`gpa_raw`
  - 说明：分制切换、输入与低分警示；键：`gpa_scale_widget_key`、`gpa_raw_input_widget`。

- 语言成绩（`language_ui.render_language_section(session_manager, form_state_manager, logger)`）
  - 返回：`(language_type, language_score_raw)`
  - 说明：
    - 类型切换、分数输入、阈值与步长校验；键：`language_type_widget_key`、`language_score_input_widget`。
    - **海外院校处理**：根据背景院校是否为海外院校（通过 `school_level_service` 判断），采用不同的输入方式：
      - 海外院校：使用文本输入（`text_input`），允许为空，标签显示"（选填）"。
      - 非海外院校：使用数字输入（`number_input`），必填，当无历史值时默认分数：雅思 6.5、托福 86。
    - 雅思强制 0.5 步长校验（通过 `LanguageScoreValidator.validate_ielts_step`）并提示。
    - 低分告警：当分数低于 `LANGUAGE_WARNING_THRESHOLDS` 时显示 toast 提示。

- 其他经历（`experience_ui.render_experience_section(session_manager, form_state_manager, logger)`）
  - 返回：`(research_count, award_count, internship_count, paper_count, experience_details)`
  - 说明：键：`*_count_input`、`*_details_input`（四类：科研/获奖/实习/论文）。

- 提交按钮（`submit_ui.render_submit_button(session_manager, form_state_manager, disabled_status=False)`）
  - 返回：`bool`（是否点击）
  - 说明：禁用提示与提交态控制；键：`submit_button_key`。

---

### 十、UI 组合器（form_ui.py）

- `class FormUIComponents(session_manager: SessionManager)`
  - 作用：聚合并渲染背景/目标/GPA/语言/经历/提交六大区块，封装日志器、状态写入与提示。
  - 说明：内部使用上述分拆组件方法；日志器来源于 `utils.logger.setup_logger("page3", "prediction")`。
  - 属性：
    - `session_manager: SessionManager`：会话管理器实例
    - `form_state_manager: FormStateManager`：表单状态管理器实例
  - 方法：
    - `render_background_section(cases_df)`：渲染背景信息区块
    - `render_gpa_section()`：渲染 GPA 区块
    - `render_target_section(cases_df)`：渲染申请信息（目标筛选）区块
    - `render_language_section()`：渲染语言成绩区块
    - `render_experience_section()`：渲染其他经历区块
    - `render_submit_button(disabled_status=False)`：渲染提交按钮

---

### 八、目标筛选服务（target_options_service.py）

- `build_target_base_df(cases_df: pd.DataFrame | None, details_df: pd.DataFrame | None) -> tuple[pd.DataFrame, Dict[str, str]]`
  - 合并 `cases_df[['target_university','target_major']]` 与 `details_df`，生成列：`target_university`、`target_major`、`major_category`、`target_major_agg`、`country`。
  - 兼容：`details_df` 无 `专业英文名称_聚合` 时回退为 `专业英文名称`。
  - 返回：`(base_df, university_country_map)`，其中 `university_country_map` 为院校到国家的映射字典。

- `compute_selection_cache_key(selected_countries: Set[str], selected_universities: Set[str], selected_categories: Set[str], selected_majors: Set[str]) -> str`
  - 用于 `target_options_cache` 的稳定键（SHA256 哈希），保持与旧实现一致。

- `compute_options(base_df: pd.DataFrame, selected_countries: Set[str], selected_universities: Set[str], selected_categories: Set[str], selected_majors: Set[str]) -> tuple[List[str], List[str], List[str], List[str]]`
  - 返回四级可选项列表：`(国家列表, 院校列表, 专业大类列表, 聚合专业列表)`。
  - 维持院校排序（`UNIVERSITY_SORT_ORDER`）。
  - 选项计算基于已选条件的反向筛选（排除已选条件后计算剩余可选项）。

- `expand_aggregated_majors_for_prediction(base_df: pd.DataFrame, selected_countries: Set[str], selected_universities: Set[str], selected_categories: Set[str], aggregated_to_use: List[str]) -> List[str]`
  - 将选择的聚合专业展开为原始 `target_major` 列表。
  - 当未选择聚合专业时返回空列表。
  - 展开时会根据已选的国家、院校、专业大类进行筛选。

行为约定（重要）：
- UI 多选使用聚合名（`target_major_agg`）；传入后续流程的字段仍是 `target_majors`，内容为“原始专业名列表”。
- 未选择聚合专业时，`target_majors` 应为空，以免触发表单规则“如果先勾选目标专业，请选择目标专业对应的学校”。

自动扩展逻辑（与 `target_ui.render_target_section` 一致）：
- 当未选择院校，但选择了国家/专业大类/聚合专业之一时：`prediction_universities` 自动扩展为当前筛选条件下的所有可选院校。
- 当未选择聚合专业，但已选择国家/院校/专业大类之一时：按筛选条件给出聚合专业候选；最终传递到后续流程前，会调用 `expand_aggregated_majors_for_prediction` 将聚合专业展开为原始 `target_major` 列表。
- 选项生成与缓存：会基于 `compute_selection_cache_key` 将四级选择序列化为键；若命中 `target_options_cache` 则直接复用，未命中则实时计算并写入缓存。

---

### 九、UI 辅助工具 (widget_helpers.py) (v2.3 新增)
- `class SelectBoxHelper(session_manager, form_state_manager, logger)`
  - 封装单选框和多选框的通用渲染逻辑，包括选项生成、缓存、历史默认值恢复。
  
  - `render_cached_selectbox(label: str, widget_key: str, cache_key: str, history_key: str, options_generator_func: Callable[[], Any], on_change_callback: Callable, options_path_in_cache: str = None) -> Any`
    - 封装**单选框**通用逻辑。
    - `options_generator_func`：选项生成函数（首次调用时执行并缓存）。
    - `options_path_in_cache`：当生成结果为字典时，指定选项列表所在的键（如 `"majors_display"`）。
    - 自动从 `user_history_data` 恢复历史默认值。
    - 异常处理：渲染失败时返回空选项列表的 selectbox。
  
  - `render_multiselect(label: str, options: list, default_selections: list, widget_key: str, on_change_callback: Callable) -> list`
    - 封装**多选框**的通用渲染逻辑。
    - `placeholder` 默认为 `"不填默认全选"`。

最小示例：

```python
from functools import partial
from src.pages.prediction.input_form_components.widget_helpers import SelectBoxHelper

# 假设已有 session_manager, form_state_manager, logger
helper = SelectBoxHelper(session_manager, form_state_manager, logger)

# 单选院校（选项通过生成函数+缓存）
university = helper.render_cached_selectbox(
    label="背景院校",
    widget_key="background_university_selectbox",
    cache_key="background_universities_cache",
    history_key="background_university",
    options_generator_func=lambda: sorted(cases_df["background_university"].dropna().astype(str).unique().tolist()),
    on_change_callback=partial(form_state_manager.on_form_change, session_manager, change_type="select"),
)

# 多选目标专业（聚合名）
selected_agg_majors = helper.render_multiselect(
    label="目标专业（可多选）",
    options=sorted(base_df["target_major_agg"].dropna().astype(str).unique().tolist()),
    default_selections=session_manager.get("selected_target_majors", []),
    widget_key="target_majors_multiselect",
    on_change_callback=partial(form_state_manager.on_target_major_change, session_manager),
)
```

### 十一、最小页面渲染顺序示例（整合）
```python
import streamlit as st
from src.utils.session_manager import SessionManager
from src.pages.prediction.input_form_components.form_ui import FormUIComponents

sm = SessionManager()
ui = FormUIComponents(sm)

# 背景信息
bg_university, bg_major_original, bg_major = ui.render_background_section(cases_df)

# GPA / 语言
_ = ui.render_gpa_section()
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

### 十二、跨学院提示（cross_faculty_guard.py）

- 作用：当用户所选目标专业可能跨出其背景专业所属学院时，弹出确认对话框以二次确认，避免误触发跨学院预测。
- 依赖数据：
  - `cases_df` 中背景专业到学院的映射（要求存在列 `background_major`、`faculty`）。
  - 专业详情表（`load_school_major_details_df()`）中列：`学校`、`专业英文名称`、`专业英文名称_聚合`（可选）、`专业大类`。

- 提供方法：
  - `check_cross_faculty_situation(background_major, target_majors, target_universities, cases_df) -> (has_cross: bool, background_faculty: str | None, target_faculties: set[str])`
    - 基于“院校+原始专业”的精确匹配，判断所选目标是否跨学院，适合在已明确“目标院校+原始专业列表”后进行最终校验。
  - `quick_cross_faculty_check(background_major, selected_categories, selected_majors, cases_df=None, details_df=None) -> (has_cross: bool, background_faculty: str | None, target_faculties: set[str])`
    - 轻量快速检查：仅依赖“背景专业 + 已选专业大类/聚合专业”，适合提交前进行早期拦截与提示。
  - `cross_faculty_confirm_dialog(session_manager, background_faculty, target_faculties) -> None`
    - 使用 `st.dialog` 弹出确认框。文案："您明确选择的目标专业包含跨学院方向，是否继续？"；确认/取消分别写入会话态键并触发 `st.rerun()`。

- 会话态键（由对话框维护）：
  - `cross_faculty_confirmed: bool`：已确认跨学院继续。
  - `cross_faculty_cancelled: bool`：取消继续，清理提交态。
  - `pending_cross_faculty_prediction: bool`：等待跨学院预测的继续流程。
  - `pending_prediction_data`：可由上游在弹框前临时保存提交上下文（可选）。
  - `prediction_submit_lock: bool`、`submitted: bool`：提交相关控制位（取消时会复位）。

- 使用建议（典型接入点）：
  - 提交按钮被点击后、真正执行预测前：优先调用 `quick_cross_faculty_check(...)`；若跨学院且未确认，则弹出 `cross_faculty_confirm_dialog(...)` 并 `st.stop()`。
  - 若上游已经将聚合专业展开为原始 `target_major`，且目标院校已确定，可改用 `check_cross_faculty_situation(...)` 做更精确判断。

- 最小接入示例：
```python
import streamlit as st
from src.pages.prediction.input_form_components.cross_faculty_guard import (
    quick_cross_faculty_check,
    cross_faculty_confirm_dialog,
)

# 假设已有 session_manager, cases_df
has_cross, bg_faculty, target_faculties = quick_cross_faculty_check(
    background_major=session_manager.get("background_major"),
    selected_categories=session_manager.get("selected_major_categories"),
    selected_majors=session_manager.get("selected_target_majors"),  # 聚合或原始名均可
    cases_df=cases_df,
)

if has_cross and not session_manager.get("cross_faculty_confirmed"):
    cross_faculty_confirm_dialog(session_manager, bg_faculty, target_faculties)
    st.stop()  # 等待用户确认；确认后会 rerun
```

---

### 十三、其他说明

#### 海外院校语言成绩处理
- 系统通过 `school_level_service.is_overseas_school()` 判断背景院校是否为海外院校。
- 海外院校的语言成绩为选填项，使用文本输入框，允许为空或为 0。
- 非海外院校的语言成绩为必填项，使用数字输入框，默认值为 `DEFAULT_LANGUAGE_SCORES`。
- 海外院校的语言成绩会根据学校等级应用加成（`apply_overseas_language_boost`），但仅在特定场景下使用。

#### 表单自动保存机制
- 表单变更时自动保存（带节流）：
  - 文本输入：4 秒节流（`AUTO_SAVE_THROTTLE_TEXT`）
  - 选择操作：1.5 秒节流（`AUTO_SAVE_THROTTLE_SELECT`）
  - 默认：2 秒节流（`AUTO_SAVE_THROTTLE_DEFAULT`）
- 使用快照哈希去重，避免重复保存相同内容。
- 保存到用户级存储（`UserFormStorage`）。

#### 目标选择自动扩展逻辑
- 当用户未选择院校，但选择了国家/专业大类/专业之一时，`prediction_universities` 自动扩展为所有符合条件的院校列表。
- 当用户未选择聚合专业，但已选择国家/院校/专业大类之一时，按筛选条件给出聚合专业候选。
- 最终传递到后续流程前，会调用 `expand_aggregated_majors_for_prediction` 将聚合专业展开为原始 `target_major` 列表。

#### 国家变更时的院校自动筛选
- 当目标国家变更时，系统会自动筛选已选院校，仅保留属于新选择国家的院校。
- 此逻辑在 `FormStateManager._on_target_selection_change` 中实现。

---

维护人：lijiapeng8@xdf.cn
版本：v2.5
