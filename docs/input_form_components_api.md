## 表单组件与校验 API 文档（src/pages/prediction/input_form_components）

本模块涵盖表单校验、GPA/语言分数转换、配置常量与表单状态管理，供页面与服务侧复用/对接。

### 组成模块
- 表单校验器：`form_validator.py`
- GPA 转换：`gpa_converter.py`
- 语言分数换算：`language_score_converter.py`
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

- `validate_form_data(form_data: dict, gpa_converter: GPAConverter | None=None) -> list[str]`
  - 业务校验，返回中文错误消息列表，包含：
    - 背景院校/专业必填，且映射有效
    - GPA 不能为空/不为 0，分制有效
    - 雅思分数必须是 0.5 的倍数；语言分数不为 0
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
- `ielts_to_toefl(ielts_score) -> float | None`
  - 内置区间/最邻近映射表，适用于切换语言类型时的即时换算。

---

### 四、配置常量（form_config.py）
- `GPA_SCALES`：分制上限、步长与格式（如 `{"4.0": {"max": 4.0, ...}}`）
- `DEFAULT_GPA_SCALE`：默认分制（`"4.0"`）
- `LANGUAGE_TYPES`：`["雅思", "托福"]`
- `LANGUAGE_SCORE_RANGES`：分数范围、步长与显示格式
- `UNIVERSITY_SORT_ORDER`、`TARGET_COUNTRY_UNIVERSITY_MAP`、`TARGET_COUNTRIES`
- 低分/告警阈值：`GPA_LOW_THRESHOLDS`、`LANGUAGE_LOW_THRESHOLDS`、`GPA_WARNING_THRESHOLDS`、`LANGUAGE_WARNING_THRESHOLDS`

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

---

### 六、最小集成示例
```python
from src.pages.prediction.input_form_components.form_validator import FormValidator
from src.pages.prediction.input_form_components.gpa_converter import GPAConverter

def validate_and_prepare(form_data, school_base_df):
    gpa_conv = GPAConverter(school_base_df)
    errors = FormValidator.validate_form_data(form_data, gpa_converter=gpa_conv)
    if errors:
        return None, errors

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
  - 说明：**v2.3 重构** - 使用 `widget_helpers.SelectBoxHelper` 渲染，将选项生成逻辑与 UI 渲染解耦；背景院校候选来自 `cases_df` 与 `load_school_base_data()` 的并集，并按案例频次降序排序。

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
  - 说明：类型切换、分数输入、阈值与步长校验；键：`language_type_widget_key`、`language_score_input_widget`。当无历史值时默认分数：雅思 6.5、托福 86；雅思强制 0.5 步长并提示。

- 其他经历（`experience_ui.render_experience_section(session_manager, form_state_manager, logger)`）
  - 返回：`(research_count, award_count, internship_count, paper_count, experience_details)`
  - 说明：键：`*_count_input`、`*_details_input`（四类：科研/获奖/实习/论文）。

- 提交按钮（`submit_ui.render_submit_button(session_manager, form_state_manager, disabled_status=False)`）
  - 返回：`bool`（是否点击）
  - 说明：禁用提示与提交态控制；键：`submit_button_key`。

---

### 十、UI 组合器（form_ui.py）

- `class FormUIComponents(session_manager)`
  - 作用：聚合并渲染背景/目标/GPA/语言/经历/提交六大区块，封装日志器、状态写入与提示。
  - 说明：内部使用上述分拆组件方法；日志器来源于 `utils.logger.setup_logger("page3", "prediction")`。

---

### 八、目标筛选服务（target_options_service.py）

- `build_target_base_df(cases_df, details_df) -> (base_df, university_country_map)`
  - 合并 `cases_df[['target_university','target_major']]` 与 `details_df`，生成列：`target_university`、`target_major`、`major_category`、`target_major_agg`、`country`。
  - 兼容：`details_df` 无 `专业英文名称_聚合` 时回退为 `专业英文名称`。

- `compute_selection_cache_key(countries, universities, categories, majors) -> str`
  - 用于 `target_options_cache` 的稳定键，保持与旧实现一致。

- `compute_options(base_df, countries, universities, categories, majors)`
  - 返回四级可选项列表：国家、院校、专业大类、聚合专业；维持院校排序（`UNIVERSITY_SORT_ORDER`）。

- `expand_aggregated_majors_for_prediction(base_df, countries, universities, categories, aggregated) -> list[str]`
  - 将选择的聚合专业展开为原始 `target_major` 列表；当未选择聚合专业时返回空列表。

行为约定（重要）：
- UI 多选使用聚合名（`target_major_agg`）；传入后续流程的字段仍是 `target_majors`，内容为“原始专业名列表”。
- 未选择聚合专业时，`target_majors` 应为空，以免触发表单规则“如果先勾选目标专业，请选择目标专业对应的学校”。

自动扩展逻辑（与 `target_ui.render_target_section` 一致）：
- 当未选择院校，但选择了国家/专业大类/聚合专业之一时：`prediction_universities` 自动扩展为当前筛选条件下的所有可选院校。
- 当未选择聚合专业，但已选择国家/院校/专业大类之一时：按筛选条件给出聚合专业候选；最终传递到后续流程前，会调用 `expand_aggregated_majors_for_prediction` 将聚合专业展开为原始 `target_major` 列表。
- 选项生成与缓存：会基于 `compute_selection_cache_key` 将四级选择序列化为键；若命中 `target_options_cache` 则直接复用，未命中则实时计算并写入缓存。

---

### 九、UI 辅助工具 (widget_helpers.py) (v2.3 新增)
- `class SelectBoxHelper`
  - `render_cached_selectbox(label, widget_key, cache_key, history_key, options_generator_func, on_change_callback, options_path_in_cache=None)`：封装**单选框**通用逻辑，包括选项生成、缓存、历史默认值恢复；当生成结果为字典时，可通过 `options_path_in_cache` 指定选项列表所在的键（如 `"majors_display"`）。
  - `render_multiselect(...)`：封装了**多选框**的通用渲染逻辑。

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

维护人：lijiapeng8@xdf.cn
版本：v2.4
