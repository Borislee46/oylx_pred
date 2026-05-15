# Input Form Components 技术文档

## 1. 模块概述

`input_form_components` 是预测页面的表单输入模块，负责背景信息、GPA、语言成绩、标化成绩、目标选择、背提经历等表单的渲染、状态管理与校验，并与 SessionManager 协同实现自动保存、级联过滤、跨学部确认等交互逻辑。

## 2. 目录结构

```
input_form_components/
├── __init__.py
├── form_config.py          # 表单配置（分制、范围、院校映射）
├── form_state.py           # 表单状态管理
├── form_validator.py       # 表单校验
├── form_ui.py             # 表单 UI 聚合入口
├── validation_errors.py    # 校验错误类型
├── widget_helpers.py       # 通用控件封装（SelectBoxHelper）
├── background_ui.py        # 背景院校/专业
├── gpa_ui.py              # GPA 输入
├── gpa_converter.py        # GPA 转换（规则/归一化）
├── language_ui.py          # 语言成绩输入
├── language_score_converter.py   # 雅思/托福互转
├── language_score_validator.py  # 语言成绩校验
├── language_score_processor.py # 海外院校语言加成
├── standardized_test_ui.py # GRE/GMAT 输入
├── target_ui.py            # 目标国家/院校/学院/专业
├── target_options_service.py    # 目标选项计算与缓存
├── experience_ui.py        # 背提经历（科研/获奖/实习/论文）
├── submit_ui.py            # 提交按钮
└── cross_faculty_guard.py   # 跨学部确认弹窗
```

## 3. 核心流程

```
FormUIComponents (入口)
        │
        ├── render_background_section  → background_ui
        ├── render_gpa_section        → gpa_ui
        ├── render_standardized_test_section → standardized_test_ui
        ├── render_target_section     → target_ui + target_options_service
        ├── render_language_section   → language_ui
        ├── render_experience_section → experience_ui
        └── render_submit_button      → submit_ui
        │
        ▼
FormStateManager: 状态同步、变更回调、自动保存、级联过滤
        │
        ▼
FormValidator: 提交前校验 → ValidationError 列表
```

## 4. 组件说明

### 4.1 FormUIComponents

聚合入口，持有 `SessionManager` 与 `FormStateManager`，对外提供各区块的 `render_*` 方法，供预测页面按需调用。

### 4.2 FormStateManager

表单状态管理，与 `st.session_state` 通过 SessionManager 桥接。

**初始化**：`initialize_session_state` 设置默认值（目标选择、GPA、语言、背提、提交状态等）。

**变更回调**：
- `on_form_change`：表单任意变更时调用，重置 `submitted`、`form_data_changed` 等，并触发节流自动保存
- `on_target_country_change` / `on_target_university_change` 等：目标选择变更时同步 session，国家变更时级联过滤院校
- `gpa_scale_changed`：GPA 分制变更时按比例转换当前 GPA
- `on_language_type_change`：语言类型变更时转换成绩（雅思↔托福）
- `on_submit_click`：提交时标记 `submitted=True`

**自动保存**：`_auto_save_form_data` 按 `_get_current_form_snapshot` 计算 hash，节流（text 4s / select 1.5s）后更新 `last_saved_form_snapshot_hash`。

**级联过滤**：`_handle_target_cascade_filter` 在国家变更时，根据 `TARGET_COUNTRY_UNIVERSITY_MAP` 过滤已选院校。

### 4.3 FormValidator

提交前校验，返回 `list[ValidationError]`。

- **背景**：院校、专业必填
- **GPA**：非空、非 0，归一化后为 0 则分制无效
- **标化**：GRE 260–340、GMAT 200–800，整数
- **语言**：海外院校可选填；非海外必填且非 0；雅思须为 0.5 倍数
- **背提**：数量必填；数量为 0 时不能填写详情

`normalize_gpa` 优先走 `GPAConverter.convert_gpa_by_rules`（院校/国家规则），否则按分制线性归一化到 4.0。

### 4.4 FormConfig

| 配置 | 说明 |
|------|------|
| GPA_SCALES | 4.0/4.3/4.5/5.0/10/20/100 分制 |
| LANGUAGE_TYPES | 雅思、托福 |
| LANGUAGE_SCORE_RANGES | 各类型 min/max/step/format |
| TARGET_COUNTRY_UNIVERSITY_MAP | 国家→院校列表 |
| GPA_WARNING_THRESHOLDS | 各分制低分提示线 |
| LANGUAGE_WARNING_THRESHOLDS | 雅思 5.5、托福 72 |
| STANDARDIZED_TEST_TYPES | GRE、GMAT |
| GRE_SCORE_RANGE / GMAT_SCORE_RANGE | 分数范围 |

### 4.5 Background UI

- **院校**：`cases_df` + `school_base_df` 合并去重，按出现频次排序
- **专业**：从 `cases_df` 取 `background_major_original` → `background_major` 映射，按频次排序
- 使用 `SelectBoxHelper.render_cached_selectbox` 缓存选项

### 4.6 GPA UI

- 分制：`st.segmented_control`，变更时 `gpa_scale_changed` 转换当前值
- 数值：`st.number_input`，按分制限制 min/max/step
- 低于 `GPA_WARNING_THRESHOLDS` 时 toast 提示

### 4.7 GPAConverter

- **规则转换**：从 `config/gpa_conversion_rules.json` 加载，支持院校级 `conversion_rules` 与国家级 `country_rules`
- **区间规则**：`ranges` 内 `min/max` 映射到 `target_gpa` 或 `target_min/target_max` 插值
- **兜底**：`fallback_multiplier` 或 `is_percentage` 线性换算到 4.0

### 4.8 Language UI

- **海外院校**：`text_input`，`LanguageScoreValidator.validate_and_parse_score` 解析，选填
- **非海外**：`number_input`，必填，默认 `DEFAULT_LANGUAGE_SCORES`
- 类型切换时 `on_language_type_change` 自动转换成绩
- 低于 `LANGUAGE_WARNING_THRESHOLDS` 或雅思非 0.5 倍数时 toast

### 4.9 LanguageScoreConverter

雅思↔托福对照表转换，`toefl_to_ielts` / `ielts_to_toefl`。

### 4.10 LanguageScoreValidator

- `validate_ielts_step`：雅思须为 0.5 倍数
- `validate_score_range`：范围校验
- `validate_and_parse_score`：解析文本输入，返回 `(score, error_msg, has_input_error)`

### 4.11 Target UI / TargetOptionsService

- **选项计算**：`compute_options` 根据已选国家/院校/学院/专业过滤，返回四类选项列表，结果按 `selection_cache_key` 缓存
- **预测范围**：`_calculate_prediction_scope` 在未选院校/专业时展开为全量，`expand_aggregated_majors_for_prediction` 将聚合专业展开为具体专业列表
- **数据源**：`target_base_df` 来自 `cases_df` + `details_df`，含 `country`、`major_category`、`target_major_agg`

### 4.12 Experience UI

科研/获奖/实习/论文四类，每类：`number_input`（数量 0–99）+ `text_input`（详情选填），数据写入 `user_history_data`。提交时经 `TextPreprocessingAgent.validate_fields_batch()` 批量校验文本有效性。

### 4.13 Submit UI

`st.button`，`disabled` 条件：外部传入 `disabled_status` 或 `submitted && !form_data_changed`（已提交且未变更时不重复提交）。提交前设 `prediction_submit_lock` 防重复。

### 4.14 CrossFacultyGuard

- `quick_cross_faculty_check`：根据背景专业解析学部（调用 `BackgroundFacultyAgent`），与目标学院/专业学部对比，判断是否跨学部
- `cross_faculty_confirm_dialog`：`@st.dialog` 弹窗，用户确认后设置 `cross_faculty_confirmed`、`pending_cross_faculty_prediction`
- 已集成到 `ui/handler.py`：跨学部检测 → 弹窗确认 → 继续/取消预测

### 4.15 WidgetHelpers

- **SelectBoxHelper**：`render_cached_selectbox` 缓存选项、支持 `options_path_in_cache`、从 `user_history_data` 恢复默认；`render_multiselect` 封装多选
- 复用 `school_alias_resolver` 处理院校别名展开

### 4.14 CrossFacultyGuard

- `quick_cross_faculty_check`：根据背景专业解析学部，与目标学院/专业学部对比，判断是否跨学部
- `cross_faculty_confirm_dialog`：`@st.dialog` 弹窗，用户确认后设置 `cross_faculty_confirmed`、`pending_cross_faculty_prediction`

### 4.15 WidgetHelpers

- **SelectBoxHelper**：`render_cached_selectbox` 缓存选项、支持 `options_path_in_cache`、从 `user_history_data` 恢复默认；`render_multiselect` 封装多选

## 5. 类型定义

**ValidationError**：`field`、`message`、`severity`（error/warning）。

## 6. 依赖

- `streamlit`：UI 组件
- `pandas`：数据处理
- `SessionManager`：会话状态
- `school_level_service`：院校等级、海外判定、语言加成
