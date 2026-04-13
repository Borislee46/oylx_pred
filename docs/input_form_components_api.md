# 预测表单组件与校验

| 项 | 说明 |
|----|------|
| 源码路径 | `src/pages/prediction/input_form_components/` |
| 依赖配置 | `config/gpa_conversion_rules.json`；语言/GPA 常量见 `form_config.py` |
| 消费者 | `pages/hk.py`、JSON 归一化 `api/json_api.py`（通过 `FormValidator` / `GPAConverter`） |

## 1. 文件职责一览

| 文件 | 职责 |
|------|------|
| `form_validator.py` | `FormValidator`：GPA/标化/语言/经历全量校验 |
| `validation_errors.py` | `ValidationError` 数据类 |
| `gpa_converter.py` | `GPAConverter`：院校/国家规则 + 线性缩放 |
| `language_score_converter.py` | 托福↔雅思分段映射 |
| `language_score_validator.py` | 范围、雅思 0.5 步长、文本解析 |
| `language_score_processor.py` | 海外院校语言加成 |
| `form_config.py` | 分制、语言类型、阈值、海外等级等常量 |
| `form_state.py` | `FormStateManager`：会话键、节流自动保存 |
| `target_options_service.py` | 国家/院校/大类/专业四级联动与聚合专业展开 |
| `widget_helpers.py` | `SelectBoxHelper` 封装选项缓存与历史恢复 |
| `form_ui.py` | `FormUIComponents` 聚合各区块 |
| `background_ui.py` / `target_ui.py` / `gpa_ui.py` / `language_ui.py` / `experience_ui.py` / `submit_ui.py` / `standardized_test_ui.py` | 各区块 UI |
| `cross_faculty_guard.py` | 跨学院申请二次确认 |

语言分数核心工具：`src/pages/prediction/core/utils.py`（`normalize_language_score`、`denormalize_language_score`）。

## 2. `FormValidator`（`form_validator.py`）

| 方法 | 行为摘要 |
|------|----------|
| `validate_standardized_test_score(exam_type, score)` | GRE/GMAT；空为合法；须整数且在 `GRE_SCORE_RANGE` / `GMAT_SCORE_RANGE` 内 |
| `normalize_gpa(raw_gpa, scale_key, background_university?, gpa_converter?)` | 先规则表后线性缩放到 4.0 制，截断 `[0,4]`，两位小数 |
| `validate_form_data(form_data, gpa_converter?)` | 返回 `list[ValidationError]`：背景校/专业、GPA 非零、标化、语言、经历计数与详情一致性等 |

## 3. `GPAConverter`（`gpa_converter.py`）

- `__init__(school_base_df)`：构建 `school_country_map`。
- `load_gpa_conversion_rules(path, mtime)`：`@st.cache_data` 读 JSON。
- `convert_gpa_by_rules(raw_gpa, scale_key, background_university?, country?)`：先匹配院校 `conversion_rules`，再国家 `country_rules`，命中则 `_apply_conversion_rule`（区间插值或 `fallback_multiplier`，支持百分比制）。

规则 JSON 顶层键示例：`conversion_rules`、`country_rules`（见仓库内 `config/gpa_conversion_rules.json`）。

## 4. 语言分模块

| 模块 | 要点 |
|------|------|
| `LanguageScoreConverter` | `toefl_to_ielts` / `ielts_to_toefl` 查表 |
| `LanguageScoreValidator` | `validate_ielts_step`；`validate_and_parse_score` 返回 `(score, err, has_input_error)` 供 UI 禁用提交 |
| `language_score_processor.apply_overseas_language_boost` | `school_level_service.is_overseas_school` 为真时按 `OVERSEAS_SCHOOL_LEVELS` × `LANGUAGE_BOOST_MULTIPLIERS` 放大，上限受 `LANGUAGE_SCORE_RANGES` 约束 |

## 5. `form_config.py` 常用常量

- `GPA_SCALES`、`DEFAULT_GPA_SCALE`
- `LANGUAGE_TYPES`、`LANGUAGE_SCORE_RANGES`、`DEFAULT_LANGUAGE_SCORES`
- `STANDARDIZED_TEST_TYPES`、`GRE_SCORE_RANGE`、`GMAT_SCORE_RANGE` 及 sigmoid 加成参数（供上游 GPA bonus）
- `TARGET_COUNTRY_UNIVERSITY_MAP`、`UNIVERSITY_SORT_ORDER`、`TARGET_COUNTRIES`
- `GPA_WARNING_THRESHOLDS`、`LANGUAGE_WARNING_THRESHOLDS`
- `OVERSEAS_SCHOOL_LEVELS`、`LANGUAGE_BOOST_MULTIPLIERS`

## 6. `ValidationError`（`validation_errors.py`）

字段：`field`、`message`、`severity`（`error` | `warning`）。`to_dict()` 供 API 序列化。

## 7. `FormStateManager`（`form_state.py`）

| 方法 | 行为 |
|------|------|
| `initialize_session_state(session_manager)` | 写入选择类、分数类、提交锁等初始键 |
| `on_form_change(session_manager, change_type?)` | 重置提交态；文本/选择不同节流周期触发自动保存 |
| `on_target_country_change` / `on_major_category_change` / … | 级联筛选与缓存失效 |
| `on_language_type_change` | 雅思↔托福分数互转（`lang_conversion_cache`） |
| `gpa_scale_changed` | 分制切换线性换算（`gpa_conversion_cache`） |
| `update_form_snapshot_hash_after_prediction` | 预测完成后更新快照，避免重复提交 |

自动保存仅会话内，**不落库**。

## 8. UI 区块契约

| 函数 | 主要返回值 |
|------|------------|
| `background_ui.render_background_section` | `(background_university, background_major_original, background_major)` |
| `target_ui.render_target_section` | `(prediction_universities, prediction_majors, all_unis, all_majors)`；未选校但选了国家/大类/专业时可扩展院校列表 |
| `gpa_ui.render_gpa_section` | `gpa_raw` |
| `standardized_test_ui.render_standardized_test_section` | `(exam_type, exam_score)` |
| `language_ui.render_language_section` | `(language_type, language_score_raw)`；海外背景为选填文本框 |
| `experience_ui.render_experience_section` | 四类 count + `experience_details` |
| `submit_ui.render_submit_button` | 是否点击 |

`FormUIComponents(session_manager)` 封装上述调用顺序与日志（`setup_logger("page3", "prediction")`）。

## 9. `target_options_service.py`

- `build_target_base_df`：合并案例与详情，产出 `base_df` 与 `university_country_map`。
- `compute_selection_cache_key` / `compute_options`：四级联动可选集；院校顺序服从 `UNIVERSITY_SORT_ORDER`。
- `expand_aggregated_majors_for_prediction`：聚合专业名 → 模型用原始专业名列表。

约定：UI 多选聚合名，下游 `target_majors` 仍为**原始专业名**；未选聚合专业时列表为空。

## 10. `SelectBoxHelper`（`widget_helpers.py`）

`render_cached_selectbox`、`render_multiselect`：统一选项缓存、`user_history_data` 默认值恢复与 `on_change` 回调。

## 11. `cross_faculty_guard.py`

| 函数 | 用途 |
|------|------|
| `check_cross_faculty_situation` | 精确：院校 + 原始专业 |
| `quick_cross_faculty_check` | 快速：背景专业 + 已选大类/聚合专业 |
| `cross_faculty_confirm_dialog` | `st.dialog` 二次确认 |

依赖：`cases_df` 的 `faculty`；专业详情表中学部/大类列（见源码列名）。

## 12. 集成示例

```python
from src.pages.prediction.core.utils import normalize_language_score
from src.pages.prediction.input_form_components.form_validator import FormValidator
from src.pages.prediction.input_form_components.gpa_converter import GPAConverter

def validate_and_prepare(form_data, school_base_df):
    gpa_conv = GPAConverter(school_base_df)
    errors = FormValidator.validate_form_data(form_data, gpa_converter=gpa_conv)
    if errors:
        return None, [e.message for e in errors]
    gpa_4 = FormValidator.normalize_gpa(
        form_data["gpa_raw"],
        form_data["gpa_scale"],
        form_data.get("background_university"),
        gpa_conv,
    )
    lang_norm = normalize_language_score(
        form_data.get("language_score_raw"),
        form_data.get("language_type"),
    )
    return {"gpa_4": gpa_4, "language_norm": lang_norm}, []
```

---

相关：[prediction_api.md](prediction_api.md)。

维护：与 `src/pages/prediction/input_form_components/` 同步更新。
