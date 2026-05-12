from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pages.prediction.page_data_loader import machine_learning_model
    from src.utils.session_manager import SessionManager


@dataclass
class SessionKeys:
    form_data_changed: str = "form_data_changed"
    input_data: str = "input_data"
    predict_lock: str = "prediction_submit_lock"
    has_predicted: str = "has_predicted"
    is_school_selection_submit: str = "is_school_selection_submit"
    last_submission_logged: str = "last_submission_logged"


# ═══════════════════════════════════════════════════════════════════════════════
#  Session Key 注册表 — 所有模块共享 key 的单一定义源
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class UIStateKeys:
    """页面 UI 生命周期 key — hk.py / handler.py / cross_faculty_guard / results_handler 共享"""

    hk_ui_phase: str = "hk_ui_phase"
    hk_run_id: str = "hk_run_id"
    hk_last_error: str = "hk_last_error"
    pending_cross_faculty_prediction: str = "pending_cross_faculty_prediction"
    pending_prediction_data: str = "pending_prediction_data"
    cross_faculty_confirmed: str = "cross_faculty_confirmed"
    cross_faculty_cancelled: str = "cross_faculty_cancelled"
    form_expanded: str = "form_expanded"
    processing_lock: str = "processing_lock"
    lock_start_time: str = "lock_start_time"
    app_initialized: str = "app_initialized"
    fresh_prediction_result: str = "fresh_prediction_result"
    student_background_chart_visible: str = "student_background_chart_visible"
    prediction_results: str = "prediction_results"
    last_saved_results_hash: str = "last_saved_results_hash"
    previous_prediction_results: str = "previous_prediction_results"
    previous_input_data: str = "previous_input_data"


@dataclass
class FormStateKeys:
    """表单输入状态 key — input_form.py / form_state.py / form_bridge.py / 各 UI 组件共享"""

    # GPA
    gpa_raw_input: str = "gpa_raw_input"
    gpa_scale: str = "gpa_scale"
    gpa_conversion_cache: str = "gpa_conversion_cache"
    gpa_converter: str = "gpa_converter"
    last_gpa_warning_key: str = "last_gpa_warning_key"
    # 语言
    language_type: str = "language_type"
    language_score_input: str = "language_score_input"
    language_score_input_error: str = "language_score_input_error"
    lang_conversion_cache: str = "lang_conversion_cache"
    last_lang_warning_key: str = "last_lang_warning_key"
    last_ielts_step_warning_key: str = "last_ielts_step_warning_key"
    # 标准化考试
    standardized_test_type: str = "standardized_test_type"
    current_exam_score: str = "current_exam_score"
    # 目标
    selected_target_countries: str = "selected_target_countries"
    selected_target_universities: str = "selected_target_universities"
    selected_target_majors: str = "selected_target_majors"
    selected_major_categories: str = "selected_major_categories"
    target_options_cache: str = "target_options_cache"
    # 背景
    school_base_df: str = "school_base_df"
    background_university: str = "background_university"
    background_universities_cache: str = "background_universities_cache"
    background_majors_cache: str = "background_majors_cache"
    # LeadIn / Agent
    lead_in_form_summary: str = "lead_in_form_summary"
    lead_in_form_filled: str = "lead_in_form_filled"
    user_history_data: str = "user_history_data"
    user_nickname: str = "user_nickname"
    user_message: str = "user_message"
    # 提交 / 表单生命周期
    submitted: str = "submitted"
    current_user_id: str = "current_user_id"
    last_auto_save_ts: str = "last_auto_save_ts"
    last_saved_form_snapshot_hash: str = "last_saved_form_snapshot_hash"
    _input_form_pending_submission: str = "_input_form_pending_submission"
    # 经历初始值
    research_count_initial: str = "research_count_initial"
    award_count_initial: str = "award_count_initial"
    internship_count_initial: str = "internship_count_initial"
    paper_count_initial: str = "paper_count_initial"
    research_details_initial: str = "research_details_initial"
    award_details_initial: str = "award_details_initial"
    internship_details_initial: str = "internship_details_initial"
    paper_details_initial: str = "paper_details_initial"
    # 背景初始值
    background_university_initial: str = "background_university_initial"
    background_major_original_initial: str = "background_major_original_initial"


@dataclass
class FormWidgetKeys:
    """Streamlit widget key 注册表 — form_bridge.py / form_state.py / 各 UI 组件共享"""

    background_university: str = "background_university_selectbox"
    background_major: str = "background_major_selectbox"
    gpa_scale: str = "gpa_scale_widget_key"
    gpa_raw_input: str = "gpa_raw_input_widget"
    language_type: str = "language_type_widget_key"
    language_score: str = "language_score_input_widget"
    target_countries: str = "target_countries_multiselect"
    target_universities: str = "target_universities_multiselect"
    target_majors: str = "target_majors_multiselect"
    standardized_test_type: str = "standardized_test_type_widget"
    research_count: str = "research_count_input"
    award_count: str = "award_count_input"
    internship_count: str = "internship_count_input"
    paper_count: str = "paper_count_input"
    research_details: str = "research_details_input"
    award_details: str = "award_details_input"
    internship_details: str = "internship_details_input"
    paper_details: str = "paper_details_input"


@dataclass
class FormSubmissionContext:
    session_manager: "SessionManager"
    page_state: "machine_learning_model"
    input_data_from_form: dict
    all_universities_target: list[str]
    all_majors_target: list[str]
    original_form_data: dict | None
    session_keys: SessionKeys
    background_faculty: str | None = None
    admitted_combinations: set[tuple[str, str]] | None = None

    @classmethod
    def create(
        cls,
        session_manager: "SessionManager",
        page_state: "machine_learning_model",
        input_data_from_form: dict,
        all_universities_target: list[str],
        all_majors_target: list[str],
        original_form_data: dict | None = None,
        session_keys: SessionKeys | None = None,
        background_faculty: str | None = None,
        admitted_combinations: set[tuple[str, str]] | None = None,
    ) -> "FormSubmissionContext":
        return cls(
            session_manager=session_manager,
            page_state=page_state,
            input_data_from_form=input_data_from_form,
            all_universities_target=all_universities_target,
            all_majors_target=all_majors_target,
            original_form_data=original_form_data,
            session_keys=session_keys or SessionKeys(),
            background_faculty=background_faculty,
            admitted_combinations=admitted_combinations,
        )


DEFAULT_SESSION_KEYS = SessionKeys()
DEFAULT_UI_KEYS = UIStateKeys()
DEFAULT_FORM_KEYS = FormStateKeys()
DEFAULT_WIDGET_KEYS = FormWidgetKeys()
