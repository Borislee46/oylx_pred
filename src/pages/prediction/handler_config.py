from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pages.prediction.page_data_loader import machine_learning_model
    from src.utils.session_manager import SessionManager

PREDICTION_STATE_PREFIXES: tuple[str, ...] = (
    "_hk_pdf",
    "_hk_timeline",
    "_knn_",
    "_hk_percentile",
    "hk_sc_exp_",
)


@dataclass
class SessionKeys:
    form_data_changed: str = "form_data_changed"
    input_data: str = "input_data"
    predict_lock: str = "prediction_submit_lock"
    has_predicted: str = "has_predicted"
    is_school_selection_submit: str = "is_school_selection_submit"
    last_submission_logged: str = "last_submission_logged"


@dataclass
class UIStateKeys:
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
    hk_view_mode: str = "hk_view_mode"


@dataclass
class InternalStateKeys:
    pending_submission_data: str = "_pending_submission_data"
    edit_background: str = "_hk_edit_background"
    results_fresh_submission: str = "_results_fresh_submission"
    auto_submit_lead_in: str = "_auto_submit_lead_in"
    lead_in_processed: str = "_lead_in_processed"
    lead_in_low_confidence_labels: str = "lead_in_low_confidence_labels"
    form_anchor_scrolled: str = "_form_anchor_scrolled"
    hk_timeline_report_engaged: str = "_hk_timeline_report_engaged"
    hk_pdf_bytes: str = "_hk_pdf_bytes"


@dataclass
class FormStateKeys:
    gpa_raw_input: str = "gpa_raw_input"
    gpa_scale: str = "gpa_scale"
    gpa_conversion_cache: str = "gpa_conversion_cache"
    gpa_converter: str = "gpa_converter"
    last_gpa_warning_key: str = "last_gpa_warning_key"

    language_type: str = "language_type"
    language_score_input: str = "language_score_input"
    language_score_user_provided: str = "language_score_user_provided"
    language_score_input_error: str = "language_score_input_error"
    lang_conversion_cache: str = "lang_conversion_cache"
    last_lang_warning_key: str = "last_lang_warning_key"
    last_ielts_step_warning_key: str = "last_ielts_step_warning_key"

    standardized_test_type: str = "standardized_test_type"
    current_exam_score: str = "current_exam_score"

    selected_target_countries: str = "selected_target_countries"
    selected_target_universities: str = "selected_target_universities"
    selected_target_majors: str = "selected_target_majors"
    selected_major_categories: str = "selected_major_categories"
    target_options_cache: str = "target_options_cache"

    school_base_df: str = "school_base_df"
    background_university: str = "background_university"
    background_universities_cache: str = "background_universities_cache"
    background_majors_cache: str = "background_majors_cache"

    lead_in_form_summary: str = "lead_in_form_summary"
    lead_in_form_filled: str = "lead_in_form_filled"
    lead_in_missing_fields: str = "lead_in_missing_fields"
    user_history_data: str = "user_history_data"
    user_nickname: str = "user_nickname"
    user_message: str = "user_message"

    submitted: str = "submitted"
    current_user_id: str = "current_user_id"
    last_auto_save_ts: str = "last_auto_save_ts"
    last_saved_form_snapshot_hash: str = "last_saved_form_snapshot_hash"
    _input_form_pending_submission: str = "_input_form_pending_submission"
    research_count_initial: str = "research_count_initial"
    award_count_initial: str = "award_count_initial"
    internship_count_initial: str = "internship_count_initial"
    paper_count_initial: str = "paper_count_initial"

    research_details_initial: str = "research_details_initial"
    award_details_initial: str = "award_details_initial"
    internship_details_initial: str = "internship_details_initial"
    paper_details_initial: str = "paper_details_initial"

    background_university_initial: str = "background_university_initial"
    background_major_original_initial: str = "background_major_original_initial"

    background_major_2_original: str = "background_major_2_original"
    background_major_2: str = "background_major_2"
    is_dual_degree: str = "is_dual_degree"
    dual_alpha: str = "dual_alpha"


@dataclass
class FormWidgetKeys:
    background_university: str = "background_university_selectbox"
    background_major: str = "background_major_selectbox"
    background_major_2: str = "background_major_2_selectbox"
    dual_degree_type: str = "dual_degree_type_radio"
    dual_alpha: str = "dual_alpha_slider"
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
class PendingSubmissionData:
    input_data: dict
    all_universities: list[str]
    all_majors: list[str]
    original_form: dict | None = None


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
DEFAULT_INTERNAL_KEYS = InternalStateKeys()
DEFAULT_FORM_KEYS = FormStateKeys()
DEFAULT_WIDGET_KEYS = FormWidgetKeys()
