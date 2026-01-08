import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.pages.prediction.config.ui_messages import PIPELINE_MESSAGES
from src.pages.prediction.core.utils import get_background_faculty
from src.pages.prediction.flow.pipeline import run_prediction_pipeline_with_progress
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.handler_config import (
    FormSubmissionContext,
    SessionKeys,
)
from src.pages.prediction.input_form_components.cross_faculty_guard import (
    cross_faculty_confirm_dialog,
    quick_cross_faculty_check,
)
from src.pages.prediction.page_components.submission_logger import (
    log_first_submission_if_needed,
)
from src.pages.prediction.prediction_preparation import (
    compute_list_fingerprint,
    prepare_input_data,
)
from src.pages.prediction.result_modifier.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.pages.prediction.result_modifier.experience_text_validator import (
    has_meaningful_experience_text,
)
from src.pages.prediction.results_handler import reset_prediction_results
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from src.pages.prediction.page_data_loader import machine_learning_model
    from src.utils.session_manager import SessionManager

prediction_handler_logger = setup_logger("page3", "prediction")

ProgressCallback = Callable[[str], None]


def _update_progress(progress_cb: ProgressCallback | None, text: str | list[str]) -> None:
    if progress_cb is not None:
        if isinstance(text, list):
            t = str(random.choice(text) if text else "").strip()
        else:
            t = str(text or "").strip()
        progress_cb(t)


def persist_input_state(
    session_manager: "SessionManager",
    current_input_data: dict,
    session_keys: SessionKeys,
) -> None:
    session_manager.set(
        **{
            session_keys.input_data: current_input_data,
            session_keys.is_school_selection_submit: False,
        }
    )


def run_prediction_with_guard(
    session_manager: "SessionManager",
    page_state: "machine_learning_model",
    current_input_data: dict,
    all_universities_target: list[str],
    all_majors_target: list[str],
    session_keys: SessionKeys,
    progress_cb: ProgressCallback | None = None,
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
) -> bool:
    input_data_with_lists = current_input_data.copy()
    input_data_with_lists["_all_universities_target"] = all_universities_target
    input_data_with_lists["_all_majors_target"] = all_majors_target
    input_data_with_lists["_cross_faculty_confirmed"] = session_manager.get(
        "cross_faculty_confirmed", False
    )

    experience_details = current_input_data.get("experience_details", {})
    pre_reporter = ProgressReporter(progress_cb)
    has_valid_experience = has_meaningful_experience_text(
        experience_details, progress_reporter=pre_reporter
    )
    input_data_with_lists["_has_valid_experience"] = has_valid_experience

    cases_df_fingerprint = page_state.cases_df_fingerprint
    all_universities_fingerprint = compute_list_fingerprint(all_universities_target)
    all_majors_fingerprint = compute_list_fingerprint(all_majors_target)

    prediction_result_model = run_prediction_pipeline_with_progress(
        input_data_with_lists,
        "xgboost",
        cases_df_fingerprint,
        page_state.loaded_feature_names,
        all_universities_fingerprint,
        all_majors_fingerprint,
        progress_cb=progress_cb,
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
        page_state=page_state,
    )
    if prediction_result_model and prediction_result_model.meta:
        session_manager.set(**prediction_result_model.meta)

    unified = getattr(prediction_result_model, "unified_results", None)
    if isinstance(unified, list) and len(unified) > 0:
        session_manager.set(
            prediction_results=prediction_result_model,
            **{session_keys.has_predicted: True, session_keys.predict_lock: False},
            fresh_prediction_result=True,
            student_background_chart_visible=True,
        )
        from src.pages.prediction.input_form_components.form_state import FormStateManager

        FormStateManager.update_form_snapshot_hash_after_prediction(session_manager)
        return True

    reset_prediction_results(session_manager)
    session_manager.set(**{session_keys.predict_lock: False})
    return False


def handle_form_submission(
    ctx: FormSubmissionContext, progress_cb: ProgressCallback | None = None
) -> None:
    session_manager = ctx.session_manager
    page_state = ctx.page_state
    input_data_from_form = ctx.input_data_from_form
    session_keys = ctx.session_keys

    session_manager.set(**{session_keys.form_data_changed: False})

    bg_major = input_data_from_form.get("background_major")
    if not all([input_data_from_form.get("background_university"), bg_major]):
        reset_prediction_results(session_manager)
        session_manager.delete(session_keys.input_data)
        return

    if not ctx.background_faculty:
        ctx.background_faculty = get_background_faculty(bg_major, page_state.cases_df)
    if not ctx.admitted_combinations:
        ctx.admitted_combinations = get_admitted_combinations_from_dataframe(
            page_state.cases_df, bg_major
        )

    user_selected_categories = session_manager.get("selected_major_categories", []) or []
    user_selected_majors = session_manager.get("selected_target_majors", []) or []

    if bg_major and (user_selected_categories or user_selected_majors):
        _update_progress(progress_cb, PIPELINE_MESSAGES["cross_check"])
        is_cross_faculty, bg_faculty, target_faculties, agent_approved = quick_cross_faculty_check(
            bg_major,
            user_selected_categories,
            user_selected_majors,
            page_state.cases_df,
        )

        if is_cross_faculty:
            if agent_approved:
                session_manager.set(cross_faculty_confirmed=True)
            elif not session_manager.get("cross_faculty_confirmed", False):
                _update_progress(progress_cb, "检测到跨学科申请跨度较大，需进一步评估风险...")
                session_manager.set(
                    hk_ui_phase="awaiting_confirm",
                    pending_prediction_data={
                        "input_data": input_data_from_form,
                        "all_universities": ctx.all_universities_target,
                        "all_majors": ctx.all_majors_target,
                        "original_form": ctx.original_form_data,
                    },
                )
                cross_faculty_confirm_dialog(session_manager, bg_faculty, target_faculties)
                return

    session_manager.set(**{session_keys.predict_lock: True})
    session_manager.set(hk_ui_phase="running", hk_last_error=None)
    current_input_data = prepare_input_data(input_data_from_form)
    persist_input_state(session_manager, current_input_data, session_keys)
    log_first_submission_if_needed(
        session_manager,
        ctx.original_form_data,
        input_data_from_form,
        session_keys.last_submission_logged,
    )

    run_prediction_with_guard(
        session_manager,
        page_state,
        current_input_data,
        ctx.all_universities_target,
        ctx.all_majors_target,
        session_keys,
        progress_cb=progress_cb,
        background_faculty=ctx.background_faculty,
        admitted_combinations=ctx.admitted_combinations,
    )
