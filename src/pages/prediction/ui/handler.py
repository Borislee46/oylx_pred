from typing import TYPE_CHECKING

from src.pages.prediction.flow.pipeline import run_prediction_pipeline
from src.pages.prediction.handler_config import (
    FormSubmissionContext,
    SessionKeys,
)
from src.pages.prediction.prediction_preparation.data_preparer import prepare_input_data
from src.pages.prediction.prediction_preparation.fingerprint import (
    compute_df_fingerprint,
    compute_list_fingerprint,
)
from src.pages.prediction.result_modifier.experience_text_validator import (
    has_meaningful_experience_text,
)
from src.pages.prediction.results_handler import reset_prediction_results
from src.utils.app_data_loader import load_raw_cases_data
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from src.pages.prediction.page_data_loader import machine_learning_model
    from src.utils.session_manager import SessionManager

prediction_handler_logger = setup_logger("page3", "prediction")


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
) -> bool:
    input_data_with_lists = current_input_data.copy()
    input_data_with_lists["_all_universities_target"] = all_universities_target
    input_data_with_lists["_all_majors_target"] = all_majors_target
    input_data_with_lists["_cross_faculty_confirmed"] = session_manager.get(
        "cross_faculty_confirmed", False
    )

    experience_details = current_input_data.get("experience_details", {})
    has_valid_experience = has_meaningful_experience_text(experience_details)
    input_data_with_lists["_has_valid_experience"] = has_valid_experience

    cases_df_fingerprint = compute_df_fingerprint(load_raw_cases_data())
    all_universities_fingerprint = compute_list_fingerprint(all_universities_target)
    all_majors_fingerprint = compute_list_fingerprint(all_majors_target)

    prediction_result_model = run_prediction_pipeline(
        input_data_with_lists,
        "xgboost",
        cases_df_fingerprint,
        page_state.loaded_feature_names,
        all_universities_fingerprint,
        all_majors_fingerprint,
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


def handle_form_submission(ctx: FormSubmissionContext) -> None:
    from src.pages.prediction.input_form_components.cross_faculty_guard import (
        cross_faculty_confirm_dialog,
        quick_cross_faculty_check,
    )
    from src.pages.prediction.page_components.submission_logger import (
        log_first_submission_if_needed,
    )

    session_manager = ctx.session_manager
    page_state = ctx.page_state
    input_data_from_form = ctx.input_data_from_form
    session_keys = ctx.session_keys

    session_manager.set(**{session_keys.form_data_changed: False})

    if not all(
        [
            input_data_from_form.get("background_university"),
            input_data_from_form.get("background_major"),
        ]
    ):
        reset_prediction_results(session_manager)
        session_manager.delete(session_keys.input_data)
        return

    background_major = input_data_from_form.get("background_major")

    user_selected_categories = session_manager.get("selected_major_categories", []) or []
    user_selected_majors = session_manager.get("selected_target_majors", []) or []

    if background_major and (user_selected_categories or user_selected_majors):
        is_cross_faculty, bg_faculty, target_faculties, agent_approved = quick_cross_faculty_check(
            background_major,
            user_selected_categories,
            user_selected_majors,
            page_state.cases_df,
        )

        if is_cross_faculty:
            if agent_approved:
                session_manager.set(cross_faculty_confirmed=True)
            elif not session_manager.get("cross_faculty_confirmed", False):
                session_manager.set(
                    pending_prediction_data={
                        "input_data": input_data_from_form,
                        "all_universities": ctx.all_universities_target,
                        "all_majors": ctx.all_majors_target,
                        "original_form": ctx.original_form_data,
                    }
                )
                cross_faculty_confirm_dialog(session_manager, bg_faculty, target_faculties)
                return

    session_manager.set(**{session_keys.predict_lock: True})
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
    )


def handle_form_submission_legacy(
    session_manager: "SessionManager",
    page_state: "machine_learning_model",
    input_data_from_form: dict,
    all_universities_target: list[str],
    all_majors_target: list[str],
    original_form_data: dict | None,
    session_key_form_data_changed: str,
    session_key_input_data: str,
    session_key_predict_lock: str,
    session_key_has_predicted: str,
    session_key_is_school_selection_submit: str,
    session_key_last_submission_logged: str,
) -> None:
    session_keys = SessionKeys(
        form_data_changed=session_key_form_data_changed,
        input_data=session_key_input_data,
        predict_lock=session_key_predict_lock,
        has_predicted=session_key_has_predicted,
        is_school_selection_submit=session_key_is_school_selection_submit,
        last_submission_logged=session_key_last_submission_logged,
    )
    ctx = FormSubmissionContext(
        session_manager=session_manager,
        page_state=page_state,
        input_data_from_form=input_data_from_form,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        original_form_data=original_form_data,
        session_keys=session_keys,
    )
    handle_form_submission(ctx)
