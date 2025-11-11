import streamlit as st

from src.pages.prediction.prediction_data_preparer import prepare_input_data
from src.pages.prediction.prediction_fingerprint import (
    compute_df_fingerprint,
    compute_list_fingerprint,
)
from src.pages.prediction.prediction_pipeline import run_prediction_pipeline
from src.pages.prediction.prediction_state_manager import persist_input_state
from src.pages.prediction.results_handler import reset_prediction_results
from src.utils.logger import setup_logger

prediction_handler_logger = setup_logger("page3", "prediction")


def run_prediction_with_guard(
    session_manager,
    page_state,
    current_input_data: dict,
    all_universities_target: list[str],
    all_majors_target: list[str],
    session_key_has_predicted: str,
    session_key_predict_lock: str,
) -> bool:
    try:
        input_data_with_lists = current_input_data.copy()
        input_data_with_lists["_all_universities_target"] = all_universities_target
        input_data_with_lists["_all_majors_target"] = all_majors_target

        cases_df_fingerprint = compute_df_fingerprint(page_state.cases_df)
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
        if prediction_result_model and prediction_result_model.unified_results is not None:
            session_manager.set(
                prediction_results=prediction_result_model,
                **{session_key_has_predicted: True, session_key_predict_lock: False},
                fresh_prediction_result=True,
            )
            from src.pages.prediction.input_form_components.form_state import FormStateManager

            FormStateManager.update_form_snapshot_hash_after_prediction(session_manager)
            return True

        reset_prediction_results(session_manager)
        session_manager.set(**{session_key_predict_lock: False})
        return False
    except Exception as e:
        error_message = f"预测过程中发生意外错误: {str(e)}"
        prediction_handler_logger.error(error_message, exc_info=True)
        st.error(f"预测过程中发生意外错误: {e}")
        reset_prediction_results(session_manager)
        session_manager.set(**{session_key_predict_lock: False})
        return False


def handle_form_submission(
    session_manager,
    page_state,
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
    from src.pages.prediction.input_form_components.cross_faculty_guard import (
        cross_faculty_confirm_dialog,
        quick_cross_faculty_check,
    )
    from src.pages.prediction.page_components.submission_logger import (
        log_first_submission_if_needed,
    )

    session_manager.set(**{session_key_form_data_changed: False})

    if not all(
        [
            input_data_from_form.get("background_university"),
            input_data_from_form.get("background_major"),
        ]
    ):
        reset_prediction_results(session_manager)
        session_manager.delete(session_key_input_data)
        return

    background_major = input_data_from_form.get("background_major")

    user_selected_categories = session_manager.get("selected_major_categories", []) or []
    user_selected_majors = session_manager.get("selected_target_majors", []) or []

    if background_major and (user_selected_categories or user_selected_majors):
        is_cross_faculty, bg_faculty, target_faculties = quick_cross_faculty_check(
            background_major,
            user_selected_categories,
            user_selected_majors,
            page_state.cases_df,
        )

        if is_cross_faculty:
            if session_manager.get("cross_faculty_cancelled", False):
                session_manager.set(
                    cross_faculty_cancelled=False,
                    cross_faculty_confirmed=False,
                    pending_prediction_data=None,
                    pending_cross_faculty_prediction=False,
                    prediction_submit_lock=False,
                    submitted=False,
                )
                st.info("已取消预测操作")
                return

            if not session_manager.get("cross_faculty_confirmed", False):
                session_manager.set(
                    pending_prediction_data={
                        "input_data": input_data_from_form,
                        "all_universities": all_universities_target,
                        "all_majors": all_majors_target,
                        "original_form": original_form_data,
                    }
                )
                cross_faculty_confirm_dialog(session_manager, bg_faculty, target_faculties)
                return

    session_manager.set(**{session_key_predict_lock: True})
    current_input_data = prepare_input_data(input_data_from_form)
    persist_input_state(
        session_manager,
        current_input_data,
        session_key_input_data,
        session_key_is_school_selection_submit,
    )
    log_first_submission_if_needed(
        session_manager,
        original_form_data,
        input_data_from_form,
        session_key_last_submission_logged,
    )
    run_prediction_with_guard(
        session_manager,
        page_state,
        current_input_data,
        all_universities_target,
        all_majors_target,
        session_key_has_predicted,
        session_key_predict_lock,
    )
