from typing import Any

from src.pages.prediction.prediction_utils import format_field, format_float, format_list_field
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

submission_logger = setup_logger("page3", "prediction")


def build_user_form_log(
    session_manager: SessionManager, log_data_source: dict[str, Any]
) -> dict[str, Any]:
    language_type = log_data_source.get("language_type", "未知")
    exp_details = (
        (log_data_source.get("experience_details") or {})
        if isinstance(log_data_source, dict)
        else {}
    )

    def snippet(val: Any, max_len: int = 100) -> str:
        s = (val or "").strip()
        if not s:
            return ""
        return s[:max_len] + ("..." if len(s) > max_len else "")

    return {
        "background_university": format_field(log_data_source.get("background_university")),
        "background_major": format_field(log_data_source.get("background_major_original")),
        "gpa_scale": format_field(log_data_source.get("gpa_scale")),
        "gpa_score": format_float(log_data_source.get("gpa_raw"), 2),
        "target_universities": format_list_field(log_data_source.get("target_universities", [])),
        "major_categories": format_list_field(session_manager.get("selected_major_categories", [])),
        "target_majors": format_list_field(log_data_source.get("target_majors", [])),
        "language_type": format_field(language_type),
        "language_score": format_float(log_data_source.get("language_score_raw"), 2),
        "research_count": format_field(log_data_source.get("research_count")),
        "award_count": format_field(log_data_source.get("award_count")),
        "internship_count": format_field(log_data_source.get("internship_count")),
        "paper_count": format_field(log_data_source.get("paper_count")),
        "research_details": snippet(exp_details.get("research_details")),
        "award_details": snippet(exp_details.get("award_details")),
        "internship_details": snippet(exp_details.get("internship_details")),
        "paper_details": snippet(exp_details.get("paper_details")),
    }


def log_first_submission_if_needed(
    session_manager: SessionManager,
    original_form_data: dict[str, Any] | None,
    input_data_from_form: dict[str, Any],
    session_key_last_submission_logged: str,
) -> None:
    is_new_submission = not session_manager.get(session_key_last_submission_logged, False)
    if not is_new_submission:
        return
    log_data_source = original_form_data if original_form_data else input_data_from_form
    user_form_log = build_user_form_log(session_manager, log_data_source)
    submission_logger.info(f"用户输入: {user_form_log}")
    session_manager.set(**{session_key_last_submission_logged: True})
