from typing import Any

from src.pages.prediction.core.utils import format_field, format_float, format_list_field
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

submission_logger = setup_logger("page3", "prediction")

DEFAULT_SNIPPET_MAX_LEN = 100


def _snippet(val: Any, max_len: int = DEFAULT_SNIPPET_MAX_LEN) -> str:
    if not val:
        return ""
    s = str(val).strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def build_user_form_log(
    session_manager: SessionManager, log_data_source: dict[str, Any]
) -> dict[str, Any]:
    language_type = log_data_source.get("language_type", "未知")
    exp_details = (
        log_data_source.get("experience_details") or {} if isinstance(log_data_source, dict) else {}
    )

    return {
        "background_university": format_field(log_data_source.get("background_university")),
        "background_major": format_field(log_data_source.get("background_major_original")),
        "gpa_scale": format_field(log_data_source.get("gpa_scale")),
        "gpa_score": format_float(log_data_source.get("gpa_raw"), 2),
        "exam_type": format_field(log_data_source.get("exam_type")),
        "exam_score": format_field(log_data_source.get("exam_score")),
        "target_universities": format_list_field(log_data_source.get("target_universities", [])),
        "major_categories": format_list_field(session_manager.get("selected_major_categories", [])),
        "target_majors": format_list_field(log_data_source.get("target_majors", [])),
        "language_type": format_field(language_type),
        "language_score": format_float(log_data_source.get("language_score_raw"), 2),
        "research_count": format_field(log_data_source.get("research_count")),
        "award_count": format_field(log_data_source.get("award_count")),
        "internship_count": format_field(log_data_source.get("internship_count")),
        "paper_count": format_field(log_data_source.get("paper_count")),
        "research_details": _snippet(exp_details.get("research_details")),
        "award_details": _snippet(exp_details.get("award_details")),
        "internship_details": _snippet(exp_details.get("internship_details")),
        "paper_details": _snippet(exp_details.get("paper_details")),
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
    log_data_source = original_form_data or input_data_from_form
    user_form_log = build_user_form_log(session_manager, log_data_source)
    submission_logger.info(f"用户输入: {user_form_log}")
    session_manager.set(**{session_key_last_submission_logged: True})
