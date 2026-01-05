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
    exp = log_data_source.get("experience_details", {})

    mapping = {
        "background_university": "background_university",
        "background_major": "background_major_original",
        "gpa_scale": "gpa_scale",
        "exam_type": "exam_type",
        "exam_score": "exam_score",
        "language_type": "language_type",
        "research_count": "research_count",
        "award_count": "award_count",
        "internship_count": "internship_count",
        "paper_count": "paper_count",
    }

    res = {k: format_field(log_data_source.get(v)) for k, v in mapping.items()}
    res.update(
        {
            "gpa_score": format_float(log_data_source.get("gpa_raw"), 2),
            "language_score": format_float(log_data_source.get("language_score_raw"), 2),
            "target_universities": format_list_field(
                log_data_source.get("target_universities", [])
            ),
            "major_categories": format_list_field(
                session_manager.get("selected_major_categories", [])
            ),
            "target_majors": format_list_field(log_data_source.get("target_majors", [])),
        }
    )

    res.update(
        {
            f: _snippet(exp.get(f))
            for f in ("research_details", "award_details", "internship_details", "paper_details")
        }
    )

    return res


def log_first_submission_if_needed(
    session_manager: SessionManager,
    original_form_data: dict[str, Any] | None,
    input_data_from_form: dict[str, Any],
    session_key_last_submission_logged: str,
) -> None:
    if not session_manager.get(session_key_last_submission_logged, False):
        user_form_log = build_user_form_log(
            session_manager, original_form_data or input_data_from_form
        )
        submission_logger.info(f"用户输入: {user_form_log}")
        session_manager.set(**{session_key_last_submission_logged: True})
