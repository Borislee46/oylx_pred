from typing import Any

from src.pages.prediction.core.utils import format_field, format_float, format_list_field
from src.pages.prediction.handler_config import DEFAULT_FORM_KEYS
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

    gpa_raw = log_data_source.get("gpa_raw")
    if gpa_raw is None:
        gpa_raw = session_manager.get(DEFAULT_FORM_KEYS.gpa_raw_input)
    gpa_scale = log_data_source.get("gpa_scale")
    if not gpa_scale:
        gpa_scale = session_manager.get(DEFAULT_FORM_KEYS.gpa_scale)

    language_score_raw = log_data_source.get("language_score_raw")
    if language_score_raw is None:
        language_score_raw = session_manager.get(DEFAULT_FORM_KEYS.language_score_input)

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

    enriched_source = {
        **log_data_source,
        "gpa_raw": gpa_raw,
        "gpa_scale": gpa_scale,
        "language_score_raw": language_score_raw,
    }
    res = {k: format_field(enriched_source.get(v)) for k, v in mapping.items()}
    res.update(
        {
            "gpa_score": format_float(gpa_raw, 2),
            "language_score": format_float(language_score_raw, 2),
            "target_universities": format_list_field(
                enriched_source.get("target_universities", [])
            ),
            "major_categories": format_list_field(
                session_manager.get("selected_major_categories", [])
            ),
            "target_majors": format_list_field(enriched_source.get("target_majors", [])),
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
