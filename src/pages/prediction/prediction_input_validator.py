from typing import Any

from src.pages.prediction.prediction_types import PredictionInput


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_list_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v]


def _safe_dict_str(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


def validate_and_clean_input(input_data: dict[str, Any]) -> PredictionInput:
    cleaned: PredictionInput = {
        "background_university": _safe_str(input_data.get("background_university")),
        "background_major": _safe_str(input_data.get("background_major")),
        "target_universities": _safe_list_str(input_data.get("target_universities")),
        "target_majors": _safe_list_str(input_data.get("target_majors")),
        "experience_details": {},
    }

    if (gpa := _safe_float(input_data.get("gpa"))) is not None:
        cleaned["gpa"] = gpa

    if (lang := _safe_float(input_data.get("language_score"))) is not None:
        cleaned["language_score"] = lang

    cleaned["internship_count"] = _safe_int(input_data.get("internship_count"))
    cleaned["research_count"] = _safe_int(input_data.get("research_count"))
    cleaned["award_count"] = _safe_int(input_data.get("award_count"))
    cleaned["paper_count"] = _safe_int(input_data.get("paper_count"))

    if "school_level" in input_data:
        cleaned["school_level"] = _safe_int(input_data["school_level"])

    exp_details = _safe_dict_str(input_data.get("experience_details"))

    for k in ("research_count", "award_count", "internship_count", "paper_count"):
        if k in input_data:
            val = cleaned.get(k, 0)
            exp_details[k] = str(val)

    cleaned["experience_details"] = exp_details

    return cleaned
