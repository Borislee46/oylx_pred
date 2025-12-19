import math

from src.pages.prediction.core.utils import normalize_language_score
from src.pages.prediction.input_form_components import FormValidator, GPAConverter
from src.pages.prediction.input_form_components.form_config import (
    GMAT_BONUS_THRESHOLD,
    GMAT_MAX_BONUS,
    GMAT_SIGMOID_MIDPOINT,
    GMAT_SIGMOID_STEEPNESS,
    GRE_BONUS_THRESHOLD,
    GRE_MAX_BONUS,
    GRE_SIGMOID_MIDPOINT,
    GRE_SIGMOID_STEEPNESS,
)
from src.pages.prediction.input_form_components.language_score_processor import (
    apply_overseas_language_boost,
)
from src.pages.prediction.user_background_analyzer import find_substitute_university
from src.utils.school_level_service import get_school_level_service


def calculate_gpa_bonus(exam_type, exam_score) -> float:
    if not exam_type or exam_type == "无" or not exam_score:
        return 0.0

    if exam_type == "GRE":
        if exam_score < GRE_BONUS_THRESHOLD:
            return 0.0
        bonus = GRE_MAX_BONUS / (
            1 + math.exp(-GRE_SIGMOID_STEEPNESS * (exam_score - GRE_SIGMOID_MIDPOINT))
        )
        return max(0.0, float(bonus))

    if exam_type == "GMAT":
        if exam_score < GMAT_BONUS_THRESHOLD:
            return 0.0
        bonus = GMAT_MAX_BONUS / (
            1 + math.exp(-GMAT_SIGMOID_STEEPNESS * (exam_score - GMAT_SIGMOID_MIDPOINT))
        )
        return max(0.0, float(bonus))

    return 0.0


def calculate_processed_gpa(
    raw_gpa, scale, background_university, gpa_converter, exam_type=None, exam_score=None
) -> float | None:
    normalized_gpa = FormValidator.normalize_gpa(
        raw_gpa, scale, background_university, gpa_converter
    )

    if normalized_gpa is not None:
        bonus_gpa = calculate_gpa_bonus(exam_type, exam_score)
        if bonus_gpa > 0:
            normalized_gpa += bonus_gpa
    return normalized_gpa


def calculate_processed_language_score(
    raw_score, language_type, background_university, is_overseas=False
) -> tuple[float | None, float | None]:
    display_score = raw_score
    if (display_score is None or display_score == 0) and is_overseas:
        display_score = apply_overseas_language_boost(background_university, language_type)

    normalized_score = None
    if display_score is not None:
        normalized_score = normalize_language_score(display_score, language_type)

    return display_score, normalized_score


def get_background_university_for_model(
    selected_background_university: str | None,
    cases_df,
    background_university_set: set[str] | None = None,
) -> str | None:
    if not selected_background_university:
        return None

    if background_university_set is None:
        background_university_set = set(
            cases_df["background_university"].dropna().astype(str).unique()
        )

    if selected_background_university not in background_university_set:
        return find_substitute_university(selected_background_university, cases_df)

    return selected_background_university


def normalize_form_data_for_prediction(
    form_data: dict,
    cases_df,
    gpa_converter: GPAConverter | None,
    background_university_set: set[str] | None = None,
) -> tuple[dict, list[str]]:
    warnings: list[str] = []

    normalized_gpa = calculate_processed_gpa(
        form_data.get("gpa_raw"),
        form_data.get("gpa_scale"),
        form_data.get("background_university"),
        gpa_converter,
        form_data.get("exam_type"),
        form_data.get("exam_score"),
    )

    bonus_gpa = calculate_gpa_bonus(form_data.get("exam_type"), form_data.get("exam_score"))
    if normalized_gpa is not None and bonus_gpa > 0:
        warnings.append(f"标化成绩加成生效: GPA +{bonus_gpa:.3f}")

    school_service = get_school_level_service()
    background_university = form_data.get("background_university")
    is_overseas = (
        school_service.is_overseas_school(background_university) if background_university else False
    )

    language_score_for_submission = form_data.get("language_score_raw")
    if (
        language_score_for_submission is None or language_score_for_submission == 0
    ) and is_overseas:
        warnings.append("海外背景触发语言成绩默认加成")

    _, final_normalized_lang_score = calculate_processed_language_score(
        language_score_for_submission,
        form_data.get("language_type"),
        background_university,
        is_overseas,
    )

    background_uni_for_model = get_background_university_for_model(
        form_data.get("background_university"), cases_df, background_university_set
    )

    input_data = {
        "background_university": background_uni_for_model,
        "background_major": form_data.get("background_major"),
        "background_major_original": form_data.get("background_major_original"),
        "target_universities": form_data.get("target_universities", []),
        "target_majors": form_data.get("target_majors", []),
        "gpa": normalized_gpa,
        "language_score": final_normalized_lang_score,
        "language_type": form_data.get("language_type"),
        "research_count": form_data.get("research_count"),
        "award_count": form_data.get("award_count"),
        "internship_count": form_data.get("internship_count"),
        "paper_count": form_data.get("paper_count"),
        "experience_details": form_data.get("experience_details", {}),
    }

    return input_data, warnings
