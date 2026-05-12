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
from src.utils.school_level_service import get_school_level_service


def calculate_gpa_bonus(exam_type: str | None, exam_score: float | None) -> float:
    if not exam_type or exam_type == "无" or exam_score is None:
        return 0.0

    configs = {
        "GRE": (GRE_BONUS_THRESHOLD, GRE_MAX_BONUS, GRE_SIGMOID_STEEPNESS, GRE_SIGMOID_MIDPOINT),
        "GMAT": (
            GMAT_BONUS_THRESHOLD,
            GMAT_MAX_BONUS,
            GMAT_SIGMOID_STEEPNESS,
            GMAT_SIGMOID_MIDPOINT,
        ),
    }

    if exam_type not in configs:
        return 0.0

    threshold, max_bonus, steepness, midpoint = configs[exam_type]
    if exam_score < threshold:
        return 0.0

    bonus = max_bonus / (1 + math.exp(-steepness * (exam_score - midpoint)))
    return max(0.0, float(bonus))


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
    raw_score: float | None,
    language_type: str | None,
    background_university: str | None,
    is_overseas: bool = False,
) -> tuple[float | None, float | None]:
    display_score = raw_score
    if (not display_score) and is_overseas:
        display_score = apply_overseas_language_boost(background_university, language_type)

    normalized_score = (
        normalize_language_score(display_score, language_type) if display_score else None
    )
    return display_score, normalized_score


def get_background_university_for_model(
    selected_background_university: str | None,
    cases_df,
    background_university_set: set[str] | None = None,
) -> str | None:
    if background_university_set is None:
        background_university_set = set(
            cases_df["background_university"].dropna().astype(str).unique()
        )

    return selected_background_university


def normalize_form_data_for_prediction(
    form_data: dict,
    cases_df,
    gpa_converter: GPAConverter | None,
    background_university_set: set[str] | None = None,
) -> dict:
    raw_gpa = form_data.get("gpa_raw")
    gpa_scale = form_data.get("gpa_scale")
    bg_uni = form_data.get("background_university")
    exam_type = form_data.get("exam_type")
    exam_score = form_data.get("exam_score")

    gpa_before_bonus = FormValidator.normalize_gpa(raw_gpa, gpa_scale, bg_uni, gpa_converter)
    normalized_gpa = gpa_before_bonus
    gpa_bonus = 0.0
    if normalized_gpa is not None:
        gpa_bonus = calculate_gpa_bonus(exam_type, exam_score)
        if gpa_bonus > 0:
            normalized_gpa += gpa_bonus

    school_service = get_school_level_service()
    is_overseas = school_service.is_overseas_school(bg_uni) if bg_uni else False
    raw_lang = form_data.get("language_score_raw")
    lang_type = form_data.get("language_type")

    display_lang, final_normalized_lang_score = calculate_processed_language_score(
        raw_lang, lang_type, bg_uni, is_overseas
    )

    bg_uni_for_model = get_background_university_for_model(
        bg_uni, cases_df, background_university_set
    )

    input_data = {
        "background_university": bg_uni_for_model,
        "background_major": form_data.get("background_major"),
        "background_major_original": form_data.get("background_major_original"),
        "target_universities": form_data.get("target_universities", []),
        "target_majors": form_data.get("target_majors", []),
        "gpa": normalized_gpa,
        "gpa_raw": gpa_before_bonus,
        "exam_type": exam_type,
        "exam_score": exam_score,
        "language_score": final_normalized_lang_score,
        "language_score_raw": display_lang,
        "language_type": lang_type,
        "research_count": form_data.get("research_count"),
        "award_count": form_data.get("award_count"),
        "internship_count": form_data.get("internship_count"),
        "paper_count": form_data.get("paper_count"),
        "experience_details": form_data.get("experience_details", {}),
    }

    return input_data
