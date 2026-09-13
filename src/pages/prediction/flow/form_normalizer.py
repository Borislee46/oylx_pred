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
from src.pages.prediction.input_form_components.language_ui import (
    apply_overseas_language_boost,
)
from src.utils.logger import setup_logger
from src.utils.numeric import sigmoid_k
from src.utils.schools.level_service import get_school_level_service

form_normalizer_logger = setup_logger("page3", "prediction")


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

    bonus = max_bonus * sigmoid_k(exam_score, steepness, midpoint)
    return max(0.0, round(bonus, 4))


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


def resolve_language_score_raw(
    raw_score: float | None,
    *,
    user_provided: bool = False,
) -> float | None:
    if not user_provided:
        return None
    if raw_score is None:
        return None
    try:
        val = float(raw_score)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def calculate_processed_language_score(
    raw_score: float | None,
    language_type: str | None,
    background_university: str | None,
    is_overseas: bool = False,
    *,
    user_provided: bool = False,
) -> tuple[float | None, float | None]:
    raw_score = resolve_language_score_raw(raw_score, user_provided=user_provided)
    display_score = raw_score
    if (not display_score) and is_overseas:
        display_score = apply_overseas_language_boost(background_university, language_type)
        form_normalizer_logger.info(
            "海外本科语言豁免 | uni=%s type=%s boost=%.1f",
            background_university,
            language_type,
            display_score,
        )

    normalized_score = (
        normalize_language_score(display_score, language_type) if display_score else None
    )
    return display_score, normalized_score


def get_background_university_for_model(selected_background_university: str | None) -> str | None:
    """返回模型使用的背景院校名（当前与表单值一致；院校白名单解析在 modeling 层完成）。"""
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
    lang_user_provided = bool(form_data.get("language_score_user_provided", False))
    raw_lang = form_data.get("language_score_raw")
    lang_type = form_data.get("language_type")

    display_lang, final_normalized_lang_score = calculate_processed_language_score(
        raw_lang,
        lang_type,
        bg_uni,
        is_overseas,
        user_provided=lang_user_provided,
    )

    bg_uni_for_model = get_background_university_for_model(bg_uni)

    form_normalizer_logger.info(
        "表单归一化完成 | gpa=%.2f→%.2f(bonus=%.2f) lang=%s→%.2f "
        "overseas=%s unis=%d majors=%d dual=%s",
        gpa_before_bonus or 0,
        normalized_gpa or 0,
        gpa_bonus,
        display_lang or "缺失",
        final_normalized_lang_score or 0,
        is_overseas,
        len(form_data.get("target_universities", [])),
        len(form_data.get("target_majors", [])),
        form_data.get("is_dual_degree", False),
    )

    input_data = {
        "background_university": bg_uni_for_model,
        "background_major": form_data.get("background_major"),
        "background_major_original": form_data.get("background_major_original"),
        "background_major_2": form_data.get("background_major_2"),
        "background_major_2_original": form_data.get("background_major_2_original"),
        "is_dual_degree": form_data.get("is_dual_degree", False),
        "dual_alpha": form_data.get("dual_alpha", 0.85),
        "degree_type": form_data.get("degree_type", "辅修"),
        "target_universities": form_data.get("target_universities", []),
        "target_majors": form_data.get("target_majors", []),
        "gpa": gpa_before_bonus,
        "gpa_model": normalized_gpa,
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
