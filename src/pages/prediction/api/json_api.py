"""
非生产模块，后续待从streamlit框架把后端解耦出来用
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import pandas as pd

from src.pages.prediction.core.utils import get_background_faculty, is_new_major
from src.pages.prediction.flow.run_prediction import run_single_prediction
from src.pages.prediction.input_form_components import FormValidator, GPAConverter
from src.pages.prediction.input_form_components.cross_faculty_guard import (
    quick_cross_faculty_check,
)
from src.pages.prediction.input_form_components.form_config import (
    DEFAULT_GPA_SCALE,
    GPA_SCALES,
    LANGUAGE_TYPES,
)
from src.pages.prediction.input_form_components.validation_errors import ValidationError
from src.pages.prediction.page_data_loader import cached_get_prediction_model
from src.pages.prediction.prediction_preparation import (
    prepare_input_data,
    validate_and_clean_input,
)
from src.pages.prediction.prediction_preparation.form_normalizer import (
    normalize_form_data_for_prediction,
)
from src.pages.prediction.result_modifier import AdjustmentContext, ProbabilityAdjustmentPipeline
from src.pages.prediction.result_modifier.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.pages.prediction.result_modifier.config import DEFAULT_TEXT_BOOST_CONFIG
from src.pages.prediction.result_modifier.experience_text_validator import (
    has_meaningful_experience_text,
)
from src.pages.prediction.result_modifier.probability_adjuster import ProbabilityAdjuster
from src.pages.prediction.result_modifier.text_boost_provider import get_text_boost_provider
from src.pages.prediction.results_handler import combine_and_deduplicate_results
from src.utils.app_data_loader import (
    load_bg_target_similarity_cache,
    load_raw_cases_data,
    load_school_base_data,
)

logger = logging.getLogger(__name__)


def _list_str(value: Any) -> list[str]:
    return (
        [str(v).strip() for v in value if v and str(v).strip()] if isinstance(value, list) else []
    )


def _dict_str(value: Any) -> dict[str, str]:
    return (
        {str(k): str(v) for k, v in value.items() if k is not None}
        if isinstance(value, dict)
        else {}
    )


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value)) if value is not None and value != "" else default
    except (ValueError, TypeError):
        return default


def _parse_gpa_raw(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    num = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            num += ch
        elif num:
            break

    try:
        return float(num) if num else None
    except ValueError:
        return None


def _normalize_gpa_scale(value: Any) -> str:
    if value is None or value == "":
        return DEFAULT_GPA_SCALE

    if isinstance(value, (int, float)):
        v = float(value)
        if abs(v - 4.0) < 1e-9:
            return "4.0"
        if abs(v - 4.3) < 1e-9:
            return "4.3"
        if abs(v - 4.5) < 1e-9:
            return "4.5"
        if abs(v - 5.0) < 1e-9:
            return "5.0"
        if v.is_integer():
            s = str(int(v))
            if s == "4":
                return "4.0"
            if s == "5":
                return "5.0"
            return s
        return str(value).strip()

    s = str(value).strip()
    if s == "4":
        s = "4.0"
    elif s == "5":
        s = "5.0"

    if s not in GPA_SCALES:
        return DEFAULT_GPA_SCALE

    return s


def _build_form_data(payload: dict[str, Any]) -> dict[str, Any]:
    language_type = payload.get("language_type")
    if language_type not in LANGUAGE_TYPES:
        language_type = LANGUAGE_TYPES[0]

    return {
        "target_majors": _list_str(payload.get("target_majors")),
        "target_universities": _list_str(payload.get("target_universities")),
        "background_university": payload.get("background_university"),
        "background_major_original": payload.get("background_major_original"),
        "background_major": payload.get("background_major"),
        "gpa_raw": _parse_gpa_raw(payload.get("gpa_raw")),
        "gpa_scale": _normalize_gpa_scale(payload.get("gpa_scale")),
        "exam_type": payload.get("exam_type"),
        "exam_score": _safe_float(payload.get("exam_score")),
        "language_type": language_type,
        "language_score_raw": _safe_float(payload.get("language_score_raw")),
        "language_score_input_error": bool(payload.get("language_score_input_error", False)),
        "research_count": _safe_int(payload.get("research_count"), 0),
        "award_count": _safe_int(payload.get("award_count"), 0),
        "internship_count": _safe_int(payload.get("internship_count"), 0),
        "paper_count": _safe_int(payload.get("paper_count"), 0),
        "experience_details": _dict_str(payload.get("experience_details")),
    }


def _errors_to_dict(errors: list[ValidationError]) -> list[dict[str, Any]]:
    return [e.to_dict() for e in errors if isinstance(e, ValidationError)]


def validate_and_normalize(
    payload: dict[str, Any],
    cases_df: pd.DataFrame | None = None,
    school_base_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "errors": [
                {"field": "_", "message": "payload 必须是 JSON object", "severity": "error"}
            ],
        }

    cases_df = cases_df if cases_df is not None else load_raw_cases_data()
    school_base_df = school_base_df if school_base_df is not None else load_school_base_data()
    gpa_converter = GPAConverter(school_base_df)

    form_data = _build_form_data(payload)
    errors = FormValidator.validate_form_data(form_data, gpa_converter)
    if errors:
        return {"ok": False, "errors": _errors_to_dict(errors), "warnings": []}

    normalized_input, warnings = normalize_form_data_for_prediction(
        form_data, cases_df, gpa_converter
    )
    return {
        "ok": True,
        "errors": [],
        "warnings": warnings,
        "normalized_input": normalized_input,
        "original_form": form_data,
    }


@lru_cache(maxsize=1)
def _load_model_and_features():
    model = cached_get_prediction_model("xgboost")
    if model is None:
        raise ValueError("模型加载失败")
    features = model.feature_names
    if not isinstance(features, list) or not features:
        raise ValueError("模型特征列表为空")
    return model, features


def predict(payload: dict[str, Any], confirm_cross_faculty: bool = False) -> dict[str, Any]:
    try:
        cases_df = load_raw_cases_data()
        school_base_df = load_school_base_data()

        v = validate_and_normalize(payload, cases_df=cases_df, school_base_df=school_base_df)
        if not v.get("ok"):
            v["needs_confirmation"] = False
            return v

        normalized_input = v["normalized_input"]

        selected_categories = _list_str(payload.get("selected_major_categories"))
        selected_majors = _list_str(payload.get("selected_target_majors")) or _list_str(
            normalized_input.get("target_majors")
        )

        is_cross_faculty = False
        background_faculty = None
        target_faculties: set[str] = set()
        agent_approved = False

        if normalized_input.get("background_major") and (selected_categories or selected_majors):
            is_cross_faculty, background_faculty, target_faculties, agent_approved = (
                quick_cross_faculty_check(
                    normalized_input.get("background_major"),
                    selected_categories,
                    selected_majors,
                    cases_df,
                )
            )

        if is_cross_faculty and not agent_approved and not confirm_cross_faculty:
            return {
                "ok": True,
                "errors": [],
                "warnings": v.get("warnings", []),
                "needs_confirmation": True,
                "confirmation": {
                    "background_faculty": background_faculty,
                    "target_faculties": sorted(target_faculties),
                    "agent_approved": agent_approved,
                },
                "normalized_input": normalized_input,
                "result": None,
            }

        all_unis = _list_str(payload.get("all_universities_target"))
        all_majors = _list_str(payload.get("all_majors_target"))

        if not all_unis or not all_majors:
            from src.pages.prediction.core.utils import _data_manager

            if not all_unis:
                all_unis = sorted(_data_manager.valid_universities)
            if not all_majors:
                all_majors = sorted(_data_manager.valid_majors)

        input_data = prepare_input_data(normalized_input)
        cleaned_input = validate_and_clean_input(input_data)
        current_input_data = input_data.copy()
        current_input_data.update(cleaned_input)

        num_target_universities = len(cleaned_input.get("target_universities", []))
        cross_faculty_confirmed = bool(confirm_cross_faculty or agent_approved)

        gpa = cleaned_input.get("gpa")
        language_score = cleaned_input.get("language_score")
        background_university = cleaned_input.get("background_university")

        probability_adjuster = None
        if gpa is not None and language_score is not None:
            probability_adjuster = ProbabilityAdjuster(cases_df)

        model, expected_features = _load_model_and_features()
        bg_target_similarity_cache = load_bg_target_similarity_cache()

        sim_results, cross_results, user_specified_results, meta = run_single_prediction(
            current_input_data=current_input_data,
            prediction_model=model,
            cases_df=cases_df,
            bg_target_similarity_cache=bg_target_similarity_cache,
            expected_features=expected_features,
            all_universities_target=all_unis,
            all_majors_target=all_majors,
            num_target_universities=num_target_universities,
            cross_faculty_confirmed=cross_faculty_confirmed,
            probability_adjuster=probability_adjuster,
            gpa=gpa,
            language_score=language_score,
            background_university=background_university,
        )

        internship_count = cleaned_input.get("internship_count", 0)
        user_specified_majors = cleaned_input.get("target_majors", [])

        # 准备录取组合缓存
        admitted_combos = get_admitted_combinations_from_dataframe(
            cases_df, cleaned_input.get("background_major", "")
        )
        bg_faculty = get_background_faculty(cleaned_input.get("background_major", ""), cases_df)

        # 批量查询新专业状态缓存
        all_res_raw = sim_results + cross_results + (user_specified_results or [])
        new_major_cache = {}
        for r in all_res_raw:
            u, m = r.get("university"), r.get("major")
            if u and m and (u, m) not in new_major_cache:
                new_major_cache[(u, m)] = is_new_major(u, m)

        adj_ctx = AdjustmentContext(
            gpa=gpa,
            language_score=language_score,
            background_university=background_university,
            background_major=cleaned_input.get("background_major"),
            background_faculty=bg_faculty,
            internship_count=internship_count,
            user_specified_majors=user_specified_majors,
            experience_details=cleaned_input.get("experience_details", {}),
            cases_df=cases_df,
            admitted_combinations=admitted_combos,
            is_new_major_cache=new_major_cache,
        )

        has_valid_exp = has_meaningful_experience_text(adj_ctx.experience_details)
        text_provider = (
            get_text_boost_provider(DEFAULT_TEXT_BOOST_CONFIG) if has_valid_exp else None
        )

        adjuster_pipeline = ProbabilityAdjustmentPipeline(
            probability_adjuster=probability_adjuster,
            text_boost_provider=text_provider,
            enable_cross_major_penalty=True,
        )

        sim_results = adjuster_pipeline.adjust_batch(sim_results, adj_ctx)
        cross_results = adjuster_pipeline.adjust_batch(cross_results, adj_ctx)
        if user_specified_results:
            user_specified_results = adjuster_pipeline.adjust_batch(user_specified_results, adj_ctx)

        unified_results = combine_and_deduplicate_results(
            sim_results, cross_results, user_specified_results
        )
        result = {
            "similarity_results": sim_results,
            "cross_major_results": cross_results,
            "user_specified_results": user_specified_results,
            "unified_results": unified_results,
            "meta": meta,
        }

        return {
            "ok": True,
            "errors": [],
            "warnings": v.get("warnings", []),
            "needs_confirmation": False,
            "normalized_input": normalized_input,
            "result": result,
        }

    except Exception as e:
        logger.error(f"预测失败: {e}", exc_info=True)
        return {
            "ok": False,
            "errors": [{"field": "_", "message": f"Server Error: {str(e)}", "severity": "error"}],
            "warnings": [],
        }
