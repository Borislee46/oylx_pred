from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.page_data_loader import cached_get_prediction_model
from src.pages.prediction.prediction_fingerprint import compute_df_fingerprint
from src.pages.prediction.prediction_input_validator import validate_and_clean_input
from src.pages.prediction.prediction_result_adjuster import batch_adjust_results
from src.pages.prediction.result_modifier.config import DEFAULT_TEXT_BOOST_CONFIG
from src.pages.prediction.result_modifier.probability_adjuster import (
    ProbabilityAdjuster,
    penalize_cross_major_without_cases,
)
from src.pages.prediction.result_modifier.professional_adjustment import (
    adjust_for_professional_majors,
)
from src.pages.prediction.result_modifier.text_boost_provider import (
    get_text_boost_provider,
)
from src.pages.prediction.results_handler import combine_and_deduplicate_results
from src.pages.prediction.run_prediction import run_single_prediction
from src.utils.app_data_loader import load_bg_target_similarity_cache, load_raw_cases_data
from src.utils.logger import setup_logger
from src.utils.session_manager import PredictionResultModel

prediction_handler_logger = setup_logger("page3", "prediction")


@st.cache_data(ttl=600, show_spinner=False)
def run_prediction_pipeline(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    all_universities_fingerprint: tuple[int, int],
    all_majors_fingerprint: tuple[int, int],
) -> PredictionResultModel:
    prediction_model = cached_get_prediction_model(model_name)

    if prediction_model is None:
        return PredictionResultModel()

    cases_df = load_raw_cases_data()

    actual_cases_fingerprint = compute_df_fingerprint(cases_df)
    if cases_df_fingerprint != actual_cases_fingerprint:
        prediction_handler_logger.warning(
            f"案例数据指纹不匹配: 期望 {cases_df_fingerprint}, 实际 {actual_cases_fingerprint}"
        )
        return PredictionResultModel(
            meta={
                "error": "cases_df_fingerprint_mismatch",
                "expected_cases_df_fingerprint": cases_df_fingerprint,
                "actual_cases_df_fingerprint": actual_cases_fingerprint,
            }
        )

    cleaned_input = validate_and_clean_input(input_data)

    current_input_data = input_data.copy()
    current_input_data.update(cleaned_input)

    all_universities_target_raw = input_data.get("_all_universities_target")
    all_universities_target: list[str] = (
        [str(item) for item in all_universities_target_raw]
        if isinstance(all_universities_target_raw, list)
        else []
    )
    all_majors_target_raw = input_data.get("_all_majors_target")
    all_majors_target: list[str] = (
        [str(item) for item in all_majors_target_raw]
        if isinstance(all_majors_target_raw, list)
        else []
    )

    bg_target_similarity_cache = load_bg_target_similarity_cache()

    num_target_universities = len(cleaned_input.get("target_universities", []))
    cross_faculty_confirmed = input_data.get("_cross_faculty_confirmed", False)

    gpa = cleaned_input.get("gpa")
    language_score = cleaned_input.get("language_score")
    background_university = cleaned_input.get("background_university")

    probability_adjuster = None
    if gpa is not None and language_score is not None:
        probability_adjuster = ProbabilityAdjuster(
            cases_df if cases_df is not None else pd.DataFrame()
        )

    sim_results, cross_results, user_specified_results, meta = run_single_prediction(
        current_input_data=current_input_data,
        prediction_model=prediction_model,
        cases_df=cases_df,
        bg_target_similarity_cache=bg_target_similarity_cache,
        expected_features=loaded_feature_names,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        num_target_universities=num_target_universities,
        cross_faculty_confirmed=cross_faculty_confirmed,
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        background_university=background_university,
    )

    if meta and meta.get("error"):
        prediction_handler_logger.info(f"预测未生成有效结果: {meta.get('error')}")
        return PredictionResultModel(meta=meta)

    internship_count = cleaned_input.get("internship_count", 0)
    user_specified_majors = cleaned_input.get("target_majors", [])

    results_to_adjust = [
        ("sim_results", sim_results),
        ("cross_results", cross_results),
        (
            "user_specified_results",
            user_specified_results if isinstance(user_specified_results, list) else [],
        ),
    ]

    adjusted_results = {}
    for result_name, result_list in results_to_adjust:
        adjusted_results[result_name] = adjust_for_professional_majors(
            result_list, internship_count, user_specified_majors
        )

    sim_results = adjusted_results["sim_results"]
    cross_results = adjusted_results["cross_results"]
    user_specified_results = adjusted_results["user_specified_results"]

    background_major = cleaned_input.get("background_major", "")
    user_specified_results = penalize_cross_major_without_cases(
        user_specified_results=user_specified_results,
        background_major=background_major,
        cases_df=cases_df,
    )

    experience_details = cleaned_input.get("experience_details", {})
    has_valid_experience = input_data.get("_has_valid_experience", False)

    if has_valid_experience:
        text_provider = get_text_boost_provider(DEFAULT_TEXT_BOOST_CONFIG)
    else:
        text_provider = None

    sim_results, cross_results, user_specified_results = batch_adjust_results(
        [sim_results, cross_results, user_specified_results],
        probability_adjuster,
        text_provider,
        experience_details,
        gpa,
        language_score,
        background_university,
    )

    unique_results = combine_and_deduplicate_results(
        sim_results, cross_results, user_specified_results
    )

    if not unique_results:
        if meta is None:
            meta = {}
        meta["error"] = "empty_results"
        prediction_handler_logger.info("预测结果为空")
        return PredictionResultModel(meta=meta)

    return PredictionResultModel(
        similarity_results=sim_results,
        cross_major_results=cross_results,
        user_specified_results=user_specified_results,
        unified_results=unique_results,
        meta=meta,
    )
