from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.page_data_loader import cached_get_prediction_model
from src.pages.prediction.prediction_fingerprint import compute_df_fingerprint
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
from src.pages.prediction.result_modifier.utils import has_meaningful_experience_text
from src.pages.prediction.results_handler import combine_and_deduplicate_results
from src.pages.prediction.run_prediction import run_single_prediction
from src.utils.app_data_loader import load_bg_target_similarity_cache, load_raw_cases_data
from src.utils.logger import setup_logger
from src.utils.session_manager import PredictionResultModel

prediction_handler_logger = setup_logger("page3", "prediction")


@st.cache_data(ttl=600, show_spinner="预测中...")
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

    if cases_df_fingerprint != compute_df_fingerprint(cases_df):
        prediction_handler_logger.debug(
            f"案例数据指纹不匹配: 期望 {cases_df_fingerprint}, 实际 {compute_df_fingerprint(cases_df)}"
        )

    all_universities_target_value = input_data.get("_all_universities_target", [])
    all_universities_target = (
        all_universities_target_value if isinstance(all_universities_target_value, list) else []
    )

    all_majors_target_value = input_data.get("_all_majors_target", [])
    all_majors_target = all_majors_target_value if isinstance(all_majors_target_value, list) else []

    bg_target_similarity_cache = load_bg_target_similarity_cache()

    target_universities_value = input_data.get("target_universities", [])
    target_universities = (
        target_universities_value if isinstance(target_universities_value, list) else []
    )
    num_target_universities = len(target_universities) if target_universities else 0

    sim_results, cross_results, user_specified_results, _ = run_single_prediction(
        current_input_data=input_data,
        prediction_model=prediction_model,
        cases_df=cases_df,
        bg_target_similarity_cache=bg_target_similarity_cache,
        expected_features=loaded_feature_names,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        num_target_universities=num_target_universities,
    )

    if all(x is None for x in [sim_results, cross_results, user_specified_results]):
        prediction_handler_logger.error("预测失败：所有结果为None")
        return PredictionResultModel()

    internship_count_value = input_data.get("internship_count", 0)
    internship_count = (
        int(internship_count_value) if isinstance(internship_count_value, (int, float, str)) else 0
    )

    user_specified_majors_value = input_data.get("target_majors", [])
    user_specified_majors = (
        user_specified_majors_value if isinstance(user_specified_majors_value, list) else []
    )

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

    background_major_value = input_data.get("background_major")
    background_major = background_major_value if isinstance(background_major_value, str) else None
    user_specified_results = penalize_cross_major_without_cases(
        user_specified_results=user_specified_results,
        background_major=background_major or "",
        cases_df=cases_df,
    )

    gpa_value = input_data.get("gpa")
    gpa = float(gpa_value) if isinstance(gpa_value, (int, float)) else None

    language_score_value = input_data.get("language_score")
    language_score = (
        float(language_score_value) if isinstance(language_score_value, (int, float)) else None
    )

    background_university_value = input_data.get("background_university")
    background_university = (
        background_university_value if isinstance(background_university_value, str) else None
    )

    probability_adjuster = None
    if gpa is not None and language_score is not None:
        probability_adjuster = ProbabilityAdjuster(
            cases_df if cases_df is not None else pd.DataFrame()
        )

    experience_details_value = input_data.get("experience_details", {})
    experience_details: dict[str, str] = (
        experience_details_value if isinstance(experience_details_value, dict) else {}
    )
    if isinstance(experience_details, dict):
        for k in ("research_count", "award_count", "internship_count", "paper_count"):
            if k in input_data:
                try:
                    value = input_data.get(k, 0) or 0
                    experience_details[k] = (
                        str(int(value)) if isinstance(value, (int, float, str)) else "0"
                    )
                except Exception:
                    experience_details[k] = "0"
    if has_meaningful_experience_text(experience_details):
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

    return PredictionResultModel(
        similarity_results=sim_results,
        cross_major_results=cross_results,
        user_specified_results=user_specified_results,
        unified_results=unique_results,
    )
