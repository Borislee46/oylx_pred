from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.core.utils import get_background_faculty
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.flow.result_adjuster import batch_adjust_results
from src.pages.prediction.flow.run_prediction import run_single_prediction
from src.pages.prediction.page_data_loader import cached_get_prediction_model
from src.pages.prediction.prediction_preparation.fingerprint import compute_df_fingerprint
from src.pages.prediction.prediction_preparation.input_validator import validate_and_clean_input
from src.pages.prediction.result_modifier.config import DEFAULT_TEXT_BOOST_CONFIG
from src.pages.prediction.result_modifier.faculty_filters import apply_out_of_scope_faculty_penalty
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
from src.utils.app_data_loader import load_bg_target_similarity_cache, load_raw_cases_data
from src.utils.logger import setup_logger
from src.utils.session_manager import PredictionResultModel

prediction_handler_logger = setup_logger("page3", "prediction")

ProgressCallback = Callable[[float, str], None]


def _update_progress(progress_cb: ProgressCallback | None, progress: float, text: str) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(progress, text)
    except Exception as e:
        prediction_handler_logger.debug(f"progress_cb 调用失败，已忽略: {e}")


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

    background_major = cleaned_input.get("background_major", "")
    user_specified_results = penalize_cross_major_without_cases(
        user_specified_results=user_specified_results,
        background_major=background_major,
        cases_df=cases_df,
    )

    bg_faculty_for_penalty = get_background_faculty(background_major, cases_df)
    sim_results = apply_out_of_scope_faculty_penalty(sim_results, bg_faculty_for_penalty)
    cross_results = apply_out_of_scope_faculty_penalty(cross_results, bg_faculty_for_penalty)
    user_specified_results = apply_out_of_scope_faculty_penalty(
        user_specified_results, bg_faculty_for_penalty
    )

    unique_results = combine_and_deduplicate_results(
        sim_results, cross_results, user_specified_results
    )

    if not unique_results:
        if meta is None:
            meta = {}
        meta["error"] = "empty_results"
        prediction_handler_logger.info("预测结果为空")
        meta.setdefault("user_message", "未找到匹配结果，请缩小目标范围或更换专业方向")
        return PredictionResultModel(meta=meta)

    return PredictionResultModel(
        similarity_results=sim_results,
        cross_major_results=cross_results,
        user_specified_results=user_specified_results,
        unified_results=unique_results,
        meta=meta,
    )


def run_prediction_pipeline_with_progress(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    all_universities_fingerprint: tuple[int, int],
    all_majors_fingerprint: tuple[int, int],
    *,
    progress_cb: ProgressCallback | None = None,
) -> PredictionResultModel:
    reporter = ProgressReporter(progress_cb)
    reporter.set_stage(0.08, 0.14, "加载模型...")
    prediction_model = cached_get_prediction_model(model_name)

    if prediction_model is None:
        return PredictionResultModel(meta={"error": "model_load_failed"})

    reporter.set_stage(0.14, 0.18, "加载案例数据...")
    cases_df = load_raw_cases_data()

    reporter.set_stage(0.18, 0.24, "校验案例数据一致性...")
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

    reporter.set_stage(0.24, 0.30, "清洗与标准化输入...")
    cleaned_input = validate_and_clean_input(input_data)
    reporter.advance_ratio(0.25)

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

    reporter.set_stage(0.30, 0.36, "准备相似度缓存...")
    bg_target_similarity_cache = load_bg_target_similarity_cache()
    reporter.advance_ratio(0.25)

    num_target_universities = len(cleaned_input.get("target_universities", []))
    cross_faculty_confirmed = input_data.get("_cross_faculty_confirmed", False)

    gpa = cleaned_input.get("gpa")
    language_score = cleaned_input.get("language_score")
    background_university = cleaned_input.get("background_university")

    reporter.set_stage(0.36, 0.48, "初始化概率校准器...")
    probability_adjuster = None
    if gpa is not None and language_score is not None:
        probability_adjuster = ProbabilityAdjuster(
            cases_df if cases_df is not None else pd.DataFrame()
        )
    reporter.advance_ratio(0.30)

    reporter.set_stage(0.48, 0.62, "生成初始预测...")
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
        progress_reporter=reporter,
    )

    if meta and meta.get("error"):
        prediction_handler_logger.info(f"预测未生成有效结果: {meta.get('error')}")
        return PredictionResultModel(meta=meta)

    reporter.set_stage(0.62, 0.72, "结果修正...")
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
        reporter.advance_ratio(0.25)

    sim_results = adjusted_results["sim_results"]
    cross_results = adjusted_results["cross_results"]
    user_specified_results = adjusted_results["user_specified_results"]

    reporter.set_stage(0.72, 0.80, "经验文本概率校准...")
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
        progress_reporter=reporter,
    )

    reporter.set_stage(0.80, 0.88, "跨专业惩罚与范围校验...")
    background_major = cleaned_input.get("background_major", "")
    user_specified_results = penalize_cross_major_without_cases(
        user_specified_results=user_specified_results,
        background_major=background_major,
        cases_df=cases_df,
    )
    reporter.advance_ratio(0.30)

    bg_faculty_for_penalty = get_background_faculty(background_major, cases_df)
    sim_results = apply_out_of_scope_faculty_penalty(sim_results, bg_faculty_for_penalty)
    cross_results = apply_out_of_scope_faculty_penalty(cross_results, bg_faculty_for_penalty)
    user_specified_results = apply_out_of_scope_faculty_penalty(
        user_specified_results, bg_faculty_for_penalty
    )
    reporter.advance_ratio(0.30)

    reporter.set_stage(0.88, 1.0, "合并去重...")
    unique_results = combine_and_deduplicate_results(
        sim_results, cross_results, user_specified_results
    )

    if not unique_results:
        if meta is None:
            meta = {}
        meta["error"] = "empty_results"
        meta.setdefault("user_message", "未找到匹配结果，请缩小目标范围或更换专业方向")
        prediction_handler_logger.info("预测结果为空")
        reporter.force_progress(1.0, "完成")
        return PredictionResultModel(meta=meta)

    reporter.force_progress(1.0, "完成")
    return PredictionResultModel(
        similarity_results=sim_results,
        cross_major_results=cross_results,
        user_specified_results=user_specified_results,
        unified_results=unique_results,
        meta=meta,
    )
