from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.config.ui_messages import PIPELINE_MESSAGES
from src.pages.prediction.core.utils import get_background_faculty, is_new_major
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.flow.run_prediction import run_single_prediction
from src.pages.prediction.page_data_loader import cached_get_prediction_model
from src.pages.prediction.prediction_preparation import (
    compute_df_fingerprint,
    validate_and_clean_input,
)
from src.pages.prediction.result_modifier import (
    AdjustmentContext,
    ProbabilityAdjustmentPipeline,
)
from src.pages.prediction.result_modifier.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.pages.prediction.result_modifier.config import DEFAULT_TEXT_BOOST_CONFIG
from src.pages.prediction.result_modifier.probability_adjuster import (
    ProbabilityAdjuster,
)
from src.pages.prediction.result_modifier.text_boost_provider import (
    get_text_boost_provider,
)
from src.pages.prediction.results_handler import combine_and_deduplicate_results
from src.utils.app_data_loader import load_bg_target_similarity_cache, load_raw_cases_data
from src.utils.background_animator import BackgroundAnimator
from src.utils.logger import setup_logger
from src.utils.session_manager import PredictionResultModel

prediction_handler_logger = setup_logger("page3", "prediction")

ProgressCallback = Callable[[float, str], None]


def _execute_prediction_pipeline(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    all_universities_target: list[str],
    all_majors_target: list[str],
    reporter: ProgressReporter,
) -> PredictionResultModel:
    def _fmt_value(val: Any, *, none_text: str = "未提供") -> str:
        if val is None:
            return none_text
        s = str(val).strip()
        return s if s else none_text

    msg_load_model = PIPELINE_MESSAGES["wake_model"].format(model_name=_fmt_value(model_name))
    reporter.set_stage(0.08, 0.14, msg_load_model)
    with BackgroundAnimator(reporter, msg_load_model):
        prediction_model = cached_get_prediction_model(model_name)

    if prediction_model is None:
        return PredictionResultModel(meta={"error": "model_load_failed"})

    msg_load_cases = PIPELINE_MESSAGES["search_cases"]
    reporter.set_stage(0.14, 0.18, msg_load_cases)
    with BackgroundAnimator(reporter, msg_load_cases):
        cases_df = load_raw_cases_data()
    case_count = len(cases_df) if cases_df is not None else 0

    msg_check_fingerprint = PIPELINE_MESSAGES["check_consistency"].format(count=case_count)
    reporter.set_stage(0.18, 0.24, msg_check_fingerprint)
    with BackgroundAnimator(reporter, msg_check_fingerprint):
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

    msg_clean_input = PIPELINE_MESSAGES["build_features"].format(dim=len(loaded_feature_names))
    reporter.set_stage(
        0.24,
        0.30,
        msg_clean_input,
    )
    with BackgroundAnimator(reporter, msg_clean_input):
        cleaned_input = validate_and_clean_input(input_data)
    reporter.advance_ratio(0.25)

    current_input_data = input_data.copy()
    current_input_data.update(cleaned_input)

    reporter.set_stage(0.30, 0.36, PIPELINE_MESSAGES["prepare_pool"])
    bg_target_similarity_cache = load_bg_target_similarity_cache()
    reporter.emit(
        PIPELINE_MESSAGES["load_similarity"].format(
            count=(
                len(bg_target_similarity_cache)
                if isinstance(bg_target_similarity_cache, dict)
                else 0
            )
        ),
        force=False,
    )
    reporter.advance_ratio(0.25)

    num_target_universities = len(cleaned_input.get("target_universities", []))
    cross_faculty_confirmed = input_data.get("_cross_faculty_confirmed", False)

    gpa = cleaned_input.get("gpa")
    language_score = cleaned_input.get("language_score")
    background_university = cleaned_input.get("background_university")
    background_major = cleaned_input.get("background_major", "")

    probability_calibration_status = (
        "已激活" if (gpa is not None and language_score is not None) else "跳过"
    )
    reporter.set_stage(
        0.36,
        0.48,
        PIPELINE_MESSAGES["extract_profile"].format(
            gpa=_fmt_value(gpa),
            lang=_fmt_value(language_score),
            uni=_fmt_value(background_university),
            status=probability_calibration_status,
        ),
    )

    probability_adjuster = None
    if gpa is not None and language_score is not None:
        probability_adjuster = ProbabilityAdjuster(
            cases_df if cases_df is not None else pd.DataFrame()
        )
    reporter.advance_ratio(0.30)

    uni_pool_count = len(all_universities_target)
    major_pool_count = len(all_majors_target)
    target_total = uni_pool_count * major_pool_count
    msg_running_pred = PIPELINE_MESSAGES["running_calc"].format(total=target_total)
    reporter.set_stage(
        0.48,
        0.62,
        msg_running_pred,
    )
    with BackgroundAnimator(reporter, msg_running_pred):
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
            language_type=cleaned_input.get("language_type"),
            background_university=background_university,
            progress_reporter=reporter,
        )

    if meta and meta.get("error"):
        prediction_handler_logger.info(f"预测未生成有效结果: {meta.get('error')}")
        return PredictionResultModel(meta=meta)

    initial_count = len(sim_results) + len(cross_results) + len(user_specified_results or [])
    reporter.set_stage(0.62, 0.72, PIPELINE_MESSAGES["initial_filter"].format(count=initial_count))

    experience_details = cleaned_input.get("experience_details", {})
    exp_text_len = sum(len(str(v)) for v in experience_details.values() if v)
    reporter.set_stage(0.72, 0.80, PIPELINE_MESSAGES["analyze_text"].format(length=exp_text_len))

    admitted_combos = get_admitted_combinations_from_dataframe(cases_df, background_major)
    
    all_res = sim_results + cross_results + (user_specified_results or [])
    new_major_cache = {}
    for r in all_res:
        u, m = r.get("university"), r.get("major")
        if u and m and (u, m) not in new_major_cache:
            new_major_cache[(u, m)] = is_new_major(u, m)

    adj_ctx = AdjustmentContext(
        gpa=gpa,
        language_score=language_score,
        background_university=background_university,
        background_major=background_major,
        background_faculty=get_background_faculty(background_major, cases_df),
        internship_count=cleaned_input.get("internship_count", 0),
        user_specified_majors=cleaned_input.get("target_majors", []),
        experience_details=experience_details,
        cases_df=cases_df,
        admitted_combinations=admitted_combos,
        is_new_major_cache=new_major_cache,
    )

    pipeline = ProbabilityAdjustmentPipeline(
        probability_adjuster=probability_adjuster,
        text_boost_provider=get_text_boost_provider(DEFAULT_TEXT_BOOST_CONFIG) if input_data.get("_has_valid_experience") else None,
    )

    sim_results = pipeline.adjust_batch(sim_results, adj_ctx, progress_reporter=reporter)
    cross_results = pipeline.adjust_batch(cross_results, adj_ctx, progress_reporter=reporter)
    if user_specified_results:
        user_specified_results = pipeline.adjust_batch(user_specified_results, adj_ctx, progress_reporter=reporter)

    msg_merging = PIPELINE_MESSAGES["merging"]
    reporter.set_stage(0.88, 1.0, msg_merging)
    with BackgroundAnimator(reporter, msg_merging):
        unique_results = combine_and_deduplicate_results(
            sim_results, cross_results, user_specified_results
        )

    if not unique_results:
        if meta is None:
            meta = {}
        meta["error"] = "empty_results"
        meta.setdefault("user_message", PIPELINE_MESSAGES["empty_results"])
        prediction_handler_logger.info("预测结果为空")
        reporter.force_progress(1.0, "分析结束")
        return PredictionResultModel(meta=meta)

    result_count = len(unique_results)
    reporter.force_progress(1.0, PIPELINE_MESSAGES["done"].format(count=result_count))

    return PredictionResultModel(
        similarity_results=sim_results,
        cross_major_results=cross_results,
        user_specified_results=user_specified_results,
        unified_results=unique_results,
        meta=meta,
    )


def _prepare_list_args(input_data: dict[str, Any]) -> tuple[list[str], list[str]]:
    all_universities_target_raw = input_data.get("_all_universities_target")
    all_universities_target: list[str] = (
        list(map(str, all_universities_target_raw))
        if isinstance(all_universities_target_raw, list)
        else []
    )
    all_majors_target_raw = input_data.get("_all_majors_target")
    all_majors_target: list[str] = (
        list(map(str, all_majors_target_raw)) if isinstance(all_majors_target_raw, list) else []
    )
    return all_universities_target, all_majors_target


@st.cache_data(ttl=600, show_spinner=False)
def run_prediction_pipeline(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    all_universities_fingerprint: tuple[int, int],
    all_majors_fingerprint: tuple[int, int],
) -> PredictionResultModel:
    all_universities_target, all_majors_target = _prepare_list_args(input_data)
    reporter = ProgressReporter(None)

    return _execute_prediction_pipeline(
        input_data=input_data,
        model_name=model_name,
        cases_df_fingerprint=cases_df_fingerprint,
        loaded_feature_names=loaded_feature_names,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        reporter=reporter,
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
    all_universities_target, all_majors_target = _prepare_list_args(input_data)
    reporter = ProgressReporter(progress_cb)

    return _execute_prediction_pipeline(
        input_data=input_data,
        model_name=model_name,
        cases_df_fingerprint=cases_df_fingerprint,
        loaded_feature_names=loaded_feature_names,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        reporter=reporter,
    )
