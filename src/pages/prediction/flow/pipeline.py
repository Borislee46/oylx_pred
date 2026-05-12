import random
from collections.abc import Callable
from typing import Any

import pandas as pd

from src.pages.prediction.config.ui_messages import (
    PIPELINE_MESSAGES,
    PIPELINE_PHASE_MAP,
    format_pipeline_done_progress,
    format_pipeline_empty_progress,
    format_pipeline_prep_progress,
    format_pipeline_refine_progress,
)
from src.pages.prediction.core.utils import get_background_faculty, is_new_major
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.flow.run_prediction import run_single_prediction
from src.pages.prediction.page_data_loader import (
    cached_get_prediction_model,
    machine_learning_model,
)
from src.pages.prediction.prediction_preparation import validate_and_clean_input
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
from src.utils.app_data_loader import load_bg_target_similarity_cache
from src.utils.logger import setup_logger
from src.utils.session_manager import PredictionResultModel

prediction_handler_logger = setup_logger("page3", "prediction")

ProgressCallback = Callable[[str], None]


def _execute_prediction_pipeline(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    all_universities_target: list[str],
    all_majors_target: list[str],
    reporter: ProgressReporter,
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
    page_state: machine_learning_model | None = None,
    cached_combinations: list[tuple[str, str]] | None = None,
) -> PredictionResultModel:
    prediction_model = cached_get_prediction_model(model_name)

    if prediction_model is None:
        return PredictionResultModel(meta={"error": "model_load_failed"})

    if page_state is None:
        page_state = machine_learning_model.resource_loader()
    cases_df = page_state.cases_df

    if cases_df_fingerprint != page_state.cases_df_fingerprint:
        prediction_handler_logger.warning(
            f"案例数据指纹不匹配: 期望 {cases_df_fingerprint}, 实际 {page_state.cases_df_fingerprint}"
        )
        return PredictionResultModel(
            meta={
                "error": "cases_df_fingerprint_mismatch",
                "expected_cases_df_fingerprint": cases_df_fingerprint,
                "actual_cases_df_fingerprint": page_state.cases_df_fingerprint,
            }
        )

    cleaned_input = validate_and_clean_input(input_data)
    current_input_data = {**input_data, **cleaned_input}

    bg_target_similarity_cache = load_bg_target_similarity_cache()

    reporter.emit(
        format_pipeline_prep_progress(
            bg_university=cleaned_input.get("background_university"),
            bg_major=cleaned_input.get("background_major"),
            language_type=cleaned_input.get("language_type"),
            language_score=cleaned_input.get("language_score"),
            target_universities=cleaned_input.get("target_universities"),
            similarity_cache_loaded=bool(bg_target_similarity_cache),
        ),
        force=True,
        phase=PIPELINE_PHASE_MAP["init_engine"],
    )

    num_target_universities = len(cleaned_input.get("target_universities", []))
    cross_faculty_confirmed = input_data.get("_cross_faculty_confirmed", False)

    gpa = cleaned_input.get("gpa")
    language_score = cleaned_input.get("language_score")
    background_university = cleaned_input.get("background_university")
    background_major = cleaned_input.get("background_major", "")

    probability_adjuster = (
        ProbabilityAdjuster(
            cases_df if cases_df is not None else pd.DataFrame(),
            data_hash=cases_df_fingerprint,
        )
        if gpa is not None and language_score is not None
        else None
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
        language_type=cleaned_input.get("language_type"),
        background_university=background_university,
        progress_reporter=reporter,
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
        page_state=page_state,
        cached_combinations=cached_combinations,
    )

    if meta and meta.get("error"):
        prediction_handler_logger.info(f"预测未生成有效结果: {meta.get('error')}")
        return PredictionResultModel(meta=meta)

    admitted_combos = (
        admitted_combinations
        if admitted_combinations is not None
        else get_admitted_combinations_from_dataframe(cases_df, background_major)
    )

    all_res = sim_results + cross_results + (user_specified_results or [])
    new_major_cache = {
        (r.get("university"), r.get("major")): is_new_major(r.get("university"), r.get("major"))
        for r in all_res
        if r.get("university") and r.get("major")
    }

    bg_faculty = (
        background_faculty
        if background_faculty
        else (
            current_input_data.get("faculty") or get_background_faculty(background_major, cases_df)
        )
    )

    route_labels: list[str] = []
    if sim_results:
        route_labels.append("相似专业")
    if cross_results:
        route_labels.append("跨专业")
    if user_specified_results:
        route_labels.append("用户指定")

    reporter.emit(
        format_pipeline_refine_progress(
            route_labels=route_labels,
            bg_faculty=bg_faculty if isinstance(bg_faculty, str) else None,
            soft_background_on=bool(input_data.get("_has_valid_experience")),
        ),
        force=True,
        phase=PIPELINE_PHASE_MAP["initial_filter"],
    )

    adj_ctx = AdjustmentContext(
        gpa=gpa,
        language_score=language_score,
        background_university=background_university,
        background_major=background_major,
        background_faculty=bg_faculty,
        internship_count=cleaned_input.get("internship_count", 0),
        user_specified_majors=cleaned_input.get("target_majors", []),
        experience_details=cleaned_input.get("experience_details", {}),
        cases_df=cases_df,
        admitted_combinations=admitted_combos,
        is_new_major_cache=new_major_cache,
    )

    pipeline = ProbabilityAdjustmentPipeline(
        probability_adjuster=probability_adjuster,
        text_boost_provider=(
            get_text_boost_provider(DEFAULT_TEXT_BOOST_CONFIG)
            if input_data.get("_has_valid_experience")
            else None
        ),
    )

    sim_results = pipeline.adjust_batch(
        sim_results, adj_ctx, progress_reporter=reporter, batch_tag="相似专业"
    )
    cross_results = pipeline.adjust_batch(
        cross_results, adj_ctx, progress_reporter=reporter, batch_tag="跨专业"
    )
    if user_specified_results:
        user_specified_results = pipeline.adjust_batch(
            user_specified_results, adj_ctx, progress_reporter=reporter, batch_tag="用户指定"
        )

    unique_results = combine_and_deduplicate_results(
        sim_results, cross_results, user_specified_results
    )

    if not unique_results:
        meta = meta or {}
        meta["error"] = "empty_results"
        empty_msg = PIPELINE_MESSAGES["empty_results"]
        meta.setdefault(
            "user_message",
            random.choice(empty_msg) if isinstance(empty_msg, list) else empty_msg,
        )
        prediction_handler_logger.info("预测结果为空")
        reporter.emit(
            format_pipeline_empty_progress(
                bg_major=background_major,
                target_universities=cleaned_input.get("target_universities"),
            ),
            force=True,
            phase=PIPELINE_PHASE_MAP["empty_results"],
        )
        return PredictionResultModel(meta=meta)

    reporter.emit(
        format_pipeline_done_progress(),
        force=True,
        phase=PIPELINE_PHASE_MAP["done"],
    )

    return PredictionResultModel(
        similarity_results=sim_results,
        cross_major_results=cross_results,
        user_specified_results=user_specified_results,
        unified_results=unique_results,
        meta=meta,
    )


def _prepare_list_args(input_data: dict[str, Any]) -> tuple[list[str], list[str]]:
    def _to_list(key: str) -> list[str]:
        raw = input_data.get(key)
        return [str(x) for x in raw] if isinstance(raw, list) else []

    return _to_list("_all_universities_target"), _to_list("_all_majors_target")


def run_prediction_pipeline(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
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
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
    )


def run_prediction_pipeline_with_progress(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    *,
    progress_cb: ProgressCallback | None = None,
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
    page_state: machine_learning_model | None = None,
    cached_combinations: list[tuple[str, str]] | None = None,
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
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
        page_state=page_state,
        cached_combinations=cached_combinations,
    )
