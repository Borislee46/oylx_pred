import time
from typing import Any

import pandas as pd

from src.pages.prediction.core.types import PredictionInput
from src.pages.prediction.core.ui_messages import (
    PIPELINE_PHASE_MAP,
    format_pipeline_compute_progress,
)
from src.pages.prediction.core.utils import get_background_faculty
from src.pages.prediction.flow import PredictionExecutor
from src.pages.prediction.flow.preparer import (
    get_user_specified_combinations,
    prepare_model_inputs,
)
from src.pages.prediction.flow.processor import (
    generate_prediction_combinations,
    process_prediction_results,
)
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.modeling import PredictionModel
from src.pages.prediction.page_data_loader import machine_learning_model
from src.utils.logger import setup_logger

prediction_runner_logger = setup_logger("page3", "prediction")


def run_single_prediction(
    current_input_data: dict[str, Any],
    prediction_model: PredictionModel,
    cases_df: pd.DataFrame,
    bg_target_similarity_cache: dict[str, float],
    expected_features: list[str],
    all_universities_target: list[str],
    all_majors_target: list[str],
    num_target_universities: int,
    cross_faculty_confirmed: bool = False,
    probability_adjuster: Any | None = None,
    gpa: float | None = None,
    language_score: float | None = None,
    language_type: str | None = None,
    background_university: str | None = None,
    progress_reporter: ProgressReporter | None = None,
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
    page_state: machine_learning_model | None = None,
    cached_combinations: list[tuple[str, str]] | None = None,
    enable_language_requirement_penalty: bool = True,
) -> tuple[
    list[dict[str, float | str]],  # similarity_results
    list[dict[str, float | str]],  # cross_major_results
    list[dict[str, float | str]] | None,  # user_specified_results
    dict[str, Any] | None,  # meta
]:
    t_start = time.monotonic()
    prediction_input: PredictionInput = current_input_data

    bg_major = prediction_input.get("background_major", "")
    bg_major_orig = str(current_input_data.get("background_major_original") or bg_major)

    prediction_runner_logger.info(
        "单次预测开始 | bg_uni=%s bg_major=%s cache=%s",
        current_input_data.get("background_university", "")[:30],
        bg_major[:30],
        bool(cached_combinations),
    )

    t1 = time.monotonic()
    if cached_combinations:
        combinations = cached_combinations
        meta = {
            "combination_count": len(combinations),
            "cached": True,
            "progress_hints": {
                "target_unis": list({u for u, _ in combinations}),
                "target_majors": list({m for _, m in combinations}),
                "user_locked_majors": bool(prediction_input.get("target_majors")),
            },
        }
        prediction_runner_logger.info(
            "Step1 候选组合(缓存复用) | combinations=%d elapsed=%.3fs",
            len(combinations),
            time.monotonic() - t1,
        )
    else:
        combinations, meta = generate_prediction_combinations(
            input_data=prediction_input,
            all_universities_target=all_universities_target,
            all_majors_target=all_majors_target,
            bg_target_similarity_cache=bg_target_similarity_cache,
            background_major_original=bg_major_orig,
        )

    meta = meta or {}
    if not combinations:
        prediction_runner_logger.warning("有效组合为空：请检查候选池或筛选条件。")
        meta["error"] = "no_valid_combinations"
        return [], [], None, meta

    if progress_reporter is not None:
        hints = meta.get("progress_hints") or {}
        progress_reporter.emit(
            format_pipeline_compute_progress(combinations, hints),
            force=True,
            phase=PIPELINE_PHASE_MAP["running_calc"],
        )

    model_input_features, missing_inputs = prepare_model_inputs(
        current_input_data, expected_features
    )

    if prediction_model is None:
        meta["error"] = "model_unavailable"
        return [], [], None, meta

    if missing_inputs:
        meta["fallback_eligible"] = True
        meta["missing_features"] = missing_inputs
        meta["_fallback_combinations"] = combinations
        prediction_runner_logger.info(
            "触发 Fallback 路径 | missing=%s combinations=%d elapsed=%.3fs",
            missing_inputs,
            len(combinations),
            time.monotonic() - t_start,
        )
        return [], [], None, meta

    t3 = time.monotonic()
    all_prediction_outputs = PredictionExecutor(len(combinations)).execute_parallel(
        prediction_model, combinations, model_input_features, expected_features
    )
    t3_elapsed = time.monotonic() - t3

    if not all_prediction_outputs:
        meta["error"] = "execution_failed"
        prediction_runner_logger.error("XGBoost 推理失败：无输出")
        return [], [], None, meta

    user_specified_combinations = get_user_specified_combinations(
        current_input_data, all_universities_target
    )

    bg_faculty = background_faculty or current_input_data.get("faculty")
    if bg_faculty is None:
        bg_faculty = get_background_faculty(bg_major, cases_df)

    if page_state is None:
        page_state = machine_learning_model.resource_loader()

    t5 = time.monotonic()
    results = process_prediction_results(
        results=all_prediction_outputs,
        background_major=bg_major,
        background_major_original=bg_major_orig,
        bg_target_similarity_cache=bg_target_similarity_cache,
        num_target_universities=num_target_universities,
        cases_df=cases_df,
        user_specified_combinations=user_specified_combinations,
        background_faculty=bg_faculty if isinstance(bg_faculty, str) else None,
        allow_degraded_user_specified=cross_faculty_confirmed,
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        language_type=language_type or prediction_input.get("language_type"),
        background_university=background_university,
        progress_reporter=progress_reporter,
        agent=None,
        admitted_combinations=admitted_combinations,
        enable_language_requirement_penalty=enable_language_requirement_penalty,
    )
    t5_elapsed = time.monotonic() - t5

    total_elapsed = time.monotonic() - t_start
    sim_count = len(results[0])
    cross_count = len(results[1])
    usr_count = len(results[2]) if results[2] else 0

    prediction_runner_logger.info(
        "单次预测完成 | combinations=%d sim=%d cross=%d user=%d "
        "infer=%.3fs postprocess=%.3fs total=%.3fs",
        len(combinations),
        sim_count,
        cross_count,
        usr_count,
        t3_elapsed,
        t5_elapsed,
        total_elapsed,
    )

    return (*results, meta)
