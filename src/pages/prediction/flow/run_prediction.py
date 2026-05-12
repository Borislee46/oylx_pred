from typing import Any

import pandas as pd

from src.pages.prediction.config.ui_messages import (
    PIPELINE_PHASE_MAP,
    format_pipeline_compute_progress,
)
from src.pages.prediction.core.types import PredictionInput
from src.pages.prediction.core.utils import get_background_faculty
from src.pages.prediction.flow.processor import (
    generate_prediction_combinations,
    process_prediction_results,
)
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.modeling.model import PredictionModel
from src.pages.prediction.page_data_loader import machine_learning_model
from src.pages.prediction.prediction_execution import PredictionExecutor
from src.pages.prediction.prediction_preparation import (
    get_user_specified_combinations,
    prepare_model_inputs,
)
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
) -> tuple[
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    list[dict[str, float | str]] | None,
    dict[str, Any] | None,
]:
    prediction_input: PredictionInput = current_input_data

    bg_major = prediction_input.get("background_major", "")
    bg_major_orig = str(current_input_data.get("background_major_original") or bg_major)

    if cached_combinations:
        combinations = cached_combinations
        meta = {
            "combination_count": len(combinations),
            "cached": True,
            "progress_hints": {
                "target_unis": list({u for u, _ in combinations}),
                "target_majors": list({m for _, m in combinations}),
                "user_locked_majors": True,
            },
        }
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
    if missing_inputs or prediction_model is None:
        meta.update(
            {
                "error": "model_unavailable" if prediction_model is None else "missing_features",
                "missing_features": missing_inputs,
            }
        )
        return [], [], None, meta

    all_prediction_outputs = PredictionExecutor(len(combinations)).execute_parallel(
        prediction_model, combinations, model_input_features, expected_features
    )

    if not all_prediction_outputs:
        meta["error"] = "execution_failed"
        return [], [], None, meta

    user_specified_combinations = get_user_specified_combinations(
        current_input_data, all_universities_target
    )

    bg_faculty = background_faculty or current_input_data.get("faculty")
    if bg_faculty is None:
        bg_faculty = get_background_faculty(bg_major, cases_df)

    if page_state is None:
        page_state = machine_learning_model.resource_loader()

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
    )

    return (*results, meta)
