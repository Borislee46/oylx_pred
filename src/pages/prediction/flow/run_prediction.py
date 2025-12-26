from typing import Any

import pandas as pd

from src.pages.prediction.core.types import PredictionInput
from src.pages.prediction.flow.processor import (
    generate_prediction_combinations,
    process_prediction_results,
)
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.modeling.model import PredictionModel
from src.pages.prediction.prediction_execution import PredictionExecutor
from src.pages.prediction.prediction_preparation import (
    get_user_specified_combinations,
    prepare_model_inputs,
    validate_and_clean_input,
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
) -> tuple[
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    list[dict[str, float | str]] | None,
    dict[str, Any] | None,
]:
    prediction_input: PredictionInput = (
        current_input_data
        if "background_major" in current_input_data
        else validate_and_clean_input(current_input_data)
    )

    if language_type is None:
        language_type = prediction_input.get("language_type")

    background_major = prediction_input.get("background_major", "")
    background_major_original_value = current_input_data.get("background_major_original", "")
    background_major_original = (
        str(background_major_original_value)
        if background_major_original_value
        else background_major
    )

    combinations, meta = generate_prediction_combinations(
        input_data=prediction_input,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        bg_target_similarity_cache=bg_target_similarity_cache,
        background_major_original=background_major_original,
    )

    meta = meta or {}

    if not combinations:
        prediction_runner_logger.warning("有效组合为空：请检查候选池或筛选条件。")
        meta["error"] = "no_valid_combinations"
        return [], [], None, meta

    model_input_features, missing_inputs = prepare_model_inputs(
        current_input_data, expected_features
    )
    if missing_inputs or prediction_model is None:
        meta["error"] = "model_unavailable" if prediction_model is None else "missing_features"
        if missing_inputs:
            meta["missing_features"] = missing_inputs
        return [], [], None, meta

    executor = PredictionExecutor(len(combinations))
    all_prediction_outputs = executor.execute_parallel(
        prediction_model, combinations, model_input_features, expected_features
    )

    if not all_prediction_outputs:
        meta["error"] = "execution_failed"
        return [], [], None, meta

    user_specified_combinations = get_user_specified_combinations(
        current_input_data, all_universities_target
    )

    faculty_value = current_input_data.get("faculty")
    background_faculty = (
        None
        if cross_faculty_confirmed
        else (faculty_value if isinstance(faculty_value, str) else None)
    )

    results = process_prediction_results(
        results=all_prediction_outputs,
        background_major=background_major,
        background_major_original=background_major_original,
        bg_target_similarity_cache=bg_target_similarity_cache,
        num_target_universities=num_target_universities,
        cases_df=cases_df,
        user_specified_combinations=user_specified_combinations,
        background_faculty=background_faculty,
        allow_degraded_user_specified=cross_faculty_confirmed,
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        language_type=language_type,
        background_university=background_university,
        progress_reporter=progress_reporter,
    )

    return (*results, meta)
