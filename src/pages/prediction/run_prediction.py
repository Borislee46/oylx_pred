from typing import Any

import pandas as pd

from src.pages.prediction.prediction_execution import PredictionExecutor
from src.pages.prediction.prediction_input_validator import validate_and_clean_input
from src.pages.prediction.prediction_model import PredictionModel
from src.pages.prediction.prediction_preparation import (
    get_user_specified_combinations,
    prepare_model_inputs,
)
from src.pages.prediction.prediction_processor import (
    generate_prediction_combinations,
    process_prediction_results,
)
from src.pages.prediction.prediction_types import PredictionInput
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
    background_university: str | None = None,
) -> tuple[
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    list[dict[str, float | str]] | None,
    dict[str, Any] | None,
]:
    prediction_input: PredictionInput = validate_and_clean_input(current_input_data)

    combinations, meta = generate_prediction_combinations(
        prediction_input, all_universities_target, all_majors_target
    )

    if not combinations:
        prediction_runner_logger.warning("有效组合为空：请检查候选池或筛选条件。")
        if meta is None:
            meta = {}
        meta["error"] = "no_valid_combinations"
        return [], [], None, meta

    model_input_features, missing_inputs = prepare_model_inputs(
        current_input_data, expected_features
    )
    if missing_inputs or prediction_model is None:
        if meta is None:
            meta = {}
        if prediction_model is None:
            meta["error"] = "model_unavailable"
        else:
            meta["error"] = "missing_features"
            meta["missing_features"] = missing_inputs
        return [], [], None, meta

    executor = PredictionExecutor(len(combinations))
    all_prediction_outputs = executor.execute_parallel(
        prediction_model, combinations, model_input_features, expected_features
    )
    if not all_prediction_outputs:
        if meta is None:
            meta = {}
        meta["error"] = "execution_failed"
        return [], [], None, meta

    all_prediction_outputs.sort(
        key=lambda x: (
            -float(x.get("probability", 0.0) or 0.0),
            str(x.get("university", "")),
            str(x.get("major", "")),
        )
    )

    user_specified_combinations = get_user_specified_combinations(
        current_input_data, all_universities_target
    )

    faculty_value = current_input_data.get("faculty")
    background_faculty = (
        None
        if cross_faculty_confirmed
        else (faculty_value if isinstance(faculty_value, str) else None)
    )

    background_major = prediction_input.get("background_major", "")

    background_major_original_value = current_input_data.get("background_major_original", "")
    background_major_original = (
        str(background_major_original_value)
        if background_major_original_value
        else background_major
    )

    results = process_prediction_results(
        results=all_prediction_outputs,
        background_major=background_major,
        background_major_original=background_major_original,
        bg_target_similarity_cache=bg_target_similarity_cache,
        num_target_universities=num_target_universities,
        cases_df=cases_df if cases_df is not None else pd.DataFrame(),
        user_specified_combinations=user_specified_combinations,
        background_faculty=background_faculty,
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        background_university=background_university,
    )

    return (*results, meta)
