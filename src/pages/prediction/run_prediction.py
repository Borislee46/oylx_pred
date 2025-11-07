from typing import Any, Optional, Tuple, cast

import pandas as pd

from src.pages.prediction.prediction_execution import PredictionExecutor
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
from src.utils.session_manager import SessionManager

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
) -> Tuple[
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    Optional[list[dict[str, float | str]]],
    None,
]:
    session_manager = SessionManager()

    try:
        prediction_input: PredictionInput = {
            "background_university": str(current_input_data.get("background_university", "")),
            "background_major": str(current_input_data.get("background_major", "")),
            "target_universities": (
                current_input_data["target_universities"]
                if isinstance(current_input_data.get("target_universities"), list)
                else []
            ),
            "target_majors": (
                current_input_data["target_majors"]
                if isinstance(current_input_data.get("target_majors"), list)
                else []
            ),
        }

        if "gpa" in current_input_data and isinstance(current_input_data["gpa"], (int, float)):
            prediction_input["gpa"] = float(current_input_data["gpa"])
        if "language_score" in current_input_data and isinstance(
            current_input_data["language_score"], (int, float)
        ):
            prediction_input["language_score"] = float(current_input_data["language_score"])
        if "internship_count" in current_input_data and isinstance(
            current_input_data["internship_count"], (int, float)
        ):
            prediction_input["internship_count"] = int(current_input_data["internship_count"])
        if "research_count" in current_input_data and isinstance(
            current_input_data["research_count"], (int, float)
        ):
            prediction_input["research_count"] = int(current_input_data["research_count"])
        if "award_count" in current_input_data and isinstance(
            current_input_data["award_count"], (int, float)
        ):
            prediction_input["award_count"] = int(current_input_data["award_count"])
        if "paper_count" in current_input_data and isinstance(
            current_input_data["paper_count"], (int, float)
        ):
            prediction_input["paper_count"] = int(current_input_data["paper_count"])
        if "school_level" in current_input_data and isinstance(
            current_input_data["school_level"], (int, float)
        ):
            prediction_input["school_level"] = int(current_input_data["school_level"])
        if "experience_details" in current_input_data and isinstance(
            current_input_data["experience_details"], dict
        ):
            prediction_input["experience_details"] = cast(
                dict[str, str], current_input_data["experience_details"]
            )

        combinations, meta = generate_prediction_combinations(
            prediction_input, all_universities_target, all_majors_target
        )

        session_manager.set(**meta)

        if not combinations:
            prediction_runner_logger.warning("有效组合为空：请检查候选池或筛选条件。")
            return [], [], None, None

        model_input_features, missing_inputs = prepare_model_inputs(
            current_input_data, expected_features
        )
        if missing_inputs or prediction_model is None:
            return [], [], None, None

        executor = PredictionExecutor(len(combinations))
        all_prediction_outputs = executor.execute_parallel(
            prediction_model, combinations, model_input_features, expected_features
        )

        all_prediction_outputs.sort(
            key=lambda x: (
                -float(x.get("probability", 0.0) or 0.0),
                str(x.get("university", "")),
                str(x.get("major", "")),
            )
        )

        user_specified_combinations = get_user_specified_combinations(
            current_input_data, all_universities_target, session_manager
        )

        cross_faculty_confirmed = session_manager.get("cross_faculty_confirmed", False)
        faculty_value = current_input_data.get("faculty")
        background_faculty = (
            None
            if cross_faculty_confirmed
            else (faculty_value if isinstance(faculty_value, str) else None)
        )

        background_major_value = current_input_data.get("background_major", "")
        background_major = background_major_value if isinstance(background_major_value, str) else ""

        results = process_prediction_results(
            results=all_prediction_outputs,
            background_major=background_major,
            bg_target_similarity_cache=bg_target_similarity_cache,
            num_target_universities=num_target_universities,
            cases_df=cases_df if cases_df is not None else pd.DataFrame(),
            user_specified_combinations=user_specified_combinations,
            background_faculty=background_faculty,
        )

        return (*results, None)

    except Exception as e:
        prediction_runner_logger.error(f"执行单个预测时发生错误: {e}", exc_info=True)
        return [], [], None, None
