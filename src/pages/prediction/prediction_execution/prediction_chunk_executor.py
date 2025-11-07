from typing import Optional

from src.pages.prediction.prediction_execution.worker_model_manager import (
    get_worker_model,
)
from src.pages.prediction.prediction_model import PredictionModel
from src.utils.logger import setup_logger

prediction_runner_logger = setup_logger("page3", "prediction")


def run_prediction_chunk(
    prediction_model: Optional[PredictionModel],
    model_input_features: dict[str, float | int | str],
    combinations_chunk: list[tuple[str, str]],
    expected_features: list[str],
) -> Optional[list[dict[str, float | str]]]:
    if not combinations_chunk or prediction_model is None:
        return None

    try:
        return prediction_model.predict_batch(
            model_input_features, combinations_chunk, expected_features
        )
    except Exception as e:
        prediction_runner_logger.error(f"预测失败: {e}", exc_info=True)
        return None


def run_prediction_chunk_in_process(
    model_input_features: dict[str, float | int | str],
    combinations_chunk: list[tuple[str, str]],
    expected_features: list[str],
) -> Optional[list[dict[str, float | str]]]:
    model = get_worker_model()
    return run_prediction_chunk(
        model,
        model_input_features,
        combinations_chunk,
        expected_features,
    )
