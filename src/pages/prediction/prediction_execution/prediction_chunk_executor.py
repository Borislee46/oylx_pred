from src.pages.prediction.modeling.model import PredictionModel
from src.pages.prediction.prediction_execution.worker_model_manager import (
    get_worker_model,
)
from src.utils.logger import setup_logger

prediction_runner_logger = setup_logger("page3", "prediction")


def run_prediction_chunk(
    prediction_model: PredictionModel | None,
    model_input_features: dict[str, float | int | str],
    combinations_chunk: list[tuple[str, str]],
    expected_features: list[str],
) -> list[dict[str, float | str]]:
    if not combinations_chunk:
        return []
    if prediction_model is None:
        raise RuntimeError("prediction_model is None")
    return prediction_model.predict_batch(
        model_input_features, combinations_chunk, expected_features
    )


def run_prediction_chunk_in_process(
    model_input_features: dict[str, float | int | str],
    combinations_chunk: list[tuple[str, str]],
    expected_features: list[str],
) -> list[dict[str, float | str]]:
    model = get_worker_model()
    if model is None:
        raise RuntimeError("worker model not initialized")
    return run_prediction_chunk(
        model,
        model_input_features,
        combinations_chunk,
        expected_features,
    )
