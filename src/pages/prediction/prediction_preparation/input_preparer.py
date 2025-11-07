from typing import Any, Tuple

from src.utils.logger import setup_logger

prediction_runner_logger = setup_logger("page3", "prediction")


def prepare_model_inputs(
    current_input_data: dict[str, Any],
    expected_features: list[str],
) -> Tuple[dict[str, float | int | str], list[str]]:
    base_expected_features = [
        f for f in expected_features if f not in ["target_university", "target_major"]
    ]

    model_input_features: dict[str, float | int | str] = {}
    for feature in base_expected_features:
        if feature in current_input_data:
            value = current_input_data[feature]
            if isinstance(value, (float, int, str)):
                model_input_features[feature] = value

    missing_inputs = [
        f for f in base_expected_features if f not in model_input_features
    ]
    if missing_inputs:
        prediction_runner_logger.error(f"缺少必要的输入特征: {missing_inputs}")

    return model_input_features, missing_inputs

