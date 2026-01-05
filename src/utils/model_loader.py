import glob
import json
import os
from typing import Any

import numpy as np
import streamlit as st
import xgboost as xgb

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def _load_serialized_xgb(file_path: str) -> xgb.XGBClassifier | None:
    model = xgb.XGBClassifier()
    model.load_model(file_path)
    return model


class _CalibratedPredictor:
    def __init__(self, base_model: Any, calibration: dict[str, Any]):
        self.base_model = base_model
        self.calibration = calibration

    def get_xgb_params(self):
        if hasattr(self.base_model, "get_xgb_params"):
            return self.base_model.get_xgb_params()
        return {}

    def get_booster(self):
        if hasattr(self.base_model, "get_booster"):
            return self.base_model.get_booster()
        return None

    def predict_proba(self, X):
        base_proba = self.base_model.predict_proba(X)
        if base_proba is None or len(base_proba.shape) != 2 or base_proba.shape[1] < 2:
            return base_proba

        p1 = base_proba[:, 1]
        method = self.calibration.get("method")
        params = self.calibration.get("params", {})

        if method == "sigmoid":
            a, b = float(params.get("a")), float(params.get("b"))
            calibrated_p1 = 1.0 / (1.0 + np.exp(a * p1 + b))
        else:
            return base_proba

        return np.vstack([1.0 - calibrated_p1, calibrated_p1]).T

    def predict(self, X, threshold: float = 0.24):
        probas = self.predict_proba(X)
        if probas is None:
            return None
        return (probas[:, 1] >= threshold).astype(int)


def _wrap_with_calibration(model: Any, calibration: dict[str, Any]) -> Any:
    return _CalibratedPredictor(model, calibration)


def load_model_dependencies(
    model_dir: str, model_prefix: str
) -> tuple[Any | None, list[str] | None, dict[str, str] | None]:
    abs_model_dir = os.path.abspath(model_dir)

    model_glob_pattern = f"{model_prefix}_????????_??????.ubj"
    model_search_path = os.path.join(abs_model_dir, model_glob_pattern)
    model_files = glob.glob(model_search_path)

    if not model_files:
        logger.warning(f"未找到 {model_prefix} 模型")
        return None, None, None

    latest_model_path = max(model_files, key=os.path.getmtime)
    xgb_model = _load_serialized_xgb(latest_model_path)
    if xgb_model is None:
        return None, None, None

    booster = xgb_model.get_booster()

    feature_names: list[str] | None = None
    feature_names_str = booster.attr("feature_names")
    if feature_names_str:
        feature_names = json.loads(feature_names_str)

    calibration: dict[str, Any] | None = None
    calibration_str = booster.attr("calibration_params")
    if calibration_str:
        calibration = json.loads(calibration_str)

    level_fallback_mapping: dict[str, str] | None = None
    fallback_str = booster.attr("level_fallback_mapping")
    if fallback_str:
        level_fallback_mapping = json.loads(fallback_str)

    final_model = (
        _wrap_with_calibration(xgb_model, calibration) if calibration is not None else xgb_model
    )

    return final_model, feature_names, level_fallback_mapping


@st.cache_resource(show_spinner=False)
def load_model(
    model_name: str = "xgboost",
) -> tuple[Any | None, list[str] | None, dict[str, str] | None]:
    model, feature_names, level_fallback_mapping = load_model_dependencies(
        "src/machine_learning_models/pre-trained_models", model_name
    )

    return model, feature_names, level_fallback_mapping
