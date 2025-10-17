import glob
import json
import os
import warnings
from typing import Any

import numpy as np
import streamlit as st
import xgboost as xgb

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

warnings.filterwarnings("ignore")


def _load_serialized_xgb(file_path: str) -> xgb.XGBClassifier | None:
    model = xgb.XGBClassifier()
    model.load_model(file_path)
    return model


class _CalibratedPredictor:
    def __init__(self, base_model: Any, calibration: dict[str, Any]):
        self.base_model = base_model
        self.calibration = calibration

    def predict_proba(self, X):
        base_proba = self.base_model.predict_proba(X)
        if base_proba is None or len(base_proba.shape) != 2 or base_proba.shape[1] < 2:
            return base_proba
        p1 = base_proba[:, 1]
        method = self.calibration.get("method")
        params = self.calibration.get("params", {})
        if method == "sigmoid":
            a = float(params.get("a"))
            b = float(params.get("b"))
            calibrated_p1 = 1.0 / (1.0 + np.exp(a * p1 + b))
        elif method == "isotonic":
            x_thr = params.get("x_thresholds", [])
            y_thr = params.get("y_thresholds", [])
            calibrated_p1 = np.interp(p1, x_thr, y_thr)
        else:
            logger.warning(f"未知校准方法: {method}")
        p0 = 1.0 - calibrated_p1
        return np.vstack([p0, calibrated_p1]).T


def _wrap_with_calibration(model: Any, calibration: dict[str, Any]) -> Any:
    return _CalibratedPredictor(model, calibration)


def _load_json_features(file_path: str) -> list[str] | None:
    with open(file_path, "r", encoding="utf-8") as f:
        features = json.load(f)
    if isinstance(features, list):
        return features
    return None


def _load_json_calibration(file_path: str) -> dict[str, Any] | None:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "method" in data and "params" in data:
        return data
    return None


def load_model_dependencies(
    model_dir: str, model_prefix: str
) -> tuple[Any | None, list[str] | None]:
    abs_model_dir = os.path.abspath(model_dir)

    model_glob_pattern = f"{model_prefix}_????????_??????.model"
    model_search_path = os.path.join(abs_model_dir, model_glob_pattern)
    model_files = glob.glob(model_search_path)

    if not model_files:
        logger.warning(f"未找到 {model_prefix} 模型")
        return None, None

    latest_model_path = max(model_files, key=os.path.getmtime)
    timestamp = (
        os.path.basename(latest_model_path).replace(f"{model_prefix}_", "").replace(".model", "")
    )

    xgb_model = _load_serialized_xgb(latest_model_path)
    if xgb_model is None:
        return None, None

    features_path = os.path.join(abs_model_dir, f"{model_prefix}_{timestamp}_features.json")
    feature_names = _load_json_features(features_path)

    calib_path = os.path.join(abs_model_dir, f"{model_prefix}_{timestamp}_calibration.json")
    calibration = _load_json_calibration(calib_path)

    if calibration is not None:
        return _wrap_with_calibration(xgb_model, calibration), feature_names

    return xgb_model, feature_names


@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "xgboost") -> tuple[Any | None, list[str] | None]:
    model, feature_names = load_model_dependencies(
        "src/machine_learning_models/pre-trained_models", model_name
    )

    return model, feature_names
