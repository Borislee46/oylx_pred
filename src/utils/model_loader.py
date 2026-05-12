import glob
import json
import os
from typing import Any

import numpy as np
import streamlit as st
import xgboost as xgb

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class _BoosterClassifierWrapper:
    def __init__(self, booster: xgb.Booster):
        self._booster = booster

    def get_booster(self) -> xgb.Booster:
        return self._booster

    def predict_proba(self, X):
        d = xgb.DMatrix(X, enable_categorical=True)
        p1 = self._booster.predict(d)
        return np.vstack([1.0 - p1, p1]).T


def _load_serialized_xgb(file_path: str) -> _BoosterClassifierWrapper | None:
    booster = xgb.Booster()
    booster.load_model(file_path)
    return _BoosterClassifierWrapper(booster)


def _safe_json_loads(raw_value: str | None, field_name: str) -> Any:
    if not raw_value:
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        logger.warning(f"模型元数据字段解析失败: {field_name}")
        return None


def _normalize_prediction_threshold(value: Any, default: float = 0.24) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return default
    return threshold if 0.0 <= threshold <= 1.0 else default


class _CalibratedPredictor:
    def __init__(
        self,
        base_model: Any,
        calibration: dict[str, Any],
        prediction_threshold: float = 0.24,
    ):
        self.base_model = base_model
        self.calibration = calibration
        self.prediction_threshold = prediction_threshold

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
            a = params.get("a")
            b = params.get("b")
            if a is None or b is None:
                logger.warning("sigmoid 校准参数缺失，回退基础概率")
                return base_proba
            a = float(a)
            b = float(b)
            calibrated_p1 = 1.0 / (1.0 + np.exp(a * p1 + b))
        elif method == "isotonic":
            x_thresholds = params.get("x_thresholds")
            y_thresholds = params.get("y_thresholds")
            if not x_thresholds or not y_thresholds:
                logger.warning("isotonic 校准参数缺失，回退基础概率")
                return base_proba
            calibrated_p1 = np.interp(p1, x_thresholds, y_thresholds)
        else:
            return base_proba

        return np.vstack([1.0 - calibrated_p1, calibrated_p1]).T

    def predict(self, X, threshold: float | None = None):
        probas = self.predict_proba(X)
        if probas is None:
            return None
        decision_threshold = self.prediction_threshold if threshold is None else threshold
        return (probas[:, 1] >= decision_threshold).astype(int)


def _wrap_with_calibration(
    model: Any, calibration: dict[str, Any], prediction_threshold: float
) -> Any:
    return _CalibratedPredictor(model, calibration, prediction_threshold=prediction_threshold)


def load_model_dependencies(
    model_dir: str, model_prefix: str, model_path: str | None = None
) -> tuple[Any | None, list[str] | None, dict[str, str] | None, dict[str, Any] | None, float]:
    abs_model_dir = os.path.abspath(model_dir)

    selected_model_path = model_path
    if selected_model_path is None:
        model_glob_pattern = f"{model_prefix}_????????_??????.ubj"
        model_search_path = os.path.join(abs_model_dir, model_glob_pattern)
        model_files = glob.glob(model_search_path)

        if not model_files:
            logger.warning(f"未找到 {model_prefix} 模型")
            return None, None, None, None, 0.24

        selected_model_path = max(model_files, key=os.path.getmtime)

    xgb_model = _load_serialized_xgb(selected_model_path)
    if xgb_model is None:
        return None, None, None, None, 0.24

    booster = xgb_model.get_booster()
    metadata = _safe_json_loads(booster.attr("model_metadata"), "model_metadata") or {}

    feature_names: list[str] | None = None
    raw_feature_names = booster.attr("feature_names")
    feature_names = _safe_json_loads(raw_feature_names, "feature_names")
    if feature_names is None:
        feature_names = metadata.get("feature_names")
    if feature_names is not None and not isinstance(feature_names, list):
        feature_names = None

    calibration: dict[str, Any] | None = None
    calibration = _safe_json_loads(booster.attr("calibration_params"), "calibration_params")
    if calibration is None:
        calibration = metadata.get("calibration_params")
    if calibration is not None and not isinstance(calibration, dict):
        calibration = None

    level_fallback_mapping: dict[str, str] | None = None
    level_fallback_mapping = _safe_json_loads(
        booster.attr("level_fallback_mapping"), "level_fallback_mapping"
    )
    if level_fallback_mapping is None:
        level_fallback_mapping = metadata.get("level_fallback_mapping")
    if level_fallback_mapping is not None and not isinstance(level_fallback_mapping, dict):
        level_fallback_mapping = None

    feature_engineer_state = _safe_json_loads(
        booster.attr("feature_engineer_state"), "feature_engineer_state"
    )
    if feature_engineer_state is None:
        feature_engineer_state = metadata.get("feature_engineer_state")
    if feature_engineer_state is not None and not isinstance(feature_engineer_state, dict):
        feature_engineer_state = None

    prediction_threshold = _normalize_prediction_threshold(
        _safe_json_loads(booster.attr("prediction_threshold"), "prediction_threshold"),
        default=_normalize_prediction_threshold(metadata.get("prediction_threshold")),
    )

    final_model = (
        _wrap_with_calibration(xgb_model, calibration, prediction_threshold)
        if calibration is not None
        else xgb_model
    )

    if not hasattr(final_model, "prediction_threshold"):
        final_model.prediction_threshold = prediction_threshold

    return (
        final_model,
        feature_names,
        level_fallback_mapping,
        feature_engineer_state,
        prediction_threshold,
    )


@st.cache_resource(show_spinner=False)
def load_model(
    model_name: str = "xgboost",
) -> tuple[Any | None, list[str] | None, dict[str, str] | None, dict[str, Any] | None, float]:
    model, feature_names, level_fallback_mapping, feature_engineer_state, prediction_threshold = (
        load_model_dependencies("src/machine_learning_models/pre-trained_models", model_name)
    )

    return (
        model,
        feature_names,
        level_fallback_mapping,
        feature_engineer_state,
        prediction_threshold,
    )
