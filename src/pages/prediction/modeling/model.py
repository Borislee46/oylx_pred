from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.machine_learning_models.data_config import (
    COUNT_COLUMNS_FOR_LOG_TRANSFORM,
    DEFAULT_PREDICTION_THRESHOLD,
)
from src.utils.logger import setup_logger
from src.utils.model_loader import load_model
from src.utils.school_level_service import get_school_level_service

page_logger = setup_logger("page3", "prediction")


def validate_model_and_features(prediction_model: Optional["PredictionModel"]) -> list[str] | None:
    if prediction_model is None:
        st.error("关键配置错误：无法加载预测模型。")
        page_logger.critical("prediction_model 为 None")
        return None

    if not prediction_model.feature_names:
        page_logger.error("模型特征列表为空")
        st.error("模型配置错误：特征列表为空。")
        return None

    return prediction_model.feature_names


class PredictionModel:
    def __init__(self, model_type: str, global_categories_df: pd.DataFrame | None = None):
        self.model_type = model_type
        (
            self.model,
            self.feature_names,
            self.level_fallback_mapping,
            self.feature_engineer_state,
            self.prediction_threshold,
        ) = load_model(model_type)

        if self.model is None:
            raise ValueError(f"加载模型 '{model_type}' 失败")

        if not isinstance(self.feature_names, list):
            self.feature_names = None
            page_logger.warning(f"模型 '{model_type}' 未提供 feature_names")

        self.school_level_service = get_school_level_service()
        self.level_fallback_mapping = self.level_fallback_mapping or {}
        self.feature_engineer_state = self.feature_engineer_state or {}
        self.prediction_threshold = float(
            getattr(self.model, "prediction_threshold", self.prediction_threshold)
            or DEFAULT_PREDICTION_THRESHOLD
        )
        self.numeric_medians = self.feature_engineer_state.get("numeric_medians", {})
        self.cap_values = self.feature_engineer_state.get("cap_values", {})
        self.saved_categorical_levels = self.feature_engineer_state.get("categorical_levels", {})

        self._setup_global_categories(global_categories_df)
        self._enable_categorical = self._check_categorical_support()

        self._get_base_features_cached = lru_cache(maxsize=128)(self._preprocess_base_features_raw)

    def _setup_global_categories(self, global_categories_df: pd.DataFrame | None):
        self.global_categories = {}
        self.global_category_index = {}

        for col in [
            "target_university",
            "target_major",
            "background_university",
            "background_major",
        ]:
            categories = self.saved_categorical_levels.get(col)
            if (
                categories is None
                and global_categories_df is not None
                and col in global_categories_df.columns
            ):
                categories = global_categories_df[col].cat.categories.tolist()
            categories = list(categories or [])
            self.global_categories[col] = categories
            self.global_category_index[col] = {str(cat): idx for idx, cat in enumerate(categories)}

    def _resolve_background_university_value(self, value: Any) -> str:
        str_val = str(value) if value is not None else ""
        if str_val in self.global_category_index.get("background_university", {}):
            return str_val
        if not self.level_fallback_mapping:
            return str_val

        level = self.school_level_service.get_school_level(str_val)
        fallback = self.level_fallback_mapping.get(level)
        return str(fallback) if fallback else str_val

    def _preprocess_numeric_value(self, col: str, value: Any) -> float:
        num = pd.to_numeric(value, errors="coerce")
        if pd.isna(num):
            num = self.numeric_medians.get(col, 0.0)
        return float(num)

    def _check_categorical_support(self) -> bool:
        if hasattr(self.model, "get_booster"):
            booster = self.model.get_booster()
            if booster and getattr(booster, "feature_types", None):
                return any(t == "c" for t in booster.feature_types)

        if hasattr(self.model, "get_xgb_params"):
            return bool(self.model.get_xgb_params().get("enable_categorical", False))

        return False

    def _preprocess_single_value(self, col: str, value: Any) -> Any:
        if col in COUNT_COLUMNS_FOR_LOG_TRANSFORM:
            num = self._preprocess_numeric_value(col, value)
            cap_value = self.cap_values.get(col)
            if cap_value is not None:
                num = min(num, float(cap_value))
            return float(np.log1p(max(0.0, num)))

        if col in [
            "target_university",
            "target_major",
            "background_university",
            "background_major",
        ]:
            str_val = str(value) if value is not None else ""
            if col == "background_university":
                str_val = self._resolve_background_university_value(value)

            if self._enable_categorical:
                return str_val if str_val in self.global_category_index.get(col, {}) else ""

            idx_map = self.global_category_index.get(col, {})
            code = idx_map.get(str_val, -1)
            return float(code)

        return self._preprocess_numeric_value(col, value)

    def _preprocess_base_features_raw(
        self, input_tuple: tuple, features_to_use: tuple
    ) -> dict[str, Any]:
        input_data = dict(input_tuple)
        exclude = {"target_university", "target_major"}
        return {
            f: self._preprocess_single_value(f, input_data.get(f, np.nan))
            for f in features_to_use
            if f not in exclude
        }

    def _create_prediction_dataframe(
        self,
        combinations: list[tuple[str, str]],
        base_features: dict[str, Any],
        features_to_use: list[str],
    ) -> pd.DataFrame:
        n = len(combinations)
        unis, majors = zip(*combinations, strict=True)
        data = {}

        for feat in features_to_use:
            if feat == "target_university":
                vals = unis
            elif feat == "target_major":
                vals = majors
            elif feat in base_features:
                val = base_features[feat]
                if self._enable_categorical and feat in self.global_categories:
                    vals = [val] * n
                else:
                    data[feat] = np.full(n, val, dtype=np.float32)
                    continue
            else:
                continue

            if self._enable_categorical and feat in self.global_categories:
                data[feat] = pd.Categorical(
                    vals, categories=self.global_categories[feat], ordered=False
                )
            elif feat in [
                "target_university",
                "target_major",
                "background_university",
                "background_major",
            ]:
                idx_map = self.global_category_index.get(feat, {})
                data[feat] = np.array([idx_map.get(str(v), -1) for v in vals], dtype=np.int32)
            else:
                data[feat] = np.array(vals, dtype=np.float32)

        return pd.DataFrame(data, columns=features_to_use)

    def predict_batch(
        self,
        input_data: dict[str, Any],
        combinations: list[tuple[str, str]],
        expected_features: list[str],
    ) -> list[dict[str, Any]]:
        if not combinations or not self.model:
            return []

        features = self.feature_names or expected_features
        if not features:
            return []

        input_tuple = tuple(sorted(input_data.items()))
        base_preprocessed = self._get_base_features_cached(input_tuple, tuple(features))

        df = self._create_prediction_dataframe(combinations, base_preprocessed, features)

        probas = self.model.predict_proba(df)
        if probas.ndim == 2 and probas.shape[1] > 1:
            probas = probas[:, 1]

        predictions = (probas >= self.prediction_threshold).astype(int)

        return [
            {"university": u, "major": m, "probability": float(p), "prediction": int(pred)}
            for (u, m), p, pred in zip(combinations, probas, predictions, strict=True)
        ]
