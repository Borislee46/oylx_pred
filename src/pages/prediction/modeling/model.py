from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.machine_learning_models.data_config import (
    CATEGORICAL_COLUMNS,
    COUNT_COLUMNS_FOR_LOG_TRANSFORM,
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

    features = getattr(prediction_model, "feature_names", None)
    if features:
        return features

    page_logger.error("模型特征列表为空")
    st.error("模型配置错误：特征列表为空。")
    return None


class PredictionModel:
    def __init__(self, model_type: str, global_categories_df: pd.DataFrame | None = None):
        self.model_type = model_type
        self.model, self.feature_names, self.level_fallback_mapping = load_model(model_type)

        if self.model is None:
            raise ValueError(f"加载模型 '{model_type}' 失败")

        if not isinstance(self.feature_names, list):
            self.feature_names = None
            page_logger.warning(f"模型 '{model_type}' 未提供 feature_names")

        self.school_level_service = get_school_level_service()
        self.level_fallback_mapping = self.level_fallback_mapping or {}

        self._setup_global_categories(global_categories_df)
        self._enable_categorical = self._check_categorical_support()

        self._get_base_features_cached = lru_cache(maxsize=128)(self._preprocess_base_features_raw)

    def _setup_global_categories(self, global_categories_df: pd.DataFrame | None):
        self.global_categories = {}
        self.global_category_index = {}

        if global_categories_df is None:
            page_logger.error("未提供 global_categories_df")
            return

        for col in CATEGORICAL_COLUMNS:
            if col not in global_categories_df.columns:
                continue

            if not pd.api.types.is_categorical_dtype(global_categories_df[col]):
                continue

            categories = global_categories_df[col].cat.categories.tolist()
            self.global_categories[col] = categories
            self.global_category_index[col] = {str(cat): idx for idx, cat in enumerate(categories)}

    def _check_categorical_support(self) -> bool:
        try:
            if hasattr(self.model, "get_booster"):
                booster = self.model.get_booster()
                if booster and getattr(booster, "feature_types", None):
                    return any(t == "c" for t in booster.feature_types)

            if hasattr(self.model, "get_xgb_params"):
                return bool(self.model.get_xgb_params().get("enable_categorical", False))
        except Exception:
            pass
        return False

    def _preprocess_single_value(self, col: str, value: Any) -> Any:
        """单值预处理逻辑"""
        if col in COUNT_COLUMNS_FOR_LOG_TRANSFORM:
            num = pd.to_numeric(value, errors="coerce")
            return float(np.log1p(max(0, num if not pd.isna(num) else 0)))

        if col in CATEGORICAL_COLUMNS:
            if self._enable_categorical:
                return (
                    str(value)
                    if value is not None and not (isinstance(value, float) and np.isnan(value))
                    else ""
                )

            # 非原生分类支持时，使用编码
            idx_map = self.global_category_index.get(col, {})
            str_val = str(value)
            code = idx_map.get(str_val, -1)

            # 学校等级回退逻辑
            if code == -1 and col == "background_university" and self.level_fallback_mapping:
                level = self.school_level_service.get_school_level(str_val)
                fallback = self.level_fallback_mapping.get(level)
                if fallback:
                    code = idx_map.get(str(fallback), -1)
            return float(code)

        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _preprocess_base_features_raw(
        self, input_tuple: tuple, features_to_use: tuple
    ) -> dict[str, Any]:
        """原始预处理逻辑（供 lru_cache 包装）"""
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
                vals = list(unis)
            elif feat == "target_major":
                vals = list(majors)
            elif feat in base_features:
                vals = [base_features[feat]] * n
            else:
                continue

            if self._enable_categorical and feat in self.global_categories:
                data[feat] = pd.Categorical(
                    vals, categories=self.global_categories[feat], ordered=False
                )
            elif feat in CATEGORICAL_COLUMNS:
                idx_map = self.global_category_index.get(feat, {})
                data[feat] = (
                    pd.Series(vals).astype(str).map(idx_map).fillna(-1).astype(np.int32).values
                )
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

        # 缓存基础特征预处理结果
        input_tuple = tuple(sorted(input_data.items()))
        base_preprocessed = self._get_base_features_cached(input_tuple, tuple(features))

        df = self._create_prediction_dataframe(combinations, base_preprocessed, features)
        if df.empty:
            return []

        try:
            probas = self.model.predict_proba(df)
            if probas.ndim == 2 and probas.shape[1] > 1:
                probas = probas[:, 1]

            return [
                {"university": u, "major": m, "probability": float(p)}
                for (u, m), p in zip(combinations, probas, strict=True)
            ]
        except Exception as e:
            page_logger.error(f"模型预测失败: {e}", exc_info=True)
            raise
