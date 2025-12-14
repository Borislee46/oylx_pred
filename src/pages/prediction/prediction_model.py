from typing import Any

import numpy as np
import pandas as pd

from src.machine_learning_models.data_config import (
    CATEGORICAL_COLUMNS,
    COUNT_COLUMNS_FOR_LOG_TRANSFORM,
)
from src.utils.logger import setup_logger
from src.utils.model_loader import load_model
from src.utils.school_level_service import get_school_level_service

page_logger = setup_logger("page3", "prediction")


class PredictionModel:
    def __init__(self, model_type: str, global_categories_df: pd.DataFrame | None = None):
        self.model_type = model_type
        self.model, self.feature_names, self.level_fallback_mapping = load_model(model_type)

        if self.model is None:
            raise ValueError(f"加载模型 '{model_type}' 失败")

        if not isinstance(self.feature_names, list):
            self.feature_names = None
            page_logger.warning(
                f"模型 '{model_type}' 未提供 feature_names，将依赖传入的 expected_features"
            )

        self.school_level_service = get_school_level_service()

        if self.level_fallback_mapping is None:
            self.level_fallback_mapping = {}
            page_logger.warning("未找到 level_fallback_mapping，未知学校将使用默认编码 -1")

        self._setup_global_categories(global_categories_df)
        self._enable_categorical = self._check_categorical_support()
        self._preprocessed_cache: dict[tuple, dict[str, Any]] = {}
        self._cache_maxsize = 128

    def _setup_global_categories(self, global_categories_df: pd.DataFrame | None):
        self.global_categories = {}
        self.global_category_index = {}

        if global_categories_df is None:
            page_logger.error("未提供 global_categories_df，无法为分类特征建立全局类别映射")
            return

        for col in CATEGORICAL_COLUMNS:
            if col not in global_categories_df.columns:
                page_logger.warning(f"分类列 '{col}' 未在提供的 global_categories_df 中找到")
                continue

            if not pd.api.types.is_categorical_dtype(global_categories_df[col]):
                page_logger.warning(f"列 '{col}' 不是 category 类型")
                continue

            categories = global_categories_df[col].cat.categories.tolist()
            self.global_categories[col] = categories
            self.global_category_index[col] = {str(cat): idx for idx, cat in enumerate(categories)}

    def _check_categorical_support(self) -> bool:
        booster = None
        if hasattr(self.model, "get_booster"):
            booster = self.model.get_booster()
        if booster is not None:
            feature_types = getattr(booster, "feature_types", None)
            if feature_types and any(t == "c" for t in feature_types):
                return True

        if hasattr(self.model, "get_xgb_params"):
            return bool(self.model.get_xgb_params().get("enable_categorical", False))
        return False

    def _log_transform_value(self, value: Any) -> float:
        numeric_val = pd.to_numeric(value, errors="coerce")
        return float(np.log1p(max(0, numeric_val if not pd.isna(numeric_val) else 0)))

    def _get_category_code(self, col: str, value: Any) -> int:
        index_map = self.global_category_index.get(col)
        if index_map is None:
            return 0

        str_val = str(value)
        code = index_map.get(str_val, -1)

        if code == -1 and col == "background_university" and self.level_fallback_mapping:
            school_level = self.school_level_service.get_school_level(str_val)
            fallback_school = self.level_fallback_mapping.get(school_level)

            if fallback_school:
                fallback_code = index_map.get(str(fallback_school), -1)
                if fallback_code != -1:
                    return fallback_code

        return code

    def _preprocess_single_value(self, col: str, value: Any) -> float:
        if col in COUNT_COLUMNS_FOR_LOG_TRANSFORM:
            return self._log_transform_value(value)
        elif col in CATEGORICAL_COLUMNS:
            if self._enable_categorical:
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return ""
                return str(value)
            return float(self._get_category_code(col, value))
        else:
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0

    def _get_preprocessed_base_features(self, input_data_tuple: tuple) -> dict[str, Any]:
        if input_data_tuple in self._preprocessed_cache:
            return self._preprocessed_cache[input_data_tuple]

        input_data = dict(input_data_tuple)
        base_features = [
            f for f in (self.feature_names or []) if f not in ["target_university", "target_major"]
        ]

        result = {
            feat: self._preprocess_single_value(feat, input_data.get(feat, np.nan))
            for feat in base_features
        }

        if len(self._preprocessed_cache) >= self._cache_maxsize:
            self._preprocessed_cache.clear()
        self._preprocessed_cache[input_data_tuple] = result
        return result

    def _create_prediction_dataframe(
        self, combinations: list[tuple[str, str]], preprocessed_base: dict[str, Any]
    ) -> pd.DataFrame:
        if not combinations:
            return pd.DataFrame()

        universities, majors = zip(*combinations, strict=True)
        n = len(combinations)

        data_dict = {}
        for feat, value in preprocessed_base.items():
            if feat in CATEGORICAL_COLUMNS:
                self._add_categorical_feature(data_dict, feat, [value] * n)
            else:
                data_dict[feat] = np.full(n, value, dtype=np.float32)

        self._add_categorical_feature(data_dict, "target_university", list(universities))
        self._add_categorical_feature(data_dict, "target_major", list(majors))

        features_to_use = self.feature_names or list(data_dict.keys())
        return pd.DataFrame(data_dict, columns=features_to_use)

    def _add_categorical_feature(self, data_dict: dict, feature_name: str, values: list):
        if feature_name not in (self.feature_names or []):
            return

        if self._enable_categorical and feature_name in self.global_categories:
            data_dict[feature_name] = pd.Categorical(
                values, categories=self.global_categories[feature_name], ordered=False
            )
        else:
            index_map = self.global_category_index.get(feature_name, {})

            codes = (
                pd.Series(values, dtype=object)
                .astype(str)
                .map(index_map)
                .fillna(-1)
                .astype(np.int32)
            )

            data_dict[feature_name] = codes.values

    def predict_batch(
        self,
        input_data: dict[str, Any],
        combinations: list[tuple[str, str]],
        expected_features: list[str],
    ) -> list[dict[str, Any]]:
        if not combinations:
            page_logger.warning("预测组合列表为空")
            return []

        if not self.model:
            page_logger.error("预测模型未初始化")
            raise ValueError("预测模型未初始化")

        input_data_tuple = tuple(sorted(input_data.items()))
        preprocessed_base = self._get_preprocessed_base_features(input_data_tuple)

        prediction_df = self._create_prediction_dataframe(combinations, preprocessed_base)

        if prediction_df.empty:
            page_logger.warning("预测DataFrame为空")
            return []

        try:
            probas = self.model.predict_proba(prediction_df)
            if probas.ndim == 2 and probas.shape[1] > 1:
                probas = probas[:, 1]

            return [
                {"university": univ, "major": major, "probability": float(proba)}
                for (univ, major), proba in zip(combinations, probas, strict=True)
            ]
        except Exception as e:
            page_logger.error(f"模型预测失败: {e}", exc_info=True)
            raise

    def predict_probability(self, input_df: pd.DataFrame) -> float | None:
        if self.model is None:
            return None

        proba = self.model.predict_proba(input_df)[0, 1]
        return float(proba)
