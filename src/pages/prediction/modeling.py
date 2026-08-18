import threading
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.ml.data_config import (
    COUNT_COLUMNS_FOR_LOG_TRANSFORM,
    DEFAULT_PREDICTION_THRESHOLD,
)
from src.pages.prediction.model_loader import load_model
from src.utils.logger import setup_logger
from src.utils.schools.level_service import get_school_level_service

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
    def __init__(
        self,
        model_type: str,
        global_categories_df: pd.DataFrame | None = None,
        level_override: dict[str, str] | None = None,
        cross_level_levels: list[str] | None = None,
        major_override: dict[str, str] | None = None,
    ):
        self.model_type = model_type
        (
            self.model,
            self.feature_names,
            self.level_fallback_mapping,
            self.feature_engineer_state,
            self.prediction_threshold,
            _loaded_cross_level,
        ) = load_model(model_type)

        if self.model is None:
            raise ValueError(f"加载模型 '{model_type}' 失败")

        if not isinstance(self.feature_names, list):
            self.feature_names = None
            page_logger.warning(f"模型 '{model_type}' 未提供 feature_names")

        self.school_level_service = get_school_level_service()
        self.level_fallback_mapping = self.level_fallback_mapping or {}
        self.level_override: dict[str, str] = dict(level_override or {})
        self.major_override: dict[str, str] = dict(major_override or {})
        self.cross_level_levels: list[str] = list(
            cross_level_levels if cross_level_levels is not None else _loaded_cross_level or []
        )
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
        self._predict_lock = threading.Lock()

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

        page_logger.info(
            "全局类别加载完成 | uni=%d major=%d bg_uni=%d bg_major=%d",
            len(self.global_categories.get("target_university", [])),
            len(self.global_categories.get("target_major", [])),
            len(self.global_categories.get("background_university", [])),
            len(self.global_categories.get("background_major", [])),
        )

    def _resolve_background_university_value(self, value: Any) -> str:
        str_val = str(value) if value is not None else ""
        if not str_val:
            return str_val

        if str_val in self.level_override:
            resolved = self.level_override[str_val]
            if resolved in self.cross_level_levels:
                page_logger.info(
                    "院校 '%s' override level='%s' 在 cross_level_levels → blocked",
                    str_val,
                    resolved,
                )
            if resolved in self.global_category_index.get("background_university", {}):
                return resolved
            fallback = self.level_fallback_mapping.get(resolved)
            if fallback:
                page_logger.info(
                    "院校 '%s' override level='%s' → fallback='%s'",
                    str_val,
                    resolved,
                    fallback,
                )
                return str(fallback)
            return resolved

        if str_val in self.global_category_index.get("background_university", {}):
            return str_val

        level = self.school_level_service.get_school_level(str_val)

        if level == "未知":
            ai_level = self._infer_level_via_agent(str_val)
            if ai_level is not None:
                self.level_override[str_val] = ai_level
                level = ai_level

        if level in self.cross_level_levels:
            page_logger.info(
                "院校 '%s' level='%s' 在 cross_level_levels → blocked",
                str_val,
                level,
            )

        if not self.level_fallback_mapping:
            return str_val

        fallback = self.level_fallback_mapping.get(level)
        return str(fallback) if fallback else str_val

    def _resolve_background_major_value(self, value: Any) -> str:
        str_val = str(value) if value is not None else ""
        if not str_val:
            return str_val

        if str_val in self.major_override:
            resolved = self.major_override[str_val]
            if resolved == str_val:
                return resolved
            if resolved in self.global_category_index.get("background_major", {}):
                return resolved
            page_logger.warning(
                "major_override 值不在 global_category_index | raw=%s override=%s",
                str_val,
                resolved,
            )
            return resolved

        if str_val in self.global_category_index.get("background_major", {}):
            return str_val

        canonical = self._infer_major_via_agent(str_val)
        if canonical is not None and canonical in self.global_category_index.get(
            "background_major", {}
        ):
            self.major_override[str_val] = canonical
            return canonical

        self.major_override[str_val] = str_val
        return str_val

    def _infer_major_via_agent(self, raw_major: str) -> str | None:
        try:
            from src.agent import get_background_major_agent

            agent = get_background_major_agent()
            decision = agent.resolve_major(raw_major, use_persistent_cache=True)
            if decision.canonical_major != "未知":
                page_logger.info(
                    "BackgroundMajorAgent 映射 | raw=%s → canonical=%s confidence=%s",
                    raw_major,
                    decision.canonical_major,
                    decision.confidence,
                )
                return decision.canonical_major
            else:
                page_logger.info(
                    "BackgroundMajorAgent 无法映射 | raw=%s confidence=%s reason=%s",
                    raw_major,
                    decision.confidence,
                    decision.reasoning,
                )
                return None
        except Exception:
            page_logger.warning(
                "BackgroundMajorAgent 调用失败 | raw=%s",
                raw_major,
                exc_info=True,
            )
            return None

    def _infer_level_via_agent(self, school_name: str) -> str | None:
        try:
            from src.agent import get_school_level_agent

            agent = get_school_level_agent()
            decision = agent.infer_school_level(school_name, use_persistent_cache=True)
            if decision.school_level != "未知":
                page_logger.info(
                    "SchoolLevelAgent 推断 | school=%s level=%s confidence=%s",
                    school_name,
                    decision.school_level,
                    decision.confidence,
                )
                return decision.school_level
            else:
                page_logger.info(
                    "SchoolLevelAgent 无法推断 | school=%s confidence=%s reason=%s",
                    school_name,
                    decision.confidence,
                    decision.reasoning,
                )
                return None
        except Exception:
            page_logger.warning(
                "SchoolLevelAgent 调用失败 | school=%s",
                school_name,
                exc_info=True,
            )
            return None

    def check_cross_level_blocked(self, school_name: str) -> bool:
        name = str(school_name or "").strip()
        if not name or not self.cross_level_levels:
            return False

        resolved = self.level_override.get(name)
        if resolved is None:
            resolved = self.school_level_service.get_school_level(name)

        if resolved == "未知":
            ai_level = self._infer_level_via_agent(name)
            if ai_level is not None:
                self.level_override[name] = ai_level
                resolved = ai_level
            else:
                self.level_override[name] = "未知"

        return resolved in self.cross_level_levels

    def _preprocess_numeric_value(self, col: str, value: Any) -> float:
        num = pd.to_numeric(value, errors="coerce")
        if pd.isna(num):
            num = self.numeric_medians.get(col, 0.0)
        return float(num)

    def _check_categorical_support(self) -> bool:
        if hasattr(self.model, "get_booster"):
            booster = self.model.get_booster()
            if booster and getattr(booster, "feature_types", None):
                result = any(t == "c" for t in booster.feature_types)
                page_logger.info("XGBoost categorical 支持 (booster feature_types): %s", result)
                return result

        if hasattr(self.model, "get_xgb_params"):
            result = bool(self.model.get_xgb_params().get("enable_categorical", False))
            page_logger.info("XGBoost categorical 支持 (xgb_params): %s", result)
            return result

        page_logger.info("XGBoost categorical 支持: False (未检测到 categorical 配置)")
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
            elif col == "background_major":
                str_val = self._resolve_background_major_value(value)

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

    def feature_contributions(
        self,
        input_data: dict[str, Any],
        university: str,
        major: str,
        top_k: int = 8,
    ) -> dict[str, Any] | None:
        try:
            import xgboost as xgb

            booster = self.model.get_booster() if hasattr(self.model, "get_booster") else None
            features = self.feature_names
            if booster is None or not features:
                return None

            input_tuple = tuple(sorted(input_data.items()))
            base = self._preprocess_base_features_raw(input_tuple, tuple(features))
            df = self._create_prediction_dataframe([(str(university), str(major))], base, features)

            with self._predict_lock:
                contribs = booster.predict(
                    xgb.DMatrix(df, enable_categorical=self._enable_categorical),
                    pred_contribs=True,
                )
            row = contribs[0]
            if len(row) != len(features) + 1:
                return None
            items = [
                {"feature": features[i], "contrib": float(row[i])} for i in range(len(features))
            ]
            items.sort(key=lambda x: abs(x["contrib"]), reverse=True)
            return {"bias": float(row[-1]), "items": items[:top_k]}
        except Exception:
            page_logger.warning("feature_contributions 计算失败", exc_info=True)
            return None

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

        n_combos = len(combinations)
        t0 = time.monotonic()
        input_tuple = tuple(sorted(input_data.items()))
        base_preprocessed = self._preprocess_base_features_raw(input_tuple, tuple(features))

        df = self._create_prediction_dataframe(combinations, base_preprocessed, features)

        with self._predict_lock:
            probas = self.model.predict_proba(df)
        if probas.ndim == 2 and probas.shape[1] > 1:
            probas = probas[:, 1]

        elapsed = time.monotonic() - t0
        predictions = (probas >= self.prediction_threshold).astype(int)

        page_logger.info(
            "XGBoost 批量推理完成 | combinations=%d features=%d elapsed=%.3fs "
            "prob_mean=%.4f prob_median=%.4f prob_min=%.4f prob_max=%.4f",
            n_combos,
            len(features),
            elapsed,
            float(probas.mean()),
            float(np.median(probas)),
            float(probas.min()),
            float(probas.max()),
        )

        return [
            {"university": u, "major": m, "probability": float(p), "prediction": int(pred)}
            for (u, m), p, pred in zip(combinations, probas, predictions, strict=True)
        ]
