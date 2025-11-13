from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class ShapExplainer:
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        prediction_model: Any = None,
        background_size: int = 100,
        nsamples: int = 100,
    ):
        self.model = model
        self.feature_names = feature_names
        self.prediction_model = prediction_model
        self.background_size = background_size
        self.nsamples = nsamples
        self.explainer = None
        self.background = None
        self._init_explainer()

    def _init_explainer(self):
        try:
            base_model = self.model.base_model if hasattr(self.model, "base_model") else self.model
            if base_model is None:
                logger.warning("无法获取基础模型，SHAP explainer 初始化失败")
                return

            dummy_df = self._make_dummy_background()
            if dummy_df.empty:
                return

            numeric_cols = dummy_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == len(dummy_df.columns) and len(dummy_df) > 1:
                self.background = shap.kmeans(dummy_df, min(self.background_size, len(dummy_df)))
            else:
                self.background = dummy_df.iloc[[0]]

            def _predict_fn(X):
                X_df = (
                    X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=dummy_df.columns)
                )
                return base_model.predict_proba(X_df)[:, 1]

            self.explainer = shap.KernelExplainer(_predict_fn, self.background)
            logger.info("SHAP KernelExplainer 初始化成功")
        except Exception as e:
            logger.error(f"SHAP KernelExplainer 初始化失败: {e}", exc_info=True)
            self.explainer = None
            self.background = None

    def _make_dummy_background(self) -> pd.DataFrame:
        if self.prediction_model is not None:
            from src.machine_learning_models.data_config import CATEGORICAL_COLUMNS

            dummy_input = {
                feat: 0.0 for feat in self.feature_names if feat not in CATEGORICAL_COLUMNS
            }
            combinations = [("", "")]
            preprocessed_base = self.prediction_model._get_preprocessed_base_features(
                tuple(sorted(dummy_input.items()))
            )
            return self.prediction_model._create_prediction_dataframe(
                combinations, preprocessed_base
            )

        return pd.DataFrame({col: [0.0] for col in self.feature_names})

    def create_force_plot(
        self,
        input_data: Dict[str, Any],
        target_university: str,
        target_major: str,
        prediction_model: Any,
        session_manager: Any = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, List[str]]]:
        try:
            prediction_df, raw_feature_values, feature_names_filtered = (
                self._create_prediction_dataframe_with_raw_values(
                    input_data, target_university, target_major, prediction_model, session_manager
                )
            )
            if prediction_df.empty or self.explainer is None:
                return None

            available_cols = [col for col in feature_names_filtered if col in prediction_df.columns]
            prediction_df_filtered = prediction_df[available_cols]

            if len(available_cols) != len(feature_names_filtered):
                missing_cols = [
                    col for col in feature_names_filtered if col not in prediction_df.columns
                ]
                logger.warning(
                    f"特征数量不匹配: 期望{len(feature_names_filtered)}个，实际可用{len(available_cols)}个，缺失: {missing_cols}"
                )

            feature_to_index = {feat: idx for idx, feat in enumerate(feature_names_filtered)}
            raw_values_reordered = [
                raw_feature_values[feature_to_index[col]] for col in available_cols
            ]
            feature_names_reordered = available_cols.copy()

            np.random.seed(42)
            if hasattr(shap, "seed"):
                shap.seed(42)

            shap_values = self.explainer.shap_values(
                prediction_df_filtered.iloc[[0]], nsamples=self.nsamples
            )
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if shap_values.ndim > 1:
                shap_values = shap_values.flatten()

            if len(shap_values) != len(raw_values_reordered):
                logger.warning(
                    f"SHAP值数量({len(shap_values)})与特征值数量({len(raw_values_reordered)})不匹配"
                )
                if len(shap_values) > len(raw_values_reordered):
                    shap_values = shap_values[: len(raw_values_reordered)]
                else:
                    raw_values_reordered = raw_values_reordered[: len(shap_values)]
                    feature_names_reordered = feature_names_reordered[: len(shap_values)]

            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = float(expected_value[0])

            raw_values_array = np.array(raw_values_reordered, dtype=object)
            return shap_values, raw_values_array, expected_value, feature_names_reordered
        except Exception as e:
            logger.error(f"生成 SHAP force plot 失败: {e}", exc_info=True)
            return None

    def _create_prediction_dataframe_with_raw_values(
        self,
        input_data: Dict[str, Any],
        target_university: str,
        target_major: str,
        prediction_model: Any,
        session_manager: Any = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        from src.machine_learning_models.data_config import CATEGORICAL_COLUMNS

        try:
            expected_features = prediction_model.feature_names or self.feature_names
            feature_names_filtered = [f for f in expected_features if f != "school_level"]
            base_features = [
                f for f in feature_names_filtered if f not in ["target_university", "target_major"]
            ]

            model_input_features = {}
            for f in base_features:
                if f in input_data and isinstance(input_data[f], (float, int, str)):
                    model_input_features[f] = input_data[f]
                else:
                    model_input_features[f] = 0.0 if f not in CATEGORICAL_COLUMNS else ""

            combinations = [(target_university, target_major)]
            preprocessed_base = prediction_model._get_preprocessed_base_features(
                tuple(sorted(model_input_features.items()))
            )
            prediction_df = prediction_model._create_prediction_dataframe(
                combinations, preprocessed_base
            )

            if prediction_df.empty:
                return pd.DataFrame(), np.array([]), feature_names_filtered

            raw_values = []
            for feat in feature_names_filtered:
                if feat == "target_university":
                    raw_values.append(target_university)
                elif feat == "target_major":
                    raw_values.append(target_major)
                elif feat == "background_major":
                    if session_manager:
                        original_form_data = session_manager.get("original_form_data", {})
                        raw_val = (
                            session_manager.get("background_major_original_initial")
                            or session_manager.get("background_major_original")
                            or original_form_data.get("background_major_original")
                        )
                        if raw_val:
                            raw_values.append(str(raw_val))
                        else:
                            raw_values.append(str(input_data.get(feat, "")))
                    else:
                        raw_values.append(str(input_data.get(feat, "")))
                elif feat == "language_score":
                    if session_manager:
                        original_form_data = session_manager.get("original_form_data", {})
                        lang_raw = (
                            session_manager.get("language_score_input")
                            or session_manager.get("language_score_raw")
                            or original_form_data.get("language_score_raw")
                        )
                        lang_type = (
                            session_manager.get("language_type")
                            or original_form_data.get("language_type")
                            or "雅思"
                        )
                        if lang_raw is not None and lang_raw != "":
                            raw_values.append(f"{lang_raw} {lang_type}")
                        else:
                            raw_values.append(str(input_data.get(feat, 0.0)))
                    else:
                        raw_values.append(str(input_data.get(feat, 0.0)))
                elif feat == "gpa":
                    if session_manager:
                        original_form_data = session_manager.get("original_form_data", {})
                        gpa_raw = session_manager.get("gpa_raw_input") or original_form_data.get(
                            "gpa_raw"
                        )
                        if gpa_raw is not None:
                            raw_values.append(str(gpa_raw))
                        else:
                            raw_values.append(str(input_data.get(feat, 0.0)))
                    else:
                        raw_values.append(str(input_data.get(feat, 0.0)))
                elif feat in input_data:
                    raw_val = input_data[feat]
                    if feat in CATEGORICAL_COLUMNS:
                        raw_values.append(str(raw_val))
                    else:
                        numeric_val = pd.to_numeric(raw_val, errors="coerce")
                        raw_values.append(str(numeric_val) if not pd.isna(numeric_val) else "0.0")
                else:
                    raw_values.append("" if feat in CATEGORICAL_COLUMNS else "0.0")

            return prediction_df, np.array(raw_values, dtype=object), feature_names_filtered
        except Exception as e:
            logger.error(f"创建预测 DataFrame 和原始值失败: {e}", exc_info=True)
            return pd.DataFrame(), np.array([]), []

    def _get_feature_display_names(self, feature_names: List[str]) -> List[str]:
        mapping = {
            "gpa": "GPA",
            "language_score": "语言成绩",
            "internship_count": "实习数量",
            "research_count": "研究数量",
            "award_count": "奖项数量",
            "paper_count": "论文数量",
            "background_university": "背景院校",
            "background_major": "背景专业",
            "target_university": "目标院校",
            "target_major": "目标专业",
        }
        return [mapping.get(f, f) for f in feature_names]
