from typing import Any

import numpy as np
import pandas as pd

from src.machine_learning_models.data_config import (
    CATEGORICAL_COLUMNS,
    COUNT_COLUMNS_FOR_LOG_TRANSFORM,
)
from src.utils.logger import setup_logger
from src.utils.model_loader import load_model

page_logger = setup_logger("page3", "prediction")


class PredictionModel:
    def __init__(self, model_type: str, global_categories_df: pd.DataFrame | None = None):
        self.model_type = model_type
        self.model, self.feature_names = load_model(model_type)
        self.global_categories_ = {}
        self.global_category_index_: dict[str, dict[str, int]] = {}
        self._base_features_cache: dict[str, Any] = {}
        self._cache_key: str | None = None

        if self.model is None:
            raise ValueError(f"加载模型 '{model_type}' 失败")

        if self.feature_names is None or not isinstance(self.feature_names, list):
            self.feature_names = None
            page_logger.warn(
                f"模型 '{model_type}' 未提供 feature_names。将依赖传入的 expected_features。"
            )

        if global_categories_df is not None:
            for col in CATEGORICAL_COLUMNS:
                if col in global_categories_df.columns and pd.api.types.is_categorical_dtype(
                    global_categories_df[col]
                ):
                    self.global_categories_[col] = global_categories_df[col].cat.categories.tolist()
                    try:
                        categories = self.global_categories_[col]
                        self.global_category_index_[col] = {
                            str(cat): idx for idx, cat in enumerate(categories)
                        }
                    except Exception:
                        self.global_category_index_[col] = {}
                elif col in global_categories_df.columns:
                    page_logger.warn(
                        f"列 '{col}' 在 global_categories_df 中存在，但不是 category 类型。请确保它已正确预处理。"
                    )
                else:
                    page_logger.warn(f"分类列 '{col}' 未在提供的 global_categories_df 中找到。")
        else:
            page_logger.error(
                "未提供 global_categories_df，无法为分类特征建立全局类别映射。预测可能不准确。"
            )

    def _preprocess_base_features(
        self,
        input_data: dict[str, Any],
        base_features: list[str],
        n_rows: int,
    ) -> dict[str, Any]:
        preprocessed_cols = {}

        for feat in base_features:
            val = input_data.get(feat, np.nan)

            if feat in COUNT_COLUMNS_FOR_LOG_TRANSFORM:
                try:
                    numeric_val = pd.to_numeric(val, errors="coerce")
                    if pd.isna(numeric_val):
                        numeric_val = 0
                    transformed_val = np.log1p(max(0, numeric_val))
                except Exception:
                    transformed_val = 0
                preprocessed_cols[feat] = transformed_val
            else:
                preprocessed_cols[feat] = val

        categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in base_features]
        for col in categorical_cols:
            if col in preprocessed_cols:
                str_val = str(preprocessed_cols[col])
                index_map = self.global_category_index_.get(col)
                if index_map is not None:
                    try:
                        code = index_map.get(str_val, -1)
                        if code == -1:
                            page_logger.warn(
                                f"列 '{col}' 的值 '{str_val}' 不在训练时的类别中，将使用 -1"
                            )
                        preprocessed_cols[col] = int(code)
                    except Exception as e:
                        page_logger.warn(f"处理分类列 '{col}' 失败: {e}")
                        preprocessed_cols[col] = -1
                else:
                    preprocessed_cols[col] = 0

        non_categorical_cols = [c for c in base_features if c not in categorical_cols]
        for col in non_categorical_cols:
            try:
                preprocessed_cols[col] = float(preprocessed_cols[col])
            except Exception:
                preprocessed_cols[col] = 0.0

        return preprocessed_cols

    def predict_batch(
        self,
        input_data: dict[str, Any],
        combinations: list[tuple[str, str]],
        expected_features: list[str],
    ) -> list[dict[str, Any]]:
        if not combinations:
            return []

        if not self.model:
            return []

        features_to_use = self.feature_names if self.feature_names else expected_features
        if not features_to_use:
            page_logger.error("predict_batch: 未确定要使用的特征。")
            return []

        try:
            universities, majors = zip(*combinations, strict=False) if combinations else ([], [])
            n = len(combinations)

            base_features = [
                f for f in features_to_use if f not in ["target_university", "target_major"]
            ]

            import hashlib

            base_input_str = ",".join(
                f"{k}={v}" for k, v in sorted(input_data.items()) if k in base_features
            )
            cache_key = hashlib.md5(base_input_str.encode()).hexdigest()

            if cache_key != self._cache_key:
                self._base_features_cache = self._preprocess_base_features(
                    input_data, base_features, n
                )
                self._cache_key = cache_key

            preprocessed_base = self._base_features_cache

            data_dict: dict[str, object] = {}

            for feat in base_features:
                if feat in preprocessed_base:
                    val = preprocessed_base[feat]
                    try:
                        data_dict[feat] = np.full(n, float(val), dtype=np.float32)
                    except Exception:
                        data_dict[feat] = np.full(n, val, dtype=object)

            enable_categorical = False
            try:
                if hasattr(self.model, "get_xgb_params"):
                    enable_categorical = bool(
                        self.model.get_xgb_params().get("enable_categorical", False)
                    )
            except Exception:
                enable_categorical = False

            if "target_university" in features_to_use:
                cats_uni = self.global_categories_.get("target_university")
                if enable_categorical and cats_uni is not None:
                    data_dict["target_university"] = pd.Categorical(
                        list(universities), categories=cats_uni, ordered=False
                    )
                else:
                    index_map_uni = self.global_category_index_.get("target_university", {})
                    try:
                        codes_uni_list = [index_map_uni.get(str(u), -1) for u in universities]
                        data_dict["target_university"] = np.asarray(codes_uni_list, dtype=np.int32)
                    except Exception:
                        data_dict["target_university"] = np.full(n, -1, dtype=np.int32)

            if "target_major" in features_to_use:
                cats_maj = self.global_categories_.get("target_major")
                if enable_categorical and cats_maj is not None:
                    data_dict["target_major"] = pd.Categorical(
                        list(majors), categories=cats_maj, ordered=False
                    )
                else:
                    index_map_maj = self.global_category_index_.get("target_major", {})
                    try:
                        codes_maj_list = [index_map_maj.get(str(m), -1) for m in majors]
                        data_dict["target_major"] = np.asarray(codes_maj_list, dtype=np.int32)
                    except Exception:
                        data_dict["target_major"] = np.full(n, -1, dtype=np.int32)

            use_numpy_fast_path = False
            try:
                has_any_categorical = any(isinstance(v, pd.Categorical) for v in data_dict.values())
                if not has_any_categorical and hasattr(self.model, "n_features_in_"):
                    use_numpy_fast_path = True
            except Exception:
                use_numpy_fast_path = False

            if use_numpy_fast_path:
                feature_columns = []
                for feat in features_to_use:
                    col = data_dict.get(feat)
                    if isinstance(col, pd.Categorical):
                        col = col.codes
                    if isinstance(col, np.ndarray):
                        feature_columns.append(col.reshape(-1, 1))
                    else:
                        feature_columns.append(np.asarray(col))
                try:
                    X = np.hstack(feature_columns).astype(np.float32, copy=False)
                    probas = self.model.predict_proba(X)
                    if probas.ndim == 2 and probas.shape[1] > 1:
                        probas = probas[:, 1]
                    probas = [float(p) for p in probas]
                    results = [
                        {"university": comb_univ, "major": comb_major, "probability": proba}
                        for (comb_univ, comb_major), proba in zip(
                            combinations, probas, strict=False
                        )
                    ]
                    return results
                except Exception:
                    pass

            preprocessed_df = pd.DataFrame(data_dict, columns=features_to_use)

        except ValueError as ve:
            page_logger.error(
                f"predict_batch: 向量化构造过程中出现 ValueError: {ve}", exc_info=True
            )
            return []
        except Exception as e:
            page_logger.error(f"predict_batch: 向量化构造失败，回退前置预处理: {e}", exc_info=True)
            try:
                base_feature_values = {
                    feat: input_data.get(feat, np.nan)
                    for feat in features_to_use
                    if feat not in ["target_university", "target_major"]
                }
                universities, majors = (
                    zip(*combinations, strict=False) if combinations else ([], [])
                )
                n_combinations = len(combinations)
                batch_data = {
                    feat: [val] * n_combinations for feat, val in base_feature_values.items()
                }
                batch_data["target_university"] = list(universities)
                batch_data["target_major"] = list(majors)
                all_combinations_df_raw = pd.DataFrame(batch_data, columns=features_to_use)
                preprocessed_df = self.preprocess_input(all_combinations_df_raw, features_to_use)
            except Exception as e2:
                page_logger.error(f"predict_batch: 回退预处理仍失败: {e2}", exc_info=True)
                return []

        if preprocessed_df.empty:
            page_logger.warn("predict_batch: 预处理后的 DataFrame 在开始预测循环前为空。")
            return []

        try:
            probas = self.model.predict_proba(preprocessed_df)
            if probas.ndim == 2 and probas.shape[1] > 1:
                probas = probas[:, 1]
            probas = [float(p) for p in probas]
            results = [
                {"university": comb_univ, "major": comb_major, "probability": proba}
                for (comb_univ, comb_major), proba in zip(combinations, probas, strict=False)
            ]
        except Exception as e:
            page_logger.error(f"predict_batch: 模型预测循环期间出现异常: {e}", exc_info=True)
            return []

        return results

    def preprocess_input(
        self, input_data: pd.DataFrame | dict, expected_features_list: list
    ) -> pd.DataFrame:
        try:
            if self.feature_names:
                expected_features_list = self.feature_names
            elif expected_features_list:
                pass
            else:
                raise ValueError(
                    "无法确定预期的特征列表 (feature_names 和 expected_features 均无效)。"
                )

            if isinstance(input_data, dict):
                data_for_df = {
                    feat: [input_data.get(feat, np.nan)] for feat in expected_features_list
                }
                input_df = pd.DataFrame(data_for_df, columns=expected_features_list)
            elif isinstance(input_data, pd.DataFrame):
                input_df = input_data.copy()
                missing_cols = [
                    col for col in expected_features_list if col not in input_df.columns
                ]
                for col in missing_cols:
                    input_df[col] = np.nan
                if list(input_df.columns) != expected_features_list:
                    input_df = input_df[expected_features_list]
            else:
                raise TypeError(f"不支持的 input_data 类型: {type(input_data)}")

            for col in COUNT_COLUMNS_FOR_LOG_TRANSFORM:
                if col in input_df.columns:
                    numeric_col = pd.to_numeric(input_df[col], errors="coerce").fillna(0)
                    numeric_col = numeric_col.clip(lower=0)
                    input_df[col] = np.log1p(numeric_col)

            categorical_cols_in_df = [col for col in CATEGORICAL_COLUMNS if col in input_df.columns]
            if categorical_cols_in_df:
                for col in categorical_cols_in_df:
                    input_df[col] = input_df[col].astype(str)

                for col in categorical_cols_in_df:
                    if col in self.global_categories_:
                        input_df[col] = pd.Categorical(
                            input_df[col], categories=self.global_categories_[col], ordered=False
                        )
                        if input_df[col].isnull().any():
                            page_logger.warn(
                                f"列 '{col}' 在使用全局类别转换后包含 NaN/NaT 值。"
                                "这意味着预测数据中出现了训练时未见过的新类别。"
                            )
                    else:
                        page_logger.warn(
                            f"列 '{col}' 的全局类别未找到。将使用临时的 .astype('category')。"
                        )
                        input_df[col] = input_df[col].astype("category")

            return input_df

        except KeyError as e:
            page_logger.error(
                f"preprocess_input: 输入数据缺少或处理列时发生 KeyError: {e}", exc_info=True
            )
            raise ValueError(f"输入数据缺少或处理列时出错: {e}") from e
        except Exception as e:
            page_logger.error(f"preprocess_input: 预处理输入时发生一般错误: {e}", exc_info=True)
            raise

    def predict_probability(self, input_df: pd.DataFrame) -> float | None:
        if self.model is None:
            return None
        try:
            proba = self.model.predict_proba(input_df)[0][1]
            return float(proba)
        except IndexError:
            page_logger.warn(
                f"predict_probability: predict_proba 可能返回了意外的结构，导致 IndexError。输入数据的形状: {input_df.shape}"
            )
            return None
        except Exception as e:
            page_logger.error(f"predict_probability: 预测概率时发生错误: {e}", exc_info=True)
            return None
