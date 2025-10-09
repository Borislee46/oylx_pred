import logging

import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def apply_sampling(X_train, y_train, sampling_method=None, sample_weight=None):
    if sampling_method is None:
        if sample_weight is not None:
            return X_train, y_train, sample_weight
        return X_train, y_train, None

    if not isinstance(X_train, pd.DataFrame):
        categorical_features_indices = []
    else:
        categorical_features_indices = [
            i for i, col in enumerate(X_train.columns) if X_train[col].dtype.name == "category"
        ]
        if categorical_features_indices and sampling_method not in ["smote", "adasyn"]:
            pass
        elif not categorical_features_indices and sampling_method == "smotenc":
            pass

    original_X_train, original_y_train = X_train.copy(), y_train.copy()
    if sample_weight is not None:
        original_sw_train = sample_weight.copy()
    else:
        original_sw_train = None

    n_minority_samples = sum(y_train == 1)
    n_majority_samples = sum(y_train == 0)

    try:
        if sampling_method == "adasyn":
            if n_minority_samples < 2:
                sampler = RandomOverSampler(random_state=42, sampling_strategy=1.0)
                X_train, y_train = sampler.fit_resample(original_X_train, original_y_train)
                if original_sw_train is not None:
                    sw_series = pd.Series(original_sw_train, index=original_X_train.index)
                    sample_indices = sampler.sample_indices_
                    sw_train = sw_series.iloc[sample_indices].reset_index(drop=True)
                else:
                    sw_train = None
            else:
                try:
                    sampler = ADASYN(
                        random_state=42,
                        n_neighbors=min(5, n_minority_samples - 1),
                        sampling_strategy=1.0,
                    )
                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                    if original_sw_train is not None:
                        sw_series = pd.Series(original_sw_train, index=original_X_train.index)
                        if hasattr(sampler, "sample_indices_"):
                            sample_indices = sampler.sample_indices_
                            base_weights = sw_series.iloc[sample_indices].reset_index(drop=True)
                        else:
                            base_weights = sw_series.reset_index(drop=True)
                        minority_mask = y_train == 1
                        minority_mean_w = (
                            float(base_weights[minority_mask[: len(base_weights)]].mean())
                            if minority_mask.any()
                            else 1.0
                        )
                        if len(base_weights) < len(y_train):
                            n_new = len(y_train) - len(base_weights)
                            append_w = pd.Series([minority_mean_w] * n_new)
                            sw_train = pd.concat([base_weights, append_w], ignore_index=True)
                        else:
                            sw_train = base_weights
                    else:
                        sw_train = None
                except (RuntimeError, ValueError) as e:
                    sampler = RandomOverSampler(random_state=42, sampling_strategy=1.0)
                    X_train, y_train = sampler.fit_resample(original_X_train, original_y_train)
                    if original_sw_train is not None:
                        sw_series = pd.Series(original_sw_train, index=original_X_train.index)
                        sample_indices = sampler.sample_indices_
                        sw_train = sw_series.iloc[sample_indices].reset_index(drop=True)
                    else:
                        sw_train = None

        elif sampling_method == "smote":
            if n_minority_samples < 2:
                sampler = RandomOverSampler(random_state=42, sampling_strategy=1.0)
                X_train, y_train = sampler.fit_resample(original_X_train, original_y_train)
                if original_sw_train is not None:
                    sw_series = pd.Series(original_sw_train, index=original_X_train.index)
                    sample_indices = sampler.sample_indices_
                    sw_train = sw_series.iloc[sample_indices].reset_index(drop=True)
                else:
                    sw_train = None
            else:
                params_smote = {
                    "random_state": 42,
                    "k_neighbors": min(5, n_minority_samples - 1),
                    "sampling_strategy": 1.0,
                }
                if params_smote["k_neighbors"] < 1:
                    params_smote["k_neighbors"] = 1

                sampler_class = SMOTE
                if categorical_features_indices:
                    sampler_class = SMOTENC
                    params_smote["categorical_features"] = categorical_features_indices
                else:
                    pass

                try:
                    sampler = sampler_class(**params_smote)
                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                    if original_sw_train is not None:
                        sw_series = pd.Series(original_sw_train, index=original_X_train.index)
                        if hasattr(sampler, "sample_indices_"):
                            sample_indices = sampler.sample_indices_
                            base_weights = sw_series.iloc[sample_indices].reset_index(drop=True)
                        else:
                            base_weights = sw_series.reset_index(drop=True)
                        minority_mask = y_train == 1
                        minority_mean_w = (
                            float(base_weights[minority_mask[: len(base_weights)]].mean())
                            if minority_mask.any()
                            else 1.0
                        )
                        if len(base_weights) < len(y_train):
                            n_new = len(y_train) - len(base_weights)
                            append_w = pd.Series([minority_mean_w] * n_new)
                            sw_train = pd.concat([base_weights, append_w], ignore_index=True)
                        else:
                            sw_train = base_weights
                    else:
                        sw_train = None
                except (RuntimeError, ValueError) as e:
                    sampler = RandomOverSampler(random_state=42, sampling_strategy=1.0)
                    X_train, y_train = sampler.fit_resample(original_X_train, original_y_train)
                    if original_sw_train is not None:
                        sw_series = pd.Series(original_sw_train, index=original_X_train.index)
                        sample_indices = sampler.sample_indices_
                        sw_train = sw_series.iloc[sample_indices].reset_index(drop=True)
                    else:
                        sw_train = None

        elif sampling_method == "random_over":
            sampler = RandomOverSampler(
                random_state=42,
                sampling_strategy=1.0,
            )
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            if original_sw_train is not None:
                sw_series = pd.Series(original_sw_train, index=original_X_train.index)
                sample_indices = sampler.sample_indices_
                sw_train = sw_series.iloc[sample_indices].reset_index(drop=True)
            else:
                sw_train = None

        elif sampling_method == "random_under":
            if n_majority_samples > n_minority_samples * 1.5:
                sampler = RandomUnderSampler(
                    random_state=42,
                    sampling_strategy="majority",
                )
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                if original_sw_train is not None:
                    sw_series = pd.Series(original_sw_train, index=original_X_train.index)
                    sample_indices = sampler.sample_indices_
                    sw_train = sw_series.iloc[sample_indices].reset_index(drop=True)
                else:
                    sw_train = None
            else:
                sw_train = sample_weight if sample_weight is not None else None
        else:
            sw_train = sample_weight if sample_weight is not None else None

    except Exception as e:
        logging.error(f"报错: {e}. 返回原始数据.")
        X_train, y_train = original_X_train, original_y_train
        sw_train = original_sw_train

    return X_train, y_train, sw_train
