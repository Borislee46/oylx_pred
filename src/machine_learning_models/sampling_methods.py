import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC


def apply_sampling(X_train, y_train, sampling_method=None, sample_weight=None):
    if sampling_method is None:
        return X_train, y_train, sample_weight

    n_minority_samples = sum(y_train == 1)
    k_neighbors = max(1, min(5, n_minority_samples - 1))

    categorical_features_indices = []
    if isinstance(X_train, pd.DataFrame):
        categorical_features_indices = [
            i for i, col in enumerate(X_train.columns) 
            if X_train[col].dtype.name == "category"
        ]

    if categorical_features_indices:
        sampler = SMOTENC(
            categorical_features=categorical_features_indices,
            random_state=42,
            k_neighbors=k_neighbors,
            sampling_strategy=1.0,
        )
    else:
        sampler = SMOTE(
            random_state=42,
            k_neighbors=k_neighbors,
            sampling_strategy=1.0,
        )

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    sw_res = _handle_sample_weights(sample_weight, X_train, sampler, y_res)
    
    return X_res, y_res, sw_res


def _handle_sample_weights(original_sw, original_X, sampler, y_resampled):
    if original_sw is None:
        return None

    sw_series = pd.Series(original_sw, index=original_X.index)
    
    if hasattr(sampler, "sample_indices_"):
        base_weights = sw_series.iloc[sampler.sample_indices_].reset_index(drop=True)
    else:
        base_weights = sw_series.reset_index(drop=True)

    if len(base_weights) < len(y_resampled):
        minority_mask = y_resampled == 1
        minority_mean_w = (
            float(base_weights[minority_mask[:len(base_weights)]].mean())
            if minority_mask.any() else 1.0
        )
        n_new = len(y_resampled) - len(base_weights)
        new_weights = pd.Series([minority_mean_w] * n_new)
        return pd.concat([base_weights, new_weights], ignore_index=True)
    
    return base_weights