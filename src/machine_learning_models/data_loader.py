import pandas as pd
from data_config import (
    RECENT_SAMPLE_BOOST_COUNT,
    RECENT_SAMPLE_BOOST_WEIGHT,
    TARGET_COLUMN,
    TEST_SIZE,
    TEXT_COLUMNS,
    TEXT_EMPTY_SAMPLE_WEIGHT,
)
from feature_engineer import FeatureEngineer, engineer_features
from sampling_methods import apply_sampling
from sklearn.model_selection import train_test_split


def load_data(data_path, sampling_method=None):
    X_train, X_test, y_train, y_test, feature_names, sample_weight_train = load_and_preprocess_data(
        data_path, sampling_method
    )
    return X_train, X_test, y_train, y_test, feature_names, sample_weight_train


def load_and_preprocess_data(data_path, sampling_method=None):
    data = None
    try:
        data = pd.read_feather(data_path)
    except Exception:
        data = None

    data = engineer_features(data)

    X = data.drop(columns=[TARGET_COLUMN], errors="ignore")
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"目标列 '{TARGET_COLUMN}' 在特征工程处理后未找到，请检查数据和特征工程步骤。"
        )
    y = data[TARGET_COLUMN]
    if y.isnull().any():
        raise ValueError(f"目标列 '{TARGET_COLUMN}' 中存在 NaN 值，请检查数据。")

    def is_all_text_empty(row):
        empties = []
        for col in TEXT_COLUMNS:
            if col in data.columns:
                val = row.get(col, None)
                empties.append((val is None) or (isinstance(val, str) and val.strip() == ""))
        return all(empties) if empties else False

    sample_weight = X.apply(is_all_text_empty, axis=1).astype(float)
    sample_weight = sample_weight.replace({1.0: TEXT_EMPTY_SAMPLE_WEIGHT, 0.0: 1.0})

    n_rows = len(sample_weight)
    if n_rows > 0 and RECENT_SAMPLE_BOOST_COUNT and RECENT_SAMPLE_BOOST_WEIGHT:
        boost_n = min(RECENT_SAMPLE_BOOST_COUNT, n_rows)
        recent_indices = sample_weight.index[-boost_n:]
        sample_weight.loc[recent_indices] = (
            sample_weight.loc[recent_indices] * RECENT_SAMPLE_BOOST_WEIGHT
        )

    X_train_raw, X_test_raw, y_train, y_test, sw_train, _ = train_test_split(
        X, y, sample_weight, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    fe = FeatureEngineer()
    X_train = fe.fit_transform(X_train_raw)
    X_test = fe.transform(X_test_raw)
    feature_names = X_train.columns.tolist()

    if sampling_method is not None:
        X_train_res, y_train_res, sw_train_res = apply_sampling(
            X_train, y_train, sampling_method, sample_weight=sw_train
        )
        X_train, y_train, sw_train = X_train_res, y_train_res, sw_train_res

    return X_train, X_test, y_train, y_test, feature_names, sw_train
