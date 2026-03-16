import pandas as pd
from data_config import (
    RECENT_SAMPLE_BOOST_COUNT,
    RECENT_SAMPLE_BOOST_WEIGHT,
    TARGET_COLUMN,
    TEST_SIZE,
    TEXT_COLUMNS,
    TEXT_EMPTY_SAMPLE_WEIGHT,
)
from feature_engineer import FeatureEngineer
from school_level_mapper import build_school_level_fallback_mapping
from sklearn.model_selection import train_test_split


def load_data(data_path):
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        sample_weight_train,
        level_fallback_mapping,
        feature_engineer_state,
    ) = load_and_preprocess_data(data_path)
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        sample_weight_train,
        level_fallback_mapping,
        feature_engineer_state,
    )


def load_and_preprocess_data(data_path):
    try:
        data = pd.read_feather(data_path)
    except Exception as e:
        raise FileNotFoundError(f"加载数据文件失败 {data_path}: {e}")

    if data is None or data.empty:
        raise ValueError(f"数据文件为空: {data_path}")

    level_fallback_mapping = build_school_level_fallback_mapping(data)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"目标列 '{TARGET_COLUMN}' 未找到，请检查原始数据。")

    if data[TARGET_COLUMN].isnull().any():
        raise ValueError(f"目标列 '{TARGET_COLUMN}' 中存在 NaN 值，请检查数据。")

    X = data.drop(columns=[TARGET_COLUMN], errors="ignore")
    y = data[TARGET_COLUMN]

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
    feature_engineer_state = fe.get_state()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        sw_train,
        level_fallback_mapping,
        feature_engineer_state,
    )
