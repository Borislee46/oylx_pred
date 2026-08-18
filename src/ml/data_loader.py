import pandas as pd
from sklearn.model_selection import train_test_split

from .data_config import (
    HELD_OUT_FEATHER_PATH,
    HELD_OUT_META_PATH,
    RECENT_SAMPLE_BOOST_COUNT,
    RECENT_SAMPLE_BOOST_WEIGHT,
    TARGET_COLUMN,
    TEST_SIZE,
    TEXT_COLUMNS,
    TEXT_EMPTY_SAMPLE_WEIGHT,
    USE_HELD_OUT_IF_AVAILABLE,
)
from .feature_engineer import FeatureEngineer
from .school_level_mapper import build_school_level_fallback_mapping


def load_data(data_path, split="random"):
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        sample_weight_train,
        level_fallback_mapping,
        feature_engineer_state,
    ) = load_and_preprocess_data(data_path, split=split)
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


def load_and_preprocess_data(data_path, split="random"):
    try:
        data = pd.read_feather(data_path)
    except Exception as e:
        raise FileNotFoundError(f"加载数据文件失败 {data_path}: {e}") from e

    if data is None or data.empty:
        raise ValueError(f"数据文件为空: {data_path}")

    level_fallback_mapping, _ = build_school_level_fallback_mapping(data)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"目标列 '{TARGET_COLUMN}' 未找到，请检查原始数据。")

    if data[TARGET_COLUMN].isnull().any():
        raise ValueError(f"目标列 '{TARGET_COLUMN}' 中存在 NaN 值，请检查数据。")

    X = data.drop(columns=[TARGET_COLUMN], errors="ignore")
    y = data[TARGET_COLUMN]

    if split == "time":
        X_train_raw, X_test_raw, y_train, y_test, sw_train = _split_by_held_out(data, X, y)
    else:
        X_train_raw, X_test_raw, y_train, y_test, sw_train = _split_random(data, X, y)

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


def _split_random(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple:
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
    return X_train_raw, X_test_raw, y_train, y_test, sw_train


def _split_by_held_out(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple:
    import os

    from .held_out import _PROJECT_ROOT, load_held_out_set

    held_out_path = os.path.join(_PROJECT_ROOT, HELD_OUT_FEATHER_PATH)
    meta_path = os.path.join(_PROJECT_ROOT, HELD_OUT_META_PATH)

    held_out_df, meta = load_held_out_set(
        feather_path=held_out_path,
        meta_path=meta_path,
        verify_integrity=USE_HELD_OUT_IF_AVAILABLE,
    )

    n_train_meta = meta.get("n_train", 0)
    n_test_meta = meta.get("n_test", 0)
    print(
        f"[data_loader] split=time (held-out) | "
        f"总样本: {len(data):,} | "
        f"训练: {n_train_meta:,} | "
        f"测试: {n_test_meta:,} | "
        f"测试正例率: {meta.get('positive_rate_test', 'N/A')}"
    )

    held_out_indices = set(held_out_df.index)
    all_indices = set(data.index)

    if not held_out_indices.issubset(all_indices):
        missing = len(held_out_indices - all_indices)
        raise ValueError(
            f"留存集索引不匹配: {missing} 个留存集行不在全量数据中。"
            f"held_out_test.feather 可能与 cases.feather 不同步——请重新运行 "
            f"python -m src.ml.held_out --freeze"
        )

    train_indices = sorted(all_indices - held_out_indices)
    test_indices = sorted(held_out_indices)

    X_train_raw = X.loc[train_indices].copy()
    X_test_raw = X.loc[test_indices].copy()
    y_train = y.loc[train_indices].copy()
    y_test = y.loc[test_indices].copy()

    def is_all_text_empty(row):
        empties = []
        for col in TEXT_COLUMNS:
            if col in data.columns:
                val = row.get(col, None)
                empties.append((val is None) or (isinstance(val, str) and val.strip() == ""))
        return all(empties) if empties else False

    sw_train = X_train_raw.apply(is_all_text_empty, axis=1).astype(float)
    sw_train = sw_train.replace({1.0: TEXT_EMPTY_SAMPLE_WEIGHT, 0.0: 1.0})

    print(f"[data_loader] 训练正例率: {y_train.mean():.4f}, 测试正例率: {y_test.mean():.4f}")

    return X_train_raw, X_test_raw, y_train, y_test, sw_train
