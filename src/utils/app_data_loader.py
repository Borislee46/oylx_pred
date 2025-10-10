import numpy as np
import pandas as pd
import streamlit as st

from src.machine_learning_models.categorical_features_processor import prepare_categorical_columns
from src.machine_learning_models.data_config import CATEGORICAL_COLUMNS
from src.utils.logger import setup_logger

data_loader_logger = setup_logger("page3", "prediction")


def _read_feather_with_hash_and_fallback(primary_path: str, fallback_path: str) -> pd.DataFrame:
    def _safe_read(path: str, label: str) -> pd.DataFrame | None:
        try:
            return pd.read_feather(path)
        except Exception as e:
            data_loader_logger.error(f"读取失败: {label} ({path})，错误: {e}")
            return None

    primary_df = _safe_read(primary_path, "cases_min")
    if isinstance(primary_df, pd.DataFrame):
        return primary_df

    fallback_df = _safe_read(fallback_path, "cases")
    if isinstance(fallback_df, pd.DataFrame):
        return fallback_df

    data_loader_logger.error("无法读取任何案例数据文件，返回空表")
    return pd.DataFrame()


def _ensure_required_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_defaults: dict[str, object] = {
        "background_university": "",
        "background_major_original": "",
        "background_major": "",
        "target_university": "",
        "target_major": "",
        "admitted": 0,
        "gpa": np.nan,
        "faculty": "",
    }

    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    try:
        if "admitted" in df.columns:
            df["admitted"] = pd.to_numeric(df["admitted"], errors="coerce").fillna(0).astype(int)
    except Exception:
        pass
    try:
        if "gpa" in df.columns:
            df["gpa"] = pd.to_numeric(df["gpa"], errors="coerce")
    except Exception:
        pass

    return df


@st.cache_data
def load_raw_cases_data(
    path: str = "src/machine_learning_models/data/cases_min.feather",
    fallback_path: str = "src/machine_learning_models/data/cases.feather",
):
    df = _read_feather_with_hash_and_fallback(primary_path=path, fallback_path=fallback_path)
    if df.empty:
        return df
    return _ensure_required_case_columns(df)


@st.cache_data
def load_global_categories_dataframe(
    path: str = "src/machine_learning_models/data/cases_min.feather",
    fallback_path: str = "src/machine_learning_models/data/cases.feather",
):
    _cases_df = load_raw_cases_data(path=path, fallback_path=fallback_path)
    if _cases_df.empty:
        return _cases_df
    for col in CATEGORICAL_COLUMNS:
        if col not in _cases_df.columns:
            _cases_df[col] = ""
    _cases_df_prepared = prepare_categorical_columns(_cases_df.copy(), CATEGORICAL_COLUMNS)
    return _cases_df_prepared


@st.cache_data
def load_school_base_data(path="src/machine_learning_models/data/school_base.feather"):
    return pd.read_feather(path)


@st.cache_data
def load_school_major_details_df(
    path="src/machine_learning_models/data/school_major_details.feather",
):
    try:
        return pd.read_feather(path)
    except Exception as e:
        data_loader_logger.error(f"加载学校专业详情数据失败: {e}")
        return pd.DataFrame()


@st.cache_data
def load_bg_target_similarity_cache(path="cache/background_target_similarity.feather"):
    try:
        df = pd.read_feather(path)
        if set(["key", "similarity"]).issubset(set(df.columns)):
            series = df.set_index("key")["similarity"]
            return series.to_dict()
        return {}
    except Exception:
        return {}


@st.cache_data
def load_bg_bg_similarity_cache(path="cache/background_background_similarity.feather"):
    try:
        df = pd.read_feather(path)
        if set(["key", "similarity"]).issubset(set(df.columns)):
            series = df.set_index("key")["similarity"]
            return series.to_dict()
        return {}
    except Exception:
        return {}
