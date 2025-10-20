import pandas as pd
import streamlit as st

from src.machine_learning_models.categorical_features_processor import (
    prepare_categorical_columns,
)
from src.machine_learning_models.data_config import CATEGORICAL_COLUMNS
from src.utils.logger import setup_logger

data_loader_logger = setup_logger("page3", "prediction")


@st.cache_data
def load_raw_cases_data(
    path: str = "src/machine_learning_models/data/cases_min.feather",
):
    df = pd.read_feather(path)
    return df


@st.cache_data
def load_global_categories_dataframe(
    path: str = "src/machine_learning_models/data/cases_min.feather",
):
    _cases_df = load_raw_cases_data(path=path)
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
    return pd.read_feather(path)


def _load_similarity_cache(path: str) -> dict:
    df = pd.read_feather(path)
    if {"key", "similarity"}.issubset(df.columns):
        series = df.set_index("key")["similarity"]
        return series.to_dict()
    return {}


@st.cache_data
def load_bg_target_similarity_cache(path="cache/background_target_similarity.feather"):
    return _load_similarity_cache(path)
