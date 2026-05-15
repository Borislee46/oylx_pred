import pandas as pd
import streamlit as st

from src.utils.logger import setup_logger

data_loader_logger = setup_logger("page3", "prediction")


def _prepare_categorical_columns(df, columns):
    for col in columns:
        series = df[col]
        if not pd.api.types.is_categorical_dtype(series):
            df[col] = series.astype("category")
    return df


@st.cache_data(show_spinner=False)
def load_raw_cases_data(
    path: str = "src/machine_learning_models/data/cases.feather",
):
    import numpy as np

    df = pd.read_feather(path)
    has_ielts = "ielts" in df.columns
    has_toefl = "toefl" in df.columns
    if has_ielts and has_toefl:
        ielts = df["ielts"]
        toefl = df["toefl"]
        df["language_score"] = np.maximum(ielts.fillna(0) / 9.0, toefl.fillna(0) / 120.0)
        both_missing = ielts.isna() & toefl.isna()
        if both_missing.any():
            df.loc[both_missing, "language_score"] = np.nan
    elif has_ielts:
        df["language_score"] = df["ielts"].fillna(0) / 9.0
    elif has_toefl:
        df["language_score"] = df["toefl"].fillna(0) / 120.0
    else:
        df["language_score"] = np.nan
    return df


@st.cache_data(show_spinner=False)
def load_global_categories_dataframe(
    path: str = "src/machine_learning_models/data/cases.feather",
):
    _cases_df = load_raw_cases_data(path=path)
    if _cases_df.empty:
        return _cases_df
    for col in [
        "target_university",
        "target_major",
        "background_university",
        "background_major",
    ]:
        if col not in _cases_df.columns:
            _cases_df[col] = ""
    _cases_df_prepared = _prepare_categorical_columns(
        _cases_df.copy(),
        [
            "target_university",
            "target_major",
            "background_university",
            "background_major",
        ],
    )
    return _cases_df_prepared


@st.cache_data(show_spinner=False)
def load_school_base_data(path="src/machine_learning_models/data/school_base.feather"):
    return pd.read_feather(path)


@st.cache_data(show_spinner=False)
def load_school_major_details_df(
    path="src/machine_learning_models/data/school_major_details.feather",
):
    return pd.read_feather(path)


def _load_similarity_cache(path: str):
    import os

    if not os.path.exists(path):
        return {}
    df = pd.read_feather(path)
    if {"bg_major", "target_major", "similarity"}.issubset(df.columns):
        return df.set_index(["bg_major", "target_major"])["similarity"].to_dict()
    return {}


@st.cache_data(show_spinner=False)
def load_bg_target_similarity_cache(path="cache/background_target_similarity.feather"):
    return _load_similarity_cache(path)
