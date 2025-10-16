import streamlit as st

from src.pages.prediction.prediction_model import PredictionModel
from src.utils.app_data_loader import (
    load_bg_target_similarity_cache,
    load_global_categories_dataframe,
    load_raw_cases_data,
)
from src.utils.logger import setup_logger

data_loader_logger = setup_logger("page3", "prediction")


def get_prediction_model(model_name):
    global_categories_df_instance = load_global_categories_dataframe()

    if global_categories_df_instance is None:
        return None

    model_instance = PredictionModel(model_name, global_categories_df=global_categories_df_instance)
    return model_instance


def load_cases_data():
    cases_df = load_raw_cases_data()
    if cases_df is None:
        data_loader_logger.error("无法加载案例数据文件")
    return cases_df


@st.cache_data
def cached_load_cases_data():
    return load_cases_data()


@st.cache_resource
def cached_get_prediction_model(model_name):
    return get_prediction_model(model_name)


@st.cache_data
def cached_load_bg_target_similarity_cache():
    return load_bg_target_similarity_cache()
