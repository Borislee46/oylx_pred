from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.modeling.model import PredictionModel
from src.utils.app_data_loader import (
    load_global_categories_dataframe,
    load_raw_cases_data,
)


def get_prediction_model(model_name):
    global_categories_df_instance = load_global_categories_dataframe()
    model_instance = PredictionModel(model_name, global_categories_df=global_categories_df_instance)
    return model_instance


@st.cache_resource(show_spinner=False)
def cached_get_prediction_model(model_name):
    return get_prediction_model(model_name)


@dataclass
class machine_learning_model:
    prediction_model: Any
    loaded_feature_names: list[str]
    cases_df: pd.DataFrame

    @classmethod
    @st.cache_resource(show_spinner=False)
    def resource_loader(cls) -> "machine_learning_model":
        from src.pages.prediction.modeling.validator import validate_model_and_features

        model = cached_get_prediction_model("xgboost")
        features = validate_model_and_features(model)
        cases = load_raw_cases_data()
        feature_list = features if features is not None else []
        return cls(prediction_model=model, loaded_feature_names=feature_list, cases_df=cases)
