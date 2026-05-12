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


@st.cache_resource(show_spinner=False, scope="global")
def cached_get_prediction_model(model_name):
    return get_prediction_model(model_name)


@dataclass
class machine_learning_model:
    prediction_model: Any
    loaded_feature_names: list[str]
    cases_df: pd.DataFrame
    cases_df_fingerprint: int
    background_universities: set[str]
    target_base_df: pd.DataFrame
    university_country_map: dict[str, str]

    @classmethod
    @st.cache_resource(show_spinner=False, scope="global")
    def resource_loader(cls) -> "machine_learning_model":
        from src.pages.prediction.core.utils import _data_manager
        from src.pages.prediction.input_form_components.target_options_service import (
            build_target_base_df,
        )
        from src.pages.prediction.modeling import validate_model_and_features
        from src.pages.prediction.prediction_preparation import (
            compute_df_fingerprint,
        )
        from src.utils.app_data_loader import (
            load_bg_target_similarity_cache,
            load_school_major_details_df,
        )

        _data_manager.warm_up()
        load_bg_target_similarity_cache()  # warm up for peak-hour users
        model = cached_get_prediction_model("xgboost")
        features = validate_model_and_features(model)
        cases = load_raw_cases_data()
        feature_list = features if features is not None else []

        fingerprint = compute_df_fingerprint(cases)
        bg_unis = (
            set(cases["background_university"].dropna().astype(str).unique())
            if "background_university" in cases.columns
            else set()
        )

        details_df = load_school_major_details_df()
        unique_targets_df = None
        if cases is not None and not cases.empty:
            cols = [c for c in ["target_university", "target_major"] if c in cases.columns]
            if cols:
                unique_targets_df = cases[cols].drop_duplicates()

        target_base_df, uni_country_map = build_target_base_df(unique_targets_df, details_df)

        return cls(
            prediction_model=model,
            loaded_feature_names=feature_list,
            cases_df=cases,
            cases_df_fingerprint=fingerprint,
            background_universities=bg_unis,
            target_base_df=target_base_df,
            university_country_map=uni_country_map,
        )
