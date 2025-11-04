from typing import Any

import streamlit as st
import shap

from src.utils.model_loader import load_model

@st.cache_resource
def get_tree_explainer(
    model_name: str = "xgboost",
) -> Any:

    model, _, _ = load_model(model_name)

    xgb_model = model.base_model
    return shap.TreeExplainer(xgb_model)