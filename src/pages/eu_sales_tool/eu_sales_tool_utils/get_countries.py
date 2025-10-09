import streamlit as st

from src.pages.eu_sales_tool.eu_sales_tool_utils.load_config import load_config_data
from src.pages.eu_sales_tool.eu_sales_tool_utils.load_data import load_education_data


def get_standardized_countries():
    config = st.session_state.get("site_config", load_config_data())
    education_data = st.session_state.get("education_data", load_education_data())

    standardized = {}
    country_descriptions = config.get("country_descriptions", {})
    country_display_names = config.get("country_display_names", {})

    for country_key, country_data in education_data.items():
        standardized[country_key] = {
            "display_name": country_display_names.get(country_key, country_key),
            "description": country_descriptions.get(country_key, "暂无描述"),
            "languages": country_data.get("languages", []),
            "study_programs": country_data.get("study_programs", []),
            "descriptions": country_data.get("descriptions", []),
            "requirements": country_data.get("requirements", []),
            "cost_estimation": country_data.get("cost_estimation", {}),
            "consultant_tips": country_data.get("consultant_tips", {}),
        }

    return standardized
