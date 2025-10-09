import json
import os

import streamlit as st


@st.cache_data
def load_app_config(path="config/app_config.json"):
    with open(path, encoding="utf-8") as f:
        all_configs = json.load(f)

    return all_configs[os.environ.get("APP_ENV", "test")]
