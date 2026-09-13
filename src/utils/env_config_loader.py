from __future__ import annotations

import streamlit as st

from config.settings import load_app_config as _load_app_config_uncached
from src.utils.config_models import AppConfig


@st.cache_data(show_spinner=False)
def load_app_config(path: str = "config/app_config.json") -> AppConfig:
    del path
    return _load_app_config_uncached()
