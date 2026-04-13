import json
import os

import streamlit as st

from src.utils.config_local_paths import resolve_app_config_path


@st.cache_data(show_spinner=False)
def _load_app_config_cached(resolved_path: str):
    with open(resolved_path, encoding="utf-8") as f:
        all_configs = json.load(f)

    return all_configs[os.environ.get("APP_ENV", "test")]


def load_app_config(path: str | None = None):
    resolved = path if path is not None else resolve_app_config_path()
    return _load_app_config_cached(resolved)


def clear_load_app_config_cache() -> None:
    _load_app_config_cached.clear()
