import json
import os

import streamlit as st


@st.cache_data(show_spinner=False)
def load_auth_config(path="auth_config.json"):
    if os.path.exists(path):
        with open(path) as f:
            config = json.load(f)
            config.setdefault("EMAIL_WHITELIST", [])
            config.setdefault("ADMIN_EMAILS", [])
            config.setdefault("MODULE_PERMISSIONS", {})
            return config
    return {
        "EMAIL_WHITELIST": [],
        "ADMIN_EMAILS": [],
        "MODULE_PERMISSIONS": {},
    }
