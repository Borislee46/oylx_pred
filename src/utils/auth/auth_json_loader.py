import json
import os

import streamlit as st


@st.cache_data
def load_auth_config(path="auth_config.json"):
    if os.path.exists(path):
        with open(path) as f:
            config = json.load(f)
            config.setdefault("EMAIL_WHITELIST", [])
            config.setdefault("EMAIL_BLACKLIST", [])
            config.setdefault("ADMIN_EMAILS", [])
            config.setdefault("MODULE_PERMISSIONS", {})
            return config
    return {
        "EMAIL_WHITELIST": [],
        "EMAIL_BLACKLIST": [],
        "ADMIN_EMAILS": [],
        "MODULE_PERMISSIONS": {},
    }
