import json
import os

import streamlit as st


def load_config_data():
    config_file = "data/site_config.json"
    try:
        if os.path.exists(config_file):
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)
                return config
        else:
            return {}
    except Exception as e:
        st.error(f"加载配置文件出错: {e}")
        return {}
