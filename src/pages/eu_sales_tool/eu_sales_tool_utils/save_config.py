import json
import os

import streamlit as st


def save_config_data(config):
    config_file = "data/site_config.json"
    try:
        os.makedirs("data", exist_ok=True)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存配置文件出错: {e}")
        return False


def save_education_data(education_data):
    data_file = "data/detailed_europe_data.json"
    try:
        os.makedirs("data", exist_ok=True)
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(education_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存教育数据文件出错: {e}")
        return False
