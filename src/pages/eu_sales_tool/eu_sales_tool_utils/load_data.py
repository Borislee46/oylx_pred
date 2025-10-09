import json

import streamlit as st


@st.cache_data
def load_education_data():
    try:
        with open("data/detailed_europe_data.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("数据文件未找到，请先运行数据处理脚本")
        return {}
    except Exception as e:
        st.error(f"加载数据文件出错: {e}")
        return {}
