import json
import os
from typing import Any

import streamlit as st


@st.cache_data
def load_all_country_data() -> dict[str, Any]:
    data_dir = "src/pages/eu_sales_tool/data"
    country_data = {}

    if not os.path.exists(data_dir):
        st.error(f"数据目录不存在: {data_dir}")
        return {}

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                    path_data = data.get("undergraduate_path", {})
                    country_name = path_data.get("country_name")
                    if country_name:
                        country_data[country_name] = path_data
            except (OSError, json.JSONDecodeError) as e:
                st.error(f"加载或解析文件 {filename} 时出错: {e}")
            except Exception as e:
                st.error(f"处理文件 {filename} 时发生未知错误: {e}")

    return country_data


@st.cache_data
def get_country_data():
    return load_all_country_data()
