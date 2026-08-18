from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_school_base_data(path: str = "src/ml/data/school_base.feather") -> pd.DataFrame:
    return pd.read_feather(path)
