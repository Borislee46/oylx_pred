from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.project_root import get_project_root


@st.cache_data(show_spinner=False)
def load_school_base_data(path: str = "src/ml/data/school_base.feather") -> pd.DataFrame:
    if not Path(path).is_absolute():
        path = str(get_project_root() / path)
    return pd.read_feather(path)
