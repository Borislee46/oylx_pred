import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="加载学校数据...")
def load_full_cases(path: str = "src/machine_learning_models/data/cases.feather") -> pd.DataFrame:
    df = pd.read_feather(path)
    ielts_norm = df["ielts"].fillna(0) / 9.0
    toefl_norm = df["toefl"].fillna(0) / 120.0
    df["language_score"] = np.maximum(ielts_norm, toefl_norm)
    df.loc[(df["ielts"].isna()) & (df["toefl"].isna()), "language_score"] = np.nan
    return df


FEATURE_LABELS = {
    "gpa": "GPA",
    "language_score": "语言成绩",
    "research_count": "科研经历",
    "internship_count": "实习经历",
    "paper_count": "论文发表",
    "award_count": "获奖经历",
}

PROFILE_FEATURES = ["gpa", "language_score", "research_count", "internship_count", "paper_count", "award_count"]
