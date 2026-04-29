"""HK Dashboard — cached CSV data loaders."""

import os

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.config import FILES

ENCODING = "utf-8-sig"


def _safe_read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.warning(f"数据文件不存在: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding=ENCODING, dtype=str)
    except Exception as e:
        st.warning(f"读取失败 {os.path.basename(path)}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_kehu_ziyuan() -> pd.DataFrame:
    return _safe_read(FILES["kehu_ziyuan"])


@st.cache_data(ttl=600, show_spinner=False)
def load_tmk_data() -> pd.DataFrame:
    df = _safe_read(FILES["tmk"])
    if not df.empty:
        # Normalize: TMK CSV uses uppercase ID, other tables use lowercase
        rename = {}
        if "资源ID" in df.columns and "资源id" not in df.columns:
            rename["资源ID"] = "资源id"
        if "工单ID" in df.columns and "工单id" not in df.columns:
            rename["工单ID"] = "工单id"
        if rename:
            df = df.rename(columns=rename)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_qianyue_data() -> pd.DataFrame:
    return _safe_read(FILES["qianyue"])


@st.cache_data(ttl=600, show_spinner=False)
def load_class_master() -> pd.DataFrame:
    return _safe_read(FILES["class_master"])


@st.cache_data(ttl=600, show_spinner=False)
def load_roster() -> pd.DataFrame:
    return _safe_read(FILES["roster"])


@st.cache_data(ttl=600, show_spinner=False)
def load_revenue() -> pd.DataFrame:
    return _safe_read(FILES["revenue"])


@st.cache_data(ttl=600, show_spinner=False)
def load_deferred_revenue() -> pd.DataFrame:
    return _safe_read(FILES["deferred_revenue"])


def load_all_data() -> dict[str, pd.DataFrame]:
    """Load all 7 CSVs. Returns dict keyed by English table name."""
    return {
        "kehu_ziyuan": load_kehu_ziyuan(),
        "tmk": load_tmk_data(),
        "qianyue": load_qianyue_data(),
        "class_master": load_class_master(),
        "roster": load_roster(),
        "revenue": load_revenue(),
        "deferred_revenue": load_deferred_revenue(),
    }
