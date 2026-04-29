"""Tab-level filter helpers."""

import pandas as pd
import streamlit as st


def month_filter(
    months: list[str],
    key: str = "month",
    label: str = "月份",
    format_func=None,
    default_index: int = 0,
) -> str | None:
    """Single-select month dropdown. `months` should be sorted list of 'YYYY-MM' strings."""
    if not months:
        return None
    idx = max(0, min(default_index, len(months) - 1))
    return st.selectbox(label, options=months, index=idx, key=key, format_func=format_func)


def category_filter(
    df: pd.DataFrame,
    column: str,
    label: str = "筛选",
    key: str = "cat_filter",
) -> str | None:
    """Single-select dropdown from a DataFrame column."""
    vals = sorted(df[column].dropna().unique().tolist())
    if not vals:
        return None
    selected = st.selectbox(label, options=["全部"] + vals, key=key)
    return None if selected == "全部" else selected


def multi_category_filter(
    df: pd.DataFrame,
    column: str,
    label: str = "多选筛选",
    key: str = "multi_cat_filter",
) -> list[str]:
    vals = sorted(df[column].dropna().unique().tolist())
    if not vals:
        return []
    return st.multiselect(label, options=vals, key=key)
