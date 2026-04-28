"""Searchable data table component."""

import pandas as pd
import streamlit as st


def render_filterable_table(
    df: pd.DataFrame,
    key: str = "table",
    column_config: dict | None = None,
    height: int = 400,
    hide_index: bool = True,
) -> None:
    if df.empty:
        st.caption("暂无数据")
        return

    search = st.text_input("搜索", key=f"search_{key}", placeholder="输入关键词...", label_visibility="collapsed")
    if search:
        mask = pd.Series(False, index=df.index)
        for col in df.select_dtypes(include=["object"]).columns:
            mask |= df[col].astype(str).str.contains(search, case=False, na=False)
        df = df[mask]

    st.dataframe(
        df, use_container_width=True, hide_index=hide_index,
        height=height, column_config=column_config,
    )
    st.caption(f"{len(df)} 条记录")
