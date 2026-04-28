"""Altair line chart — monthly trend with optional formatting."""

import altair as alt
import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.config import PRIMARY, TEXT_MID


def monthly_trend_line(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "",
    color: str = PRIMARY,
    height: int = 280,
) -> None:
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        return
    chart = (
        alt.Chart(df)
        .mark_line(point=False, color=color, strokeWidth=2)
        .encode(
            x=alt.X(f"{date_col}:T", title=None, axis=alt.Axis(labelAngle=-45, tickCount=8)),
            y=alt.Y(f"{value_col}:Q", title=None, axis=alt.Axis(grid=True, gridColor="#e5e7eb")),
            tooltip=[alt.Tooltip(f"{date_col}:T", title="日期", format="%Y-%m"),
                     alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.0f")],
        )
        .properties(height=height)
    )
    if title:
        st.caption(title)
    st.altair_chart(chart, use_container_width=True)
