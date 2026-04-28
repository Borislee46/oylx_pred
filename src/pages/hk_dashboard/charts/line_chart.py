"""Altair line chart — clean monthly trend."""

import altair as alt
import pandas as pd
import streamlit as st


def monthly_trend_line(
    df: pd.DataFrame, date_col: str, value_col: str,
    title: str = "", color: str = "#2563eb", height: int = 240,
) -> None:
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        return
    chart = (
        alt.Chart(df)
        .mark_line(point=False, color=color, strokeWidth=1.8)
        .encode(
            x=alt.X(f"{date_col}:T", title=None,
                    axis=alt.Axis(labelAngle=-30, tickCount=6, grid=False, labelFontSize=11)),
            y=alt.Y(f"{value_col}:Q", title=None,
                    axis=alt.Axis(grid=True, gridColor="#f1f5f9", labelFontSize=11, tickCount=5)),
            tooltip=[alt.Tooltip(f"{date_col}:T", title="日期", format="%Y-%m"),
                     alt.Tooltip(f"{value_col}:Q", format=",.0f")],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)
