"""Altair bar chart helpers — simple, grouped, horizontal."""

import altair as alt
import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.config import PRIMARY, ACCENT


def simple_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    color: str = PRIMARY,
    horizontal: bool = False,
    height: int = 280,
    sort_desc: bool = True,
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    if horizontal:
        x_enc = alt.X(f"{y_col}:Q", title=None, axis=alt.Axis(grid=True, gridColor="#e5e7eb"))
        y_enc = alt.Y(f"{x_col}:N", title=None, sort="-x" if sort_desc else "x")
    else:
        x_enc = alt.X(f"{x_col}:N", title=None, axis=alt.Axis(labelAngle=-30),
                      sort="-y" if sort_desc else "x")
        y_enc = alt.Y(f"{y_col}:Q", title=None, axis=alt.Axis(grid=True, gridColor="#e5e7eb"))
    chart = (
        alt.Chart(df)
        .mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(x=x_enc, y=y_enc,
                tooltip=[alt.Tooltip(f"{x_col}:N"), alt.Tooltip(f"{y_col}:Q", format=",.0f")])
        .properties(height=height)
    )
    if title:
        st.caption(title)
    st.altair_chart(chart, use_container_width=True)


def grouped_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str = "",
    height: int = 300,
) -> None:
    if df.empty:
        return
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X(f"{x_col}:N", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y(f"{y_col}:Q", title=None, axis=alt.Axis(grid=True, gridColor="#e5e7eb")),
            color=alt.Color(f"{color_col}:N", title=None,
                            scale=alt.Scale(range=[PRIMARY, ACCENT, "#f7ab00", "#73c0de", "#a9a9a9"])),
            xOffset=f"{color_col}:N",
            tooltip=[alt.Tooltip(f"{x_col}:N"), alt.Tooltip(f"{color_col}:N"),
                     alt.Tooltip(f"{y_col}:Q", format=",.0f")],
        )
        .properties(height=height)
    )
    if title:
        st.caption(title)
    st.altair_chart(chart, use_container_width=True)
