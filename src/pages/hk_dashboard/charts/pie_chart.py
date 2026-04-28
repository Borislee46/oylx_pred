"""Altair donut chart — clean categorical breakdown."""

import altair as alt
import pandas as pd
import streamlit as st

PALETTE = ["#2563eb", "#0d9488", "#d97706", "#7c3aed", "#64748b", "#94a3b8", "#cbd5e1"]


def donut_chart(
    df: pd.DataFrame, category_col: str, value_col: str = "count",
    title: str = "", max_categories: int = 6, height: int = 260,
) -> None:
    if df.empty or category_col not in df.columns:
        return
    if value_col not in df.columns:
        cnt = df.groupby(category_col).size().reset_index(name=value_col)
    else:
        cnt = df.copy()
    cnt = cnt.sort_values(value_col, ascending=False)
    if len(cnt) > max_categories:
        top = cnt.head(max_categories)
        other = pd.DataFrame([{category_col: "其他", value_col: cnt.iloc[max_categories:][value_col].sum()}])
        cnt = pd.concat([top, other], ignore_index=True)

    chart = (
        alt.Chart(cnt)
        .mark_arc(innerRadius=45, outerRadius=90, padAngle=1)
        .encode(
            theta=alt.Theta(f"{value_col}:Q", stack=True),
            color=alt.Color(f"{category_col}:N", title=None, scale=alt.Scale(range=PALETTE)),
            tooltip=[alt.Tooltip(f"{category_col}:N"), alt.Tooltip(f"{value_col}:Q", format=",.0f")],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)
