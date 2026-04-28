"""Altair donut / ring chart for categorical breakdowns."""

import altair as alt
import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.config import PALETTE


def donut_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str = "count",
    title: str = "",
    max_categories: int = 7,
    height: int = 300,
) -> None:
    if df.empty or category_col not in df.columns:
        return
    # Roll up small slices
    if value_col not in df.columns:
        cnt = df.groupby(category_col).size().reset_index(name=value_col)
    else:
        cnt = df.copy()
    cnt = cnt.sort_values(value_col, ascending=False)
    if len(cnt) > max_categories:
        top = cnt.head(max_categories)
        other_sum = cnt.iloc[max_categories:][value_col].sum()
        other_row = pd.DataFrame([{category_col: "其他", value_col: other_sum}])
        cnt = pd.concat([top, other_row], ignore_index=True)

    chart = (
        alt.Chart(cnt)
        .mark_arc(innerRadius=50, outerRadius=100)
        .encode(
            theta=alt.Theta(f"{value_col}:Q", stack=True),
            color=alt.Color(f"{category_col}:N", title=None, scale=alt.Scale(range=PALETTE)),
            tooltip=[alt.Tooltip(f"{category_col}:N"), alt.Tooltip(f"{value_col}:Q", format=",.0f")],
        )
        .properties(height=height)
    )
    if title:
        st.caption(title)
    st.altair_chart(chart, use_container_width=True)
