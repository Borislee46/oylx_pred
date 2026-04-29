"""Altair bar charts — clean styling with formatted tooltips."""

import altair as alt
import streamlit as st


def simple_bar(
    df,
    x_col: str,
    y_col: str,
    title: str = "",
    color: str = "#2563eb",
    horizontal: bool = False,
    height: int = 240,
    sort_desc: bool = True,
    fmt: str = ",.0f",
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    if horizontal:
        x = alt.X(
            f"{y_col}:Q",
            title=None,
            axis=alt.Axis(grid=False, labelFontSize=11, tickCount=4, format=fmt),
        )
        y = alt.Y(
            f"{x_col}:N",
            title=None,
            sort="-x" if sort_desc else "x",
            axis=alt.Axis(labelFontSize=11, labelLimit=100),
        )
    else:
        x = alt.X(
            f"{x_col}:N",
            title=None,
            sort="-y" if sort_desc else "x",
            axis=alt.Axis(labelAngle=-25, labelFontSize=10, labelLimit=100),
        )
        y = alt.Y(
            f"{y_col}:Q",
            title=None,
            axis=alt.Axis(
                grid=True, gridColor="#f1f5f9", labelFontSize=11, tickCount=4, format=fmt
            ),
        )

    chart = (
        alt.Chart(df)
        .mark_bar(
            color=color,
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3,
            size=22 if not horizontal else 14,
        )
        .encode(
            x=x, y=y, tooltip=[alt.Tooltip(f"{x_col}:N"), alt.Tooltip(f"{y_col}:Q", format=fmt)]
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")


def grouped_bar(
    df,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str = "",
    height: int = 260,
) -> None:
    if df.empty:
        return
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, size=18)
        .encode(
            x=alt.X(f"{x_col}:N", title=None, axis=alt.Axis(labelAngle=-25, labelFontSize=10)),
            y=alt.Y(
                f"{y_col}:Q",
                title=None,
                axis=alt.Axis(
                    grid=True, gridColor="#f1f5f9", labelFontSize=11, tickCount=4, format=",.0f"
                ),
            ),
            color=alt.Color(
                f"{color_col}:N",
                title=None,
                scale=alt.Scale(range=["#2563eb", "#0d9488", "#d97706", "#7c3aed", "#94a3b8"]),
            ),
            xOffset=f"{color_col}:N",
            tooltip=[
                alt.Tooltip(f"{x_col}:N"),
                alt.Tooltip(f"{color_col}:N"),
                alt.Tooltip(f"{y_col}:Q", format=",.0f"),
            ],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")
