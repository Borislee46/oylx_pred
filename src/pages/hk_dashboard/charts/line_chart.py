"""Altair line chart — clean monthly trend with formatted tooltips."""

import altair as alt
import pandas as pd
import streamlit as st


def _fmt_dt_cn(dt_val) -> str:
    """Convert datetime to Chinese month string: 2026年1月."""
    try:
        return f"{dt_val.year}年{dt_val.month}月"
    except Exception:
        return str(dt_val)


def monthly_trend_line(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "",
    color: str = "#2563eb",
    height: int = 240,
    currency: bool = False,
) -> None:
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        return
    fmt = ",.0f"
    tooltip_title = value_col
    df = df.copy()
    if currency:
        df[value_col] = df[value_col] / 1e4
        tooltip_title = f"{value_col} (万)"
        fmt = ",.1f"

    # Convert date_col to Chinese month string for display
    label_col = f"{date_col}_label"
    df[label_col] = pd.to_datetime(df[date_col], errors="coerce").apply(_fmt_dt_cn)
    # Preserve sort order using the original datetime
    df = df.sort_values(date_col)
    label_order = df[label_col].tolist()

    chart = (
        alt.Chart(df)
        .mark_line(point=False, color=color, strokeWidth=1.8)
        .encode(
            x=alt.X(
                f"{label_col}:N",
                title=None,
                sort=label_order,
                axis=alt.Axis(labelAngle=-30, tickCount=6, grid=False, labelFontSize=11),
            ),
            y=alt.Y(
                f"{value_col}:Q",
                title=None,
                axis=alt.Axis(grid=True, gridColor="#f1f5f9", labelFontSize=11, tickCount=5),
            ),
            tooltip=[
                alt.Tooltip(f"{label_col}:N", title="月份"),
                alt.Tooltip(f"{value_col}:Q", title=tooltip_title, format=fmt),
            ],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")
