"""Reusable KPI metric card grid."""

import streamlit as st


def render_metric_grid(metrics: list[dict], columns: int = 4) -> None:
    """Render a row of st.metric cards.

    Each metric dict: {"label": str, "value": str, "delta": str | None}
    """
    cols = st.columns(min(columns, len(metrics)))
    for i, m in enumerate(metrics):
        with cols[i]:
            delta = m.get("delta")
            color = m.get("delta_color", "normal")
            if delta:
                st.metric(label=m["label"], value=m["value"], delta=delta, delta_color=color)
            else:
                st.metric(label=m["label"], value=m["value"])
