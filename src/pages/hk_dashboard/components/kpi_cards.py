"""KPI metric cards — clean HTML cards with optional accent bars."""

import streamlit as st


def render_kpi_row(cards: list[dict]) -> None:
    """Render a row of custom KPI cards.

    Each dict: {
        "value": str,     # large number
        "label": str,     # small label below
        "delta": str | None,  # optional delta text
        "accent": str | None, # "accent" | "positive" | "warning"
    }
    """
    html = '<div class="hk-kpi-row">'
    for c in cards:
        accent_cls = c.get("accent", "")
        delta_html = ""
        if c.get("delta"):
            color = {"positive": "#059669", "warning": "#d97706"}.get(c.get("delta_color", ""), "#64748b")
            delta_html = f'<div class="kpi-delta" style="color:{color}">{c["delta"]}</div>'
        html += (
            f'<div class="hk-kpi-card {accent_cls}">'
            f'<div class="kpi-value">{c["value"]}</div>'
            f'<div class="kpi-label">{c["label"]}</div>'
            f'{delta_html}'
            f'</div>'
        )
    html += "</div>"
    st.html(html)


def render_metric_grid(metrics: list[dict], columns: int = 4) -> None:
    """Thin wrapper over st.metric with consistent styling."""
    cols = st.columns(min(columns, len(metrics)))
    for i, m in enumerate(metrics):
        with cols[i]:
            delta = m.get("delta")
            color = m.get("delta_color", "normal")
            if delta:
                st.metric(label=m["label"], value=m["value"], delta=delta, delta_color=color)
            else:
                st.metric(label=m["label"], value=m["value"])
