"""KPI metric cards — premium HTML cards with color accent bars."""

import streamlit as st


def render_kpi_row(cards: list[dict]) -> None:
    """Render a row of KPI cards.

    Each dict: {"value": str, "label": str, "accent": "blue"|"green"|"amber"|"slate", "sub": str|None}
    """
    html = '<div class="hk-kpi-row">'
    for c in cards:
        accent = c.get("accent", "slate")
        sub_html = f'<div class="kpi-sub">{c["sub"]}</div>' if c.get("sub") else ""
        html += (
            f'<div class="hk-kpi-card t-{accent}">'
            f'<div class="kpi-value">{c["value"]}</div>'
            f'<div class="kpi-label">{c["label"]}</div>'
            f'{sub_html}'
            f'</div>'
        )
    html += "</div>"
    st.html(html)


def render_metric_grid(metrics: list[dict], columns: int = 4) -> None:
    """st.metric grid — used for detail sections."""
    cols = st.columns(min(columns, len(metrics)))
    for i, m in enumerate(metrics):
        with cols[i]:
            delta = m.get("delta")
            color = m.get("delta_color", "normal")
            if delta:
                st.metric(label=m["label"], value=m["value"], delta=delta, delta_color=color)
            else:
                st.metric(label=m["label"], value=m["value"])
