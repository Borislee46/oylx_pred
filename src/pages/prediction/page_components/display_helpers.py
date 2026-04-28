"""Shared rendering helpers for AI explanation."""
from __future__ import annotations

import html

import streamlit as st


def show_explanation(explanation: dict) -> None:
    parts = ['<div class="hk-section-label">AI 选校解读</div>']

    if overview := explanation.get("overview"):
        parts.append(
            '<div class="hk-assessment">'
            f'<p>{html.escape(overview)}</p>'
            '</div>'
        )

    if strengths := explanation.get("strengths"):
        items = "".join(f"<li>{html.escape(s)}</li>" for s in strengths)
        parts.append(
            '<div class="hk-insight-card hk-insight-positive">'
            '<div class="hk-leadin-label">优势</div>'
            f'<ul>{items}</ul>'
            '</div>'
        )

    if concerns := explanation.get("concerns"):
        items = "".join(f"<li>{html.escape(c)}</li>" for c in concerns)
        parts.append(
            '<div class="hk-insight-card hk-insight-caution">'
            '<div class="hk-leadin-label">需注意</div>'
            f'<ul>{items}</ul>'
            '</div>'
        )

    if summary := explanation.get("summary"):
        parts.append(
            '<div class="hk-insight-card hk-insight-accent">'
            f'<p>{html.escape(summary)}</p>'
            '</div>'
        )

    st.html("\n".join(parts))
