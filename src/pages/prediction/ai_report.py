"""AI 选校报告 — newspaper 2-col layout, streaming, product-aware."""

from __future__ import annotations

import html
import math
from typing import Any

import streamlit as st

from src.pages.prediction.ai_report_catalog import PRODUCTS
from src.pages.prediction.ai_report_styles import REPORT_STYLE


def _fmt_lang(lang_type: str, raw_score) -> str:
    try:
        s = float(raw_score or 0)
    except (TypeError, ValueError):
        return "—"
    if s <= 0:
        return "—"
    if lang_type in ("托福", "TOEFL"):
        return f"托福 {s:.0f}"
    return f"雅思 {s:.1f}"


def _ring_color(score: float) -> str:
    if score >= 65:
        return "#10b981"
    elif score >= 40:
        return "#f59e0b"
    return "#ef4444"


def _compute_match(input_data: dict, sim: list, cross: list, user: list) -> float:
    all_items = (sim or []) + (cross or []) + (user or [])
    probs = sorted(
        (r.get("probability", 0) or 0 for r in all_items if isinstance(r, dict)),
        reverse=True,
    )
    n = len(probs)
    if n == 0:
        return 50.0
    top3 = sum(probs[:3]) / min(3, n)
    p90, p10 = (
        probs[min(int(n * 0.9), n - 1)],
        probs[min(int(n * 0.1), n - 1)] if n > 1 else (probs[0], probs[0]),
    )
    return round(
        min(top3 / 0.55, 1.0) * 35
        + min((p90 - p10) / 0.30, 1.0) * 20
        + max(_bg_health(input_data), 0),
        1,
    )


def _bg_health(input_data: dict) -> float:
    score = 45.0
    try:
        gpa = float(input_data.get("gpa", 0) or 0)
    except (TypeError, ValueError):
        gpa = 0
    try:
        raw_lang = float(input_data.get("language_score_raw", 0) or 0)
    except (TypeError, ValueError):
        raw_lang = 0
    lang_type = str(input_data.get("language_type", ""))
    exp = input_data.get("experience_details") or {}
    if 0 < gpa < 2.8:
        score -= 15
    elif 0 < gpa < 3.2:
        score -= 8
    lang_ok = (lang_type in ("雅思", "IELTS") and raw_lang >= 6.5) or (
        lang_type in ("托福", "TOEFL") and raw_lang >= 90
    )
    if raw_lang > 0 and not lang_ok:
        score -= 12
    if not exp.get("research"):
        score -= 5
    if not exp.get("internship"):
        score -= 5
    return score


def _build_products(input_data: dict, has_cross: bool) -> list[dict]:
    exp = input_data.get("experience_details") or {}
    try:
        gpa = float(input_data.get("gpa", 0) or 0)
    except (TypeError, ValueError):
        gpa = 0
    try:
        raw_lang = float(input_data.get("language_score_raw", 0) or 0)
    except (TypeError, ValueError):
        raw_lang = 0
    lang_type = str(input_data.get("language_type", ""))
    target_n = len(input_data.get("target_universities", []) or [])
    lang_ok = (lang_type in ("雅思", "IELTS") and raw_lang >= 6.5) or (
        lang_type in ("托福", "TOEFL") and raw_lang >= 90
    )

    products: list[dict] = []
    products.append(PRODUCTS["high_end_app"] if target_n >= 8 else PRODUCTS["std_app"])
    if raw_lang > 0 and not lang_ok:
        products.append(PRODUCTS["english"])
    if not exp.get("research") and gpa < 3.5:
        products.append(PRODUCTS["bg_research"])
    if not exp.get("internship"):
        products.append(PRODUCTS["bg_intern"])
    if has_cross:
        products.append(PRODUCTS["tutoring"])
    return products


def render_static_frame(
    input_data: dict[str, Any],
    sim_results: list[dict],
    cross_results: list[dict],
    user_results: list[dict],
) -> str:
    """Build the static frame as a single HTML string for zero-gap layout."""
    match_score = _compute_match(input_data, sim_results, cross_results, user_results)
    products = _build_products(input_data, bool(cross_results))

    try:
        gpa = float(input_data.get("gpa", 0) or 0)
        gpa_str = f"{gpa:.1f}" if gpa > 0 else "—"
    except (TypeError, ValueError):
        gpa_str = "—"
    lang_str = _fmt_lang(
        input_data.get("language_type", ""),
        input_data.get("language_score_raw"),
    )

    color = _ring_color(match_score)
    r = 40
    circ = 2 * math.pi * r
    offset = circ * (1 - match_score / 100)
    ring_html = (
        '<div class="ar-ring ar-reveal" style="animation-delay:0s">'
        f'<svg width="104" height="104" viewBox="0 0 104 104">'
        f'<circle class="ar-ring-bg" cx="52" cy="52" r="{r}"/>'
        f'<circle class="ar-ring-fill" cx="52" cy="52" r="{r}"'
        f' style="stroke:{color};stroke-dasharray:{circ:.1f};'
        f'stroke-dashoffset:{offset:.1f};" /></svg>'
        f'<div class="ar-ring-center">'
        f'<span class="ar-ring-score">{match_score:.0f}</span>'
        f'<span class="ar-ring-label">匹配度</span></div></div>'
    )

    all_items = (sim_results or []) + (cross_results or []) + (user_results or [])
    probs = sorted(r.get("probability", 0) or 0 for r in all_items if isinstance(r, dict))
    n = len(probs)
    bars_html = ""
    if n > 0:

        def _p(vals, pct):
            k = (pct / 100) * (n - 1)
            f_i = int(k)
            c = k - f_i
            return (
                vals[f_i] + c * (vals[f_i + 1] - vals[f_i])
                if f_i + 1 < n
                else vals[min(f_i, n - 1)]
            )

        lo, hi = _p(probs, 33), _p(probs, 66)
        for i, (label, cond, bcolor) in enumerate(
            [
                ("较稳", lambda p: p >= hi, "#10b981"),
                ("适中", lambda p: lo <= p < hi, "#f59e0b"),
                ("冲刺", lambda p: p < lo, "#ef4444"),
            ]
        ):
            cnt = sum(1 for p in probs if cond(p))
            pct = (cnt / n * 100) if n else 0
            bars_html += (
                f'<div class="ar-bar-row ar-reveal" style="animation-delay:{0.12 + i * 0.1:.2f}s">'
                f'<span class="ar-bar-label">{label}</span>'
                '<div class="ar-bar-track">'
                f'<div class="ar-bar-fill" style="width:{pct:.0f}%;background:{bcolor};"></div>'
                "</div>"
                f'<span class="ar-bar-count">{cnt}</span></div>'
            )

    prod_html = ""
    for i, p in enumerate(products):
        prod_html += (
            f'<div class="ar-product ar-reveal" style="animation-delay:{0.45 + i * 0.1:.2f}s">'
            f'<span class="ar-product-dot" style="background:{p["dot"]}"></span>'
            "<div>"
            f'<span class="ar-product-name">{html.escape(p["name"])}</span>'
            f'<span class="ar-product-meta">'
            f'{html.escape(p["variant"])} · {html.escape(p["scale"])} · '
            f'<span class="ar-product-price">{html.escape(p["price"])}</span>'
            "</span></div></div>"
        )

    full_html = (
        REPORT_STYLE
        + '<div class="ar-card"><div class="ar-grid">'
        + '<div class="ar-score-panel">'
        + '<div class="ar-section-label ar-reveal" style="animation-delay:0s">AI 选校报告</div>'
        + ring_html
        + '<div class="ar-profile-line ar-reveal" style="animation-delay:0.08s">'
        + f'<span class="ar-profile-pill">GPA {gpa_str}</span>'
        + f'<span class="ar-profile-pill">{lang_str}</span>'
        + "</div></div>"
        + '<div class="ar-main-panel">'
        + '<div class="ar-section-label ar-reveal" style="animation-delay:0.06s">选校梯度</div>'
        + bars_html
    )
    if products:
        full_html += (
            '<div class="ar-section-label ar-reveal" style="animation-delay:0.4s;margin-top:0.35rem">'
            "产品匹配</div>"
            '<div class="ar-product-grid">' + prod_html + "</div>"
        )
    full_html += "</div></div></div>"

    st.html(full_html)
    st.session_state["_ar_match"] = match_score
    st.session_state["_ar_products"] = products
    return full_html


def render_ai_section(
    explanation: dict[str, Any],
    streaming: bool = False,
    pinned: bool = False,
) -> None:
    """Render only the AI text portion (used for both streaming and final)."""
    cls = " ar-streaming" if streaming else ""
    card_cls = "ar-card ar-ai-card"
    if streaming:
        card_cls += " is-streaming"
    if pinned:
        card_cls += " is-pinned"
    parts = []

    if overview := explanation.get("overview"):
        parts.append(
            '<div class="ar-section-label">顾问解读</div>'
            f'<p class="ar-overview{cls}">{html.escape(overview)}</p>'
        )

    has_s = bool(explanation.get("strengths"))
    has_c = bool(explanation.get("concerns"))
    if has_s or has_c:
        parts.append('<div class="ar-insight-grid">')
        if has_s:
            items = "".join(f"<li>{html.escape(s)}</li>" for s in explanation["strengths"])
            parts.append(
                '<div class="ar-insight-card is-strength">'
                '<div class="ar-section-label">优势</div>'
                f'<ul class="ar-list">{items}</ul></div>'
            )
        if has_c:
            items = "".join(f"<li>{html.escape(c)}</li>" for c in explanation["concerns"])
            parts.append(
                '<div class="ar-insight-card is-concern">'
                '<div class="ar-section-label">需关注</div>'
                f'<ul class="ar-list">{items}</ul></div>'
            )
        parts.append("</div>")

    if summary := explanation.get("summary"):
        parts.append(
            f'<p class="ar-overview" style="font-weight:600;margin-top:0.55rem">'
            f"{html.escape(summary)}</p>"
        )

    if parts:
        st.html(f'<div class="{card_cls}"><hr class="ar-divider">' + "".join(parts) + "</div>")


def render_ai_section_streaming(partial_text: str, pinned: bool = True) -> None:
    """Render partial AI text during streaming."""
    card_cls = "ar-card ar-ai-card is-streaming"
    if pinned:
        card_cls += " is-pinned"
    body = html.escape(partial_text.strip()) or '<span class="ar-muted">正在生成解读...</span>'
    st.html(
        f'<div class="{card_cls}"><hr class="ar-divider">'
        '<div class="ar-section-label">顾问解读</div>'
        f'<p class="ar-overview ar-streaming">{body}</p>'
        "</div>"
    )
