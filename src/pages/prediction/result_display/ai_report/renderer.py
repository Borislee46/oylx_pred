from __future__ import annotations

import base64
import html
import math
from typing import Any

import streamlit as st

from src.pages.prediction.result_display.ai_report.sections import (
    _highlight_bold,
    render_product_reasons,
)
from src.pages.prediction.result_display.ai_report.styles import REPORT_STYLE
from src.pages.prediction.result_display.radar_scoring import (
    compute_radar_values,
)
from src.pages.prediction.result_display.sales_recommendation import (
    blocks_selection_to_product_dicts,
    build_default_blocks_selection,
)
from src.utils.numeric import clip_probability_coerce

_RADAR_ANGLES = [
    (-90, "学术绩点"),
    (-18, "语言"),
    (54, "科研论文"),
    (126, "实习获奖"),
    (198, "学校"),
]


def _build_radar_pentagon(values: list[float], labels: list[str]) -> str:
    cx, cy, r = 100, 100, 72
    n = len(values)
    angles = [(math.sin(math.radians(a)), math.cos(math.radians(a))) for a, _ in _RADAR_ANGLES]

    grids = ""
    for level in (1, 2, 3, 4):
        lr = r * level / 4
        pts = " ".join(f"{cx + sc[1] * lr:.1f},{cy + sc[0] * lr:.1f}" for sc in angles)
        sw = "1.2" if level == 2 else "0.6"
        sc_color = "rgba(148,163,184,0.45)" if level == 2 else "rgba(148,163,184,0.2)"
        grids += f'<polygon points="{pts}" fill="none" stroke="{sc_color}" stroke-width="{sw}"/>'

    axes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{cx + sc[1] * r:.1f}" y2="{cy + sc[0] * r:.1f}"'
        f' stroke="rgba(148,163,184,0.3)" stroke-width="0.6"/>'
        for sc in angles
    )

    pts = ""
    dots = ""
    for i in range(n):
        v = values[i]
        sc = angles[i]
        dr = r * v / 100
        dx = cx + sc[1] * dr
        dy = cy + sc[0] * dr
        pts += f"{dx:.1f},{dy:.1f} "
        label = labels[i]
        dots += (
            f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="3.5" fill="#06b6d4" stroke="#fff" stroke-width="1.2">'
            f"<title>{label}: {int(v)}%</title></circle>"
        )

    label_html = ""
    for _i, (sc, (_, name)) in enumerate(zip(angles, _RADAR_ANGLES, strict=True)):
        lx = cx + sc[1] * (r + 13)
        ly = cy + sc[0] * (r + 13)
        anchor = "start" if sc[1] > 0.1 else ("end" if sc[1] < -0.1 else "middle")
        dy = ' dy="-2"' if sc[0] < -0.3 else (' dy="4"' if sc[0] > 0.3 else "")
        label_html += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="11" fill="#a5b5c9"'
            f' font-family="system-ui,sans-serif" text-anchor="{anchor}"{dy}>{name}</text>'
        )

    return (
        '<svg width="175" height="175" viewBox="-15 -15 230 230" xmlns="http://www.w3.org/2000/svg">'
        + grids
        + axes
        + f'<polygon points="{pts.strip()}" fill="rgba(6,182,212,0.12)" stroke="#06b6d4" stroke-width="1.4"/>'
        + dots
        + label_html
        + "</svg>"
    )


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


def _compute_match(input_data: dict, sim: list, cross: list, user: list) -> float:
    all_items = (sim or []) + (cross or []) + (user or [])
    probs = sorted(
        (clip_probability_coerce(r.get("probability")) for r in all_items if isinstance(r, dict)),
        reverse=True,
    )
    n = len(probs)
    if n == 0:
        return 50.0
    top3 = sum(probs[:3]) / min(3, n)
    p90, p10 = (
        probs[min(int(n * 0.9), n - 1)],
        probs[min(int(n * 0.1), n - 1)],
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


def _product_meta_html(p: dict) -> str:
    segments: list[str] = []
    if p.get("variant"):
        segments.append(html.escape(p["variant"]))
    if p.get("scale"):
        segments.append(html.escape(p["scale"]))
    if p.get("price"):
        segments.append(f'<span class="ar-product-price">{html.escape(p["price"])}</span>')
    return " · ".join(segments)


def build_matched_products(input_data: dict, has_cross: bool = False) -> list[dict]:
    from src.pages.prediction.result_display.sales_recommendation import (
        _build_products,
    )

    products = _build_products(input_data, has_cross)
    if products:
        return products
    names = build_default_blocks_selection(input_data)
    return blocks_selection_to_product_dicts(names, input_data)


from src.agent.schemas import compute_tiers


def render_static_frame(
    input_data: dict[str, Any],
    sim_results: list[dict],
    cross_results: list[dict],
    user_results: list[dict],
    *,
    products: list[dict] | None = None,
    include_product_grid: bool = True,
) -> list[dict]:
    match_score = _compute_match(input_data, sim_results, cross_results, user_results)
    if products is None:
        products = build_matched_products(input_data, bool(cross_results))

    try:
        gpa = float(input_data.get("gpa", 0) or 0)
        gpa_str = f"{gpa:.1f}" if gpa > 0 else "—"
    except (TypeError, ValueError):
        gpa_str = "—"

    # Language pill: prefer raw score; fall back to normalised score (0-1).
    # Mirrors compute_radar_values() fallback in radar_scoring.py:164-169.
    raw_lang = input_data.get("language_score_raw")
    if raw_lang is None or (isinstance(raw_lang, (int, float)) and float(raw_lang) <= 0):
        lang_score = float(input_data.get("language_score", 0) or 0)
        lang_type = str(input_data.get("language_type", ""))
        if lang_score > 0:
            lang_max = 120.0 if lang_type in ("托福", "TOEFL") else 9.0
            raw_lang = lang_score * lang_max
    lang_str = _fmt_lang(
        input_data.get("language_type", ""),
        raw_lang,
    )

    radar_vals, radar_labels = compute_radar_values(input_data)

    svg_str = _build_radar_pentagon(radar_vals, radar_labels)
    svg_b64 = base64.b64encode(svg_str.encode()).decode()
    radar_html = (
        '<div class="ar-radar-wrap ar-reveal" style="animation-delay:0s">'
        f'<img src="data:image/svg+xml;base64,{svg_b64}" width="175" height="175"'
        ' style="display:block;margin:0 auto" alt="申请者画像五维图">'
        "</div>"
    )

    all_items = (sim_results or []) + (cross_results or []) + (user_results or [])
    probs = sorted(
        clip_probability_coerce(r.get("probability")) for r in all_items if isinstance(r, dict)
    )
    n = len(probs)
    bars_html = ""
    if n > 0:
        tier_labels = compute_tiers(probs)
        for i, (label, bcolor) in enumerate(
            [
                ("保底", "#10b981"),
                ("适中", "#f59e0b"),
                ("冲刺", "#ef4444"),
            ]
        ):
            cnt = tier_labels.count(label)
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
            f'<span class="ar-product-meta">{_product_meta_html(p)}</span></div></div>'
        )

    full_html = (
        REPORT_STYLE
        + '<div class="ar-card"><div class="ar-grid">'
        + '<div class="ar-score-panel">'
        + '<div class="ar-section-label ar-reveal" style="animation-delay:0s">AI 选校报告</div>'
        + radar_html
        + '<div class="ar-profile-line ar-reveal" style="animation-delay:0.08s">'
        + f'<span class="ar-profile-pill">GPA {gpa_str}</span>'
        + f'<span class="ar-profile-pill">{lang_str}</span>'
        + "</div></div>"
        + '<div class="ar-main-panel">'
        + '<div class="ar-section-label ar-reveal" style="animation-delay:0.06s">申请策略</div>'
        + bars_html
    )
    if products and include_product_grid:
        full_html += (
            '<div class="ar-section-label ar-reveal" style="animation-delay:0.4s;margin-top:0.35rem">'
            "产品匹配</div>"
            '<div class="ar-product-grid">' + prod_html + "</div>"
        )
    full_html += "</div></div></div>"

    st.html(full_html)
    st.session_state["_ar_match"] = match_score
    return products


def render_ai_section(
    explanation: dict[str, Any],
    streaming: bool = False,
    pinned: bool = False,
    unified_results: list[dict[str, Any]] | None = None,
    percentile_data: dict[str, dict[str, Any]] | None = None,
) -> None:
    card_cls = "ar-card ar-ai-card hk-sales-ai-card"
    if streaming:
        card_cls += " is-streaming"
    if pinned:
        card_cls += " is-pinned"
    parts = []
    stream_cls = " ar-streaming" if streaming else ""

    if overview := explanation.get("overview"):
        parts.append(
            '<div class="ar-section-label">你的申请画像</div>'
            f'<p class="ar-overview ar-section-enter{stream_cls}">{_highlight_bold(overview)}</p>'
        )

    has_s = bool(explanation.get("strengths"))
    has_c = bool(explanation.get("concerns"))
    if has_s or has_c:
        parts.append('<div class="ar-insight-grid">')
        if has_s:
            items = "".join(f"<li>{_highlight_bold(s)}</li>" for s in explanation["strengths"])
            parts.append(
                '<div class="ar-insight-card is-strength ar-section-enter">'
                '<div class="ar-section-label">优势</div>'
                f'<ul class="ar-list">{items}</ul></div>'
            )
        if has_c:
            items = "".join(f"<li>{_highlight_bold(c)}</li>" for c in explanation["concerns"])
            concern_label = "可提升方向"
            parts.append(
                '<div class="ar-insight-card is-concern ar-section-enter">'
                f'<div class="ar-section-label">{concern_label}</div>'
                f'<ul class="ar-list">{items}</ul></div>'
            )
        parts.append("</div>")

    if summary := explanation.get("summary"):
        parts.append(
            f'<p class="ar-overview ar-section-enter" style="font-weight:600;margin-top:0.6rem">'
            f"{_highlight_bold(summary)}</p>"
        )

    school_notes = explanation.get("school_notes")
    if school_notes and not streaming:
        from src.pages.prediction.result_display.ai_report.sections import (
            render_school_cards as _render_cards,
        )

        cards_html = _render_cards(
            school_notes,
            unified_results=unified_results,
            percentile_data=percentile_data,
            animate=True,
        )
        if cards_html:
            parts.append(cards_html)

    product_html = render_product_reasons(explanation.get("products"))
    if product_html:
        parts.append(product_html)

    if streaming:
        parts.append(
            '<span class="ar-wait">AI解读中'
            '<span class="hk-thought-wait-d1">.</span>'
            '<span class="hk-thought-wait-d2">.</span>'
            '<span class="hk-thought-wait-d3">.</span>'
            "</span>"
        )

    if not parts:
        return

    st.html(f'<div class="{card_cls}">' + "".join(parts) + "</div>")
