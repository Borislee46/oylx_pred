"""AI 选校报告 — newspaper 2-col layout, streaming, product-aware."""

from __future__ import annotations

import base64
import html
import math
from typing import Any

import streamlit as st

from src.pages.prediction.ai_report_catalog import PRODUCTS
from src.pages.prediction.ai_report_sections import (
    _highlight_bold,
    render_product_reasons,
)
from src.pages.prediction.ai_report_styles import REPORT_STYLE
from src.utils.school_constants import SCHOOL_LEVEL_SCORES
from src.utils.school_level_service import SchoolLevelService

# ── Radar chart (5-axis pentagon) ──────────────────────────────────────
_RADAR_ANGLES = [
    (-90, "学术绩点"),
    (-18, "语言"),
    (54, "科研论文"),
    (126, "实习获奖"),
    (198, "学校"),
]


def _build_radar_pentagon(values: list[float], labels: list[str]) -> str:
    """5-axis SVG radar chart (200x200), values 0-100. All styles inline."""
    cx, cy, r = 100, 100, 72
    n = len(values)
    angles = [(math.sin(math.radians(a)), math.cos(math.radians(a))) for a, _ in _RADAR_ANGLES]

    # Grid: 25%, 50%, 75%, 100%
    grids = ""
    for level in (1, 2, 3, 4):
        lr = r * level / 4
        pts = " ".join(f"{cx + sc[1] * lr:.1f},{cy + sc[0] * lr:.1f}" for sc in angles)
        sw = "1.2" if level == 2 else "0.6"  # 50% line bolder
        sc_color = "#cbd5e1" if level == 2 else "#e2e8f0"
        grids += f'<polygon points="{pts}" fill="none" stroke="{sc_color}" stroke-width="{sw}"/>'

    # Axis lines
    axes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{cx + sc[1] * r:.1f}" y2="{cy + sc[0] * r:.1f}"'
        f' stroke="#cbd5e1" stroke-width="0.6"/>'
        for sc in angles
    )

    # Data polygon + dots
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

    # Axis labels
    label_html = ""
    for _i, (sc, (_, name)) in enumerate(zip(angles, _RADAR_ANGLES, strict=True)):
        lx = cx + sc[1] * (r + 13)
        ly = cy + sc[0] * (r + 13)
        anchor = "start" if sc[1] > 0.1 else ("end" if sc[1] < -0.1 else "middle")
        dy = ' dy="-2"' if sc[0] < -0.3 else (' dy="4"' if sc[0] > 0.3 else "")
        label_html += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="11" fill="#64748b"'
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


from src.agent.schemas import compute_tiers


def render_static_frame(
    input_data: dict[str, Any],
    sim_results: list[dict],
    cross_results: list[dict],
    user_results: list[dict],
) -> list[dict]:
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

    # ── 五边形雷达图: 学术绩点 / 语言能力 / 科研论文 / 实习获奖 / 学校水平 ──
    try:
        _gpa_v = float(input_data.get("gpa", 0) or 0)
    except (TypeError, ValueError):
        _gpa_v = 0
    try:
        _lang_raw = float(input_data.get("language_score_raw", 0) or 0)
    except (TypeError, ValueError):
        _lang_raw = 0
    _lang_type = str(input_data.get("language_type", ""))
    _lang_max = 120.0 if _lang_type in ("托福", "TOEFL") else 9.0

    # 1. 学术绩点: GPA/4.0*100, GMAT≥700 or GRE≥320 bonus +10%
    _gpa_score = min(_gpa_v / 4.0 * 100, 100) if _gpa_v > 0 else 0
    _exam_type = str(input_data.get("exam_type", "")).upper()
    _exam_score = float(input_data.get("exam_score", 0) or 0)
    if _gpa_v > 0:
        if _exam_type == "GMAT" and _exam_score >= 700:
            _gpa_score = min(_gpa_score + 10, 100)
        elif _exam_type == "GRE" and _exam_score >= 320:
            _gpa_score = min(_gpa_score + 10, 100)

    # 2. 语言能力: raw / max * 100
    _lang_score = min(_lang_raw / _lang_max * 100, 100) if _lang_raw > 0 else 0

    # 3. 科研论文: (research*0.6 + paper*0.4) / 3 * 100
    _research_n = int(input_data.get("research_count", 0) or 0)
    _paper_n = int(input_data.get("paper_count", 0) or 0)
    _research_score = min((_research_n * 0.6 + _paper_n * 0.4) / 3 * 100, 100)

    # 4. 实习获奖: (internship*0.5 + award*0.5) / 3 * 100
    _intern_n = int(input_data.get("internship_count", 0) or 0)
    _award_n = int(input_data.get("award_count", 0) or 0)
    _practice_score = min((_intern_n * 0.5 + _award_n * 0.5) / 3 * 100, 100)

    # 5. 学校水平: from SCHOOL_LEVEL_SCORES
    _bg_uni = str(input_data.get("background_university", ""))
    _school_score = 50.0  # default unknown
    if _bg_uni:
        try:
            info = SchoolLevelService().get_school_info(_bg_uni)
            level = info.get("school_level", "未知")
            _school_score = SCHOOL_LEVEL_SCORES.get(level, 0.50) * 100
        except Exception:
            pass

    radar_vals = [_gpa_score, _lang_score, _research_score, _practice_score, _school_score]
    radar_labels = ["学术绩点", "语言能力", "科研论文", "实习获奖", "学校水平"]

    svg_str = _build_radar_pentagon(radar_vals, radar_labels)
    svg_b64 = base64.b64encode(svg_str.encode()).decode()
    radar_html = (
        '<div class="ar-radar-wrap ar-reveal" style="animation-delay:0s">'
        f'<img src="data:image/svg+xml;base64,{svg_b64}" width="175" height="175"'
        ' style="display:block;margin:0 auto" alt="申请者画像五维图">'
        "</div>"
    )

    all_items = (sim_results or []) + (cross_results or []) + (user_results or [])
    probs = sorted(r.get("probability", 0) or 0 for r in all_items if isinstance(r, dict))
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
        + radar_html
        + '<div class="ar-profile-line ar-reveal" style="animation-delay:0.08s">'
        + f'<span class="ar-profile-pill">GPA {gpa_str}</span>'
        + f'<span class="ar-profile-pill">{lang_str}</span>'
        + "</div></div>"
        + '<div class="ar-main-panel">'
        + '<div class="ar-section-label ar-reveal" style="animation-delay:0.06s">专业梯度</div>'
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
    return products


def render_ai_section(
    explanation: dict[str, Any],
    streaming: bool = False,
    pinned: bool = False,
) -> None:
    """Render only the AI text portion (used for both streaming and final)."""
    card_cls = "ar-card ar-ai-card"
    if streaming:
        card_cls += " is-streaming"
    if pinned:
        card_cls += " is-pinned"
    parts = []
    stream_cls = " ar-streaming" if streaming else ""

    if overview := explanation.get("overview"):
        parts.append(
            '<div class="ar-section-label">顾问解读</div>'
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
            parts.append(
                '<div class="ar-insight-card is-concern ar-section-enter">'
                '<div class="ar-section-label">需关注</div>'
                f'<ul class="ar-list">{items}</ul></div>'
            )
        parts.append("</div>")

    if summary := explanation.get("summary"):
        parts.append(
            f'<p class="ar-overview ar-section-enter" style="font-weight:600;margin-top:0.6rem">'
            f"{_highlight_bold(summary)}</p>"
        )

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

    if parts:
        st.html(f'<div class="{card_cls}"><hr class="ar-divider">' + "".join(parts) + "</div>")
