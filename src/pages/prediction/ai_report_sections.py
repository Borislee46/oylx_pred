from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st
from rapidfuzz import fuzz


def _highlight_bold(text: str) -> str:
    """Convert **text** markers to <strong>text</strong> after HTML escaping."""
    escaped = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def render_school_notes(
    school_notes: list[dict[str, Any]] | None,
    animate: bool = True,
) -> str:
    if not school_notes:
        return ""
    enter = " ar-section-enter" if animate else ""
    items_parts = []
    for sn in school_notes[:5]:
        uni = html.escape(str(sn.get("university", "")))
        major = html.escape(str(sn.get("major", "")))
        note = _highlight_bold(str(sn.get("note", "")))
        items_parts.append(
            '<div class="ar-school-note">'
            f'<span class="ar-school-note-uni">{uni} {major}</span>'
            f'<span class="ar-school-note-text">{note}</span>'
            "</div>"
        )
    return (
        f'<div class="ar-school-notes{enter}">'
        '<div class="ar-section-label">院校简析</div>' + "".join(items_parts) + "</div>"
    )


_PROB_COLORS = {
    "high": ("usc-prob-high", "#10b981"),
    "mid": ("usc-prob-mid", "#f59e0b"),
    "low": ("usc-prob-low", "#ef4444"),
}

_STAT_COLORS = [
    (75, "#10b981"),
    (50, "#06b6d4"),
    (25, "#f59e0b"),
    (0, "#ef4444"),
]


def _prob_tier(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return _PROB_COLORS["high"]
    elif probability >= 0.40:
        return _PROB_COLORS["mid"]
    return _PROB_COLORS["low"]


def _stat_fill_color(pct: float) -> str:
    for threshold, color in _STAT_COLORS:
        if pct >= threshold:
            return color
    return _STAT_COLORS[-1][1]


def _fuzzy_lookup_key(uni: str, major: str, keys: set[str]) -> str | None:
    uni_norm = " ".join(uni.split())
    major_norm = " ".join(major.split())
    norm = f"{uni_norm}|{major_norm}"
    if norm in keys:
        return norm
    candidates: list[tuple[float, str]] = []
    for key in keys:
        k_uni, _, k_major = key.partition("|")
        u_ratio = fuzz.ratio(uni_norm.lower(), k_uni.lower()) / 100.0
        m_ratio = fuzz.ratio(major_norm.lower(), k_major.lower()) / 100.0
        if u_ratio >= 0.80 and m_ratio >= 0.80:
            candidates.append(((u_ratio + m_ratio) / 2, key))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def render_school_cards(
    school_notes: list[dict[str, Any]] | None,
    unified_results: list[dict[str, Any]] | None = None,
    percentile_data: dict[str, dict[str, Any]] | None = None,
    animate: bool = True,
    label: str = "院校分析",
) -> str:
    """Build unified per-school cards: probability + AI note + percentile stats."""
    if not school_notes:
        return ""

    unified = unified_results or []
    prob_map: dict[str, float] = {}
    for r in unified:
        uni = str(r.get("university", ""))
        major = str(r.get("major", ""))
        prob = r.get("probability", 0) or 0
        prob_map[f"{uni}|{major}"] = float(prob)

    pdata = percentile_data or {}
    enter = " ar-section-enter" if animate else ""
    cards: list[str] = []

    for sn in school_notes[:5]:
        uni = html.escape(str(sn.get("university", "")))
        major = html.escape(str(sn.get("major", "")))
        note = _highlight_bold(str(sn.get("note", "")))
        raw_uni = str(sn.get("university", ""))
        raw_major = str(sn.get("major", ""))
        key = f"{raw_uni}|{raw_major}"
        prob = prob_map.get(key)
        pinfo = pdata.get(key, {})
        if prob is None:
            all_keys = set(prob_map.keys()) | set(pdata.keys())
            matched = _fuzzy_lookup_key(raw_uni, raw_major, all_keys)
            if matched:
                prob = prob_map.get(matched)
                pinfo = pdata.get(matched, {})

        # ── Header: university + probability badge ──
        header_parts = [
            '<div class="usc-header">',
            f'<span class="usc-uni">{uni} {major}</span>',
        ]
        if prob is not None:
            tier_cls, tier_color = _prob_tier(prob)
            header_parts.append(f'<span class="usc-prob-badge {tier_cls}">{prob:.0%}</span>')
        header_parts.append("</div>")

        # ── Probability bar ──
        bar_html = ""
        if prob is not None:
            _, bar_color = _prob_tier(prob)
            bar_html = (
                '<div class="usc-prob-track">'
                f'<div class="usc-prob-fill" style="width:{prob * 100:.0f}%;background:{bar_color};"></div>'
                "</div>"
            )

        # ── AI note ──
        note_html = f'<div class="usc-note">{note}</div>'

        # ── Stats grid ──
        stats_html = ""
        has_data = pinfo.get("has_data", False)
        if has_data:
            percentiles = pinfo.get("percentiles", {})
            values = pinfo.get("values", {})
            labels = pinfo.get("labels", {})
            sample_count = pinfo.get("university_count", 0)
            tiles: list[str] = []
            for feat, pct in percentiles.items():
                from src.pages.prediction.ai_school_stats import FEATURE_LABELS as _FL

                feat_label = _FL.get(feat, feat)
                val = values.get(feat, 0)
                lbl = labels.get(feat, "")
                fill_color = _stat_fill_color(pct)
                # format GPA as 2 decimals, language as int, counts as int
                if feat == "gpa":
                    val_str = f"{val:.2f}"
                elif feat == "language_score":
                    val_str = f"{val:.0f}"
                else:
                    val_str = f"{int(val)}"
                tiles.append(
                    '<div class="usc-stat">'
                    '<div class="usc-stat-header">'
                    f'<span class="usc-stat-name">{feat_label}</span>'
                    f'<span class="usc-stat-pct">{lbl} · {int(pct)}%</span>'
                    "</div>"
                    '<div class="usc-stat-bar">'
                    f'<div class="usc-stat-fill" style="width:{pct:.0f}%;background:{fill_color};"></div>'
                    "</div>"
                    f'<div class="usc-stat-value">{val_str}</div>'
                    "</div>"
                )
            stats_html = '<div class="usc-stats">' + "".join(tiles) + "</div>"
            if sample_count:
                stats_html += f'<div class="usc-samples">基于 {sample_count} 份历史数据</div>'
        else:
            stats_html = '<div class="usc-no-data">暂无足够历史数据对比</div>'

        cards.append(
            f'<div class="usc-card{enter}">'
            + "".join(header_parts)
            + bar_html
            + note_html
            + stats_html
            + "</div>"
        )

    return (
        f'<div class="usc-section">'
        f'<div class="ar-section-label">{label}</div>'
        f'<div class="usc-cards-grid">{"".join(cards)}</div>' + "</div>"
    )


def render_product_reasons(
    products: list[dict[str, Any]] | None,
    animate: bool = True,
) -> str:
    if not products:
        return ""
    items = [
        f"<li>{_highlight_bold(p.get('reason', ''))}</li>" for p in products if p.get("reason")
    ]
    if not items:
        return ""
    enter = " ar-section-enter" if animate else ""
    return (
        f'<div class="ar-product-reasons{enter}">'
        '<div class="ar-section-label">推荐说明</div>'
        f'<ul class="ar-list">{"".join(items)}</ul>'
        "</div>"
    )


def _render_overview_block(overview: str, is_streaming: bool, is_new: bool) -> str:
    cls = "ar-overview"
    if is_new:
        cls += " ar-section-enter"
    if is_streaming:
        cls += " ar-streaming"
    return (
        '<div class="ar-section-label">顾问解读</div>'
        f'<p class="{cls}">{_highlight_bold(overview)}</p>'
    )


def _render_insight_grid(
    strengths: list[str] | None,
    concerns: list[str] | None,
    seen_s: bool,
    seen_c: bool,
) -> str:
    has_s = bool(strengths)
    has_c = bool(concerns)
    if not has_s and not has_c:
        return ""

    parts = ['<div class="ar-insight-grid">']
    if has_s:
        enter = " ar-section-enter" if not seen_s else ""
        items = "".join(f"<li>{_highlight_bold(s)}</li>" for s in strengths)
        parts.append(
            f'<div class="ar-insight-card is-strength{enter}">'
            '<div class="ar-section-label">优势</div>'
            f'<ul class="ar-list">{items}</ul></div>'
        )
    if has_c:
        enter = " ar-section-enter" if not seen_c else ""
        items = "".join(f"<li>{_highlight_bold(c)}</li>" for c in concerns)
        parts.append(
            f'<div class="ar-insight-card is-concern{enter}">'
            '<div class="ar-section-label">需关注</div>'
            f'<ul class="ar-list">{items}</ul></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def render_ai_section_streaming(
    partial: dict[str, Any] | str,
    pinned: bool = True,
    seen_fields: frozenset | None = None,
    percentile_data: dict[str, dict[str, Any]] | None = None,
    unified_results: list[dict[str, Any]] | None = None,
) -> None:
    """Render streaming AI explanation with progressive section reveal.

    Args:
        partial: Either a plain string (initial loading) or a dict of
                 partially-parsed fields from the streaming JSON buffer.
        pinned: Whether the card should stick to top on scroll.
        seen_fields: Field names that have already appeared in a previous
                     frame — we skip enter animations for these.
        percentile_data: Optional per-school percentile stats for unified cards.
        unified_results: Optional prediction results for probability lookup.
    """
    seen = seen_fields or frozenset()
    card_cls = "ar-card ar-ai-card is-streaming"
    if pinned:
        card_cls += " is-pinned"

    if isinstance(partial, str):
        body = _highlight_bold(partial.strip()) or '<span class="ar-muted">正在生成解读...</span>'
        st.html(
            f'<div class="{card_cls}"><hr class="ar-divider">'
            '<div class="ar-section-label">顾问解读</div>'
            f'<p class="ar-overview ar-streaming">{body}</p>'
            "</div>"
        )
        return

    parts: list[str] = []

    # ── Overview ───────────────────────────────────────────────
    overview = partial.get("overview")
    if overview:
        parts.append(
            _render_overview_block(
                str(overview), is_streaming=True, is_new=("overview" not in seen)
            )
        )
    else:
        parts.append(
            '<div class="ar-section-label">顾问解读</div><p class="ar-overview ar-streaming">'
        )

    # ── Strengths / Concerns ───────────────────────────────────
    strengths = partial.get("strengths")
    concerns = partial.get("concerns")
    if strengths or concerns:
        parts.append(
            _render_insight_grid(
                strengths,
                concerns,
                seen_s=("strengths" in seen),
                seen_c=("concerns" in seen),
            )
        )

    # ── Summary ────────────────────────────────────────────────
    summary = partial.get("summary")
    if summary:
        enter = " ar-section-enter" if "summary" not in seen else ""
        parts.append(
            f'<p class="ar-overview{enter}" style="font-weight:600;margin-top:0.6rem">'
            f"{_highlight_bold(str(summary))}</p>"
        )

    # ── School notes (unified cards with percentile data) ──────
    school_notes = partial.get("school_notes")
    if school_notes:
        school_html = render_school_cards(
            school_notes,
            unified_results=unified_results,
            percentile_data=percentile_data,
            animate=("school_notes" not in seen),
        )
        if school_html:
            parts.append(school_html)

    # ── Product reasons ────────────────────────────────────────
    products = partial.get("products")
    if products:
        product_html = render_product_reasons(products, animate=("products" not in seen))
        if product_html:
            parts.append(product_html)

    if not parts:
        parts.append('<span class="ar-muted">正在生成解读...</span>')

    parts.append(
        '<span class="ar-wait">AI解读中'
        '<span class="hk-thought-wait-d1">.</span>'
        '<span class="hk-thought-wait-d2">.</span>'
        '<span class="hk-thought-wait-d3">.</span>'
        "</span>"
    )
    st.html(f'<div class="{card_cls}"><hr class="ar-divider">' + "".join(parts) + "</div>")
