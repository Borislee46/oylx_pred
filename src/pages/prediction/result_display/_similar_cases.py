from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.adjustment.knn_retrieval import retrieve_similar_cases
from src.utils.logger import setup_logger
from src.utils.numeric import prob_to_pct

_logger = setup_logger("page3", "prediction")


def fmt_gpa(gpa: float | None) -> str:
    if gpa is None or pd.isna(gpa):
        return "N/A"
    return f"{float(gpa):.2f}"


def fmt_lang(case: dict) -> str:
    ielts = case.get("ielts")
    toefl = case.get("toefl")
    if pd.notna(ielts) and ielts and float(ielts) > 0:
        return f"IELTS {float(ielts):.1f}"
    if pd.notna(toefl) and toefl and float(toefl) > 0:
        return f"TOEFL {float(toefl):.0f}"
    return "—"


def sim_tier(sim: float) -> str:
    if sim >= 0.70:
        return "high"
    if sim >= 0.40:
        return "mid"
    return "low"


def gpa_delta_html(case_gpa_raw, student_gpa_raw) -> str:
    try:
        c_gpa = float(case_gpa_raw)
        s_gpa = float(student_gpa_raw)
    except (ValueError, TypeError):
        return ""
    if pd.isna(c_gpa) or pd.isna(s_gpa) or s_gpa == 0:
        return ""
    delta = c_gpa - s_gpa
    if abs(delta) < 0.03:
        return '<span class="hk-sim-delta delta-flat">≈</span>'
    if delta > 0:
        return f'<span class="hk-sim-delta delta-up">↑{delta:+.2f}</span>'
    return f'<span class="hk-sim-delta delta-down">↓{delta:+.2f}</span>'


def render_similar_cases(
    student: dict,
    slot: str,
    *,
    cached_result: tuple[list[dict], int, str] | None = None,
) -> None:
    if cached_result is None:
        _logger.info("render_similar_cases: KNN cache miss for slot=%s, running retrieval", slot)
        cached_result = retrieve_similar_cases(student, k=3)
    else:
        _logger.info("render_similar_cases: KNN cache hit for slot=%s", slot)
    cases, level, note = cached_result
    _logger.info("render_similar_cases: slot=%s level=L%d n_cases=%d", slot, level, len(cases))

    level_palette = {1: "#34d399", 2: "#22d3ee", 3: "#fbbf24", 4: "#f87171"}
    lc = level_palette.get(level, "#64748b")
    st.html(
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.55rem;">'
        f'<span class="hk-sim-level-tag" style="background:{lc};">L{level}</span>'
        f'<span style="font-size:0.78rem;color:var(--hk-slate-500);">{note}</span>'
        f"</div>"
    )
    if not cases:
        _logger.warning("render_similar_cases: no cases found for slot=%s", slot)
        st.info("暂无足够相似的历史案例。")
        return

    student_gpa = student.get("gpa")
    parts: list[str] = []
    for i, c in enumerate(cases):
        admitted = c.get("admitted") == 1
        accent = "#34d399" if admitted else "#fbbf24"
        badge_bg = "rgba(5,150,105,0.12)" if admitted else "rgba(217,119,6,0.12)"
        badge_fg = "#34d399" if admitted else "#fbbf24"
        badge_text = "已录取" if admitted else "未录取"

        bg_uni = html.escape(str(c.get("background_university", "") or "未知院校"))
        bg_major = html.escape(str(c.get("background_major", "") or "未知专业"))
        tg_uni = html.escape(str(c.get("target_university", "") or "未知"))
        tg_major = html.escape(str(c.get("target_major", "") or "未知"))
        gpa_str = fmt_gpa(c.get("gpa"))
        lang = fmt_lang(c)
        sim = float(c.get("similarity", 0) or 0)
        tier = sim_tier(sim)
        delta = gpa_delta_html(c.get("gpa"), student_gpa)

        is_upset = c.get("is_upset", False)
        if is_upset:
            accent = "#f59e0b"

        card_class = (
            "hk-sim-case-card sim-weak" if (tier == "low" and not is_upset) else "hk-sim-case-card"
        )
        if is_upset:
            card_class += " sim-upset"

        upset_badge = (
            f'<span class="hk-sim-upset-badge" title="案例来自弱于你的背景档位，录取归因可能包含其他因素">'
            f'{html.escape(str(c.get("underdog_kind", "逆袭")))}</span>'
            if is_upset
            else ""
        )

        base_rate_html = ""
        br = c.get("base_rate")
        if br is not None:
            try:
                br_num = float(br)
            except (TypeError, ValueError):
                br_num = None
            if br_num is not None:
                label = "该背景录取率" if is_upset else "同类录取率"
                base_rate_html = (
                    f'<div class="hk-sim-base-rate">'
                    f'<span class="hk-sim-base-rate-label">{label}</span>'
                    f'<span class="hk-sim-base-rate-value">{br_num:.0%}</span>'
                    f"</div>"
                )

        meta_text = f"{bg_uni} · {bg_major}"
        meta_title = f"目标: {tg_uni} · {tg_major} | 背景: {bg_uni} · {bg_major} | GPA: {gpa_str} | 语言: {lang}"

        parts.append(
            f'<div class="{card_class}">'
            f'<div class="hk-sim-accent" style="background:{accent};"></div>'
            f'<div class="hk-sim-body">'
            f'<div class="hk-sim-header">'
            f'<span class="hk-sim-num">案例 #{i + 1}</span>'
            f'<span class="hk-sim-status" style="background:{badge_bg};color:{badge_fg};">{badge_text}</span>'
            f"{upset_badge}"
            f"</div>"
            f'<div class="hk-sim-meta" title="{meta_title}">{meta_text}</div>'
            f'<div class="hk-sim-metrics">'
            f'<div class="hk-sim-metric"><span class="hk-sim-metric-label">GPA</span>'
            f'<span class="hk-sim-metric-value">{gpa_str} {delta}</span></div>'
            f'<div class="hk-sim-metric"><span class="hk-sim-metric-label">语言</span>'
            f'<span class="hk-sim-metric-value">{lang}</span></div>'
            f'<div class="hk-sim-metric"><span class="hk-sim-metric-label">背景相似度</span>'
            f'<span class="hk-sim-metric-value">{sim:.0%}</span></div>'
            f"</div>"
            f"{base_rate_html}"
            f'<div class="hk-sim-bar-wrap">'
            f'<div class="hk-sim-bar-fill sim-{tier}" style="width:{prob_to_pct(sim)}%;"></div>'
            f"</div>"
            f"</div>"
            f"</div>"
        )
    st.html(f'<div class="hk-sim-cards-grid">{"".join(parts)}</div>')
