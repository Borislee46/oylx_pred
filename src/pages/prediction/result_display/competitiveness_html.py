from __future__ import annotations

import html
import time

import streamlit as st

from src.pages.prediction.result_display._comp_assets import click_js, hover_css
from src.pages.prediction.result_display._comp_bar import render_bar
from src.pages.prediction.result_display._comp_core import (
    MIN_SAMPLES_FOR_PROFILE,
    tier_color_for,
)
from src.pages.prediction.result_display._comp_sections import (
    gap_sections,
    narrative_sections,
)
from src.utils.logger import setup_logger
from src.utils.numeric import safe_float

_logger = setup_logger("page3", "prediction")


def render_card(
    schools, profiles, student, selected, tier_map, threshold_positions=None, missing_profiles=0
):
    t0 = time.perf_counter()

    swp = [s for s in schools if s["university"] in profiles]
    if not swp:
        _logger.warning("render_card: no schools with profile data, skipping")
        return
    N = len(swp)
    _logger.info(
        "render_card: %d schools with profiles (selected=%s, missing=%d)",
        N,
        selected,
        missing_profiles,
    )

    mx = max(safe_float(s.get("probability")) for s in swp)
    mn = min(safe_float(s.get("probability")) for s in swp)
    prange = mx - mn if mx > mn else 0.01  # 防除零

    selected_tier = tier_map.get(selected, "")

    missing_html = ""
    if missing_profiles > 0:
        names = "、".join(
            html.escape(str(s["university"])) for s in schools if s["university"] not in profiles
        )
        missing_html = (
            '<div style="font-size:11px;color:#94a3b8;margin-bottom:12px;padding:8px 12px;'
            'background:#fefce8;border-radius:8px;border:1px solid #fde68a">'
            f"{names} 历史录取样本不足（&lt;{MIN_SAMPLES_FOR_PROFILE}条），暂无法对比中位数据"
            "</div>"
        )

    st.html(
        hover_css() + '<div style="border:1px solid rgba(148,163,184,0.18);border-radius:18px;'
        'background:rgba(255,255,255,0.05);padding:18px 20px 14px;margin-bottom:16px">'
        + _header(N, selected, selected_tier)
        + render_bar(swp, profiles, mn, prange, selected, tier_map, threshold_positions)
        + missing_html
        + narrative_sections(swp, profiles, student, selected)
        + gap_sections(swp, profiles, student, selected, tier_map)
        + "</div>"
    )

    st.iframe(click_js(), height=1)
    _logger.info("render_card: done in %.0fms", (time.perf_counter() - t0) * 1000)


def _header(n, selected_school="", selected_tier=""):
    tier_badge = (
        f'<span id="hk-comp-cur-tier" style="background:{tier_color_for(selected_tier)};color:white;'
        f'font-size:11px;font-weight:600;padding:2px 10px;border-radius:10px;margin-left:8px;'
        f'display:{"inline-block" if selected_tier else "none"}">{selected_tier}</span>'
    )
    return (
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'margin-bottom:14px"><div style="display:flex;align-items:center;gap:10px">'
        '<span style="font-weight:700;font-size:15px;color:var(--hk-slate-800)">录取竞争力定位</span>'
        f'<span style="font-size:11px;color:#94a3b8">{n} 所院校</span></div>'
        f'<div style="display:flex;align-items:center;font-size:11px;color:var(--hk-slate-500)">'
        f"<span>当前查看：</span>"
        f'<span id="hk-comp-cur" style="font-weight:600;color:var(--hk-slate-800);margin-left:4px">{selected_school}</span>'
        f"{tier_badge}</div></div>"
    )
