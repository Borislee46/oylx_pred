"""Difficulty bar with clickable school circles. All UI switching client-side JS."""

from __future__ import annotations

import streamlit as st

from src.pages.prediction.result_display._comp_core import (
    escape_html,
    logo_b64,
    safe_float,
    tier_color_for,
)
from src.pages.prediction.result_display._comp_sections import (
    gap_sections,
    narrative_sections,
)


def render_card(schools, profiles, student, selected, tier_map, threshold_positions=None):
    swp = [s for s in schools if s["university"] in profiles]
    if not swp:
        return
    N = len(swp)
    mx = max(safe_float(s.get("probability")) for s in swp)
    mn = min(safe_float(s.get("probability")) for s in swp)
    prange = mx - mn if mx > mn else 0.01

    st.html(
        _hover_css()
        + '<div style="border:1px solid rgba(148,163,184,0.18);border-radius:18px;'
        'background:rgba(255,255,255,0.85);padding:18px 20px 14px;margin-bottom:16px">'
        + _header(N)
        + _bar(swp, profiles, mn, prange, selected, tier_map, threshold_positions)
        + gap_sections(swp, profiles, student, selected, tier_map)
        + narrative_sections(swp, profiles, student, selected)
        + "</div>"
    )
    st.components.v1.html(_click_js(), height=0)


def _header(n):
    return (
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'margin-bottom:14px"><div style="display:flex;align-items:center;gap:10px">'
        '<span style="font-weight:700;font-size:15px;color:#1e293b">录取竞争力定位</span>'
        f'<span style="font-size:11px;color:#94a3b8">{n} 所院校</span></div></div>'
    )


def _bar(schools, profiles, min_p, p_range, selected, tier_map, threshold_positions=None):
    # ── Compute dynamic zone boundaries from difficulty-weighted thresholds ──
    if threshold_positions:
        target_vals = []
        safety_vals = []
        for s in schools:
            u = s["university"]
            safety_t, target_t = threshold_positions.get(u, (0.55, 0.30))
            safety_vals.append(safety_t)
            target_vals.append(target_t)
        median_target = sorted(target_vals)[len(target_vals) // 2] if target_vals else 0.30
        median_safety = sorted(safety_vals)[len(safety_vals) // 2] if safety_vals else 0.55
        t_pos = max(5, min(95, (median_target - min_p) / p_range * 100)) if p_range > 0 else 35
        s_pos = max(5, min(95, (median_safety - min_p) / p_range * 100)) if p_range > 0 else 60
    else:
        t_pos, s_pos = 35, 60

    # ── Pre-compute positions and group into clusters for even vertical distribution ──
    CLUSTER_THRESHOLD = 5
    DEFAULT_SPREAD = 22  # px between circle centers in default state
    positions = []
    for s in schools:
        prob = safe_float(s.get("probability"))
        pos = (prob - min_p) / p_range * 100 if p_range > 0 else 50
        pos = max(5, min(95, 100 - pos))
        positions.append(pos)

    # Build groups: consecutive circles within CLUSTER_THRESHOLD %
    groups = []
    cur_group = []
    for i, pos in enumerate(positions):
        if not cur_group:
            cur_group.append(i)
        elif abs(pos - positions[i - 1]) < CLUSTER_THRESHOLD:
            cur_group.append(i)
        else:
            groups.append(cur_group)
            cur_group = [i]
    if cur_group:
        groups.append(cur_group)

    # Evenly distribute vertical offsets within each group
    offsets = {}
    for group in groups:
        n = len(group)
        if n >= 2:
            mid = (n - 1) / 2.0
            for j, idx in enumerate(group):
                offsets[idx] = int((j - mid) * DEFAULT_SPREAD)
        else:
            offsets[group[0]] = 0

    parts = []
    for i, s in enumerate(schools):
        u = s["university"]
        pos = positions[i]
        v_offset = offsets.get(i, 0)

        c = tier_color_for(tier_map.get(u, "冲刺"))
        n = profiles.get(u, {}).get("n_admitted", 0)
        sz = max(26, min(36, 24 + int(n / 50)))
        hardest = i == len(schools) - 1
        sel = u == selected
        ring = "3px" if hardest else "2px"
        sel_class = " hk-comp-sel" if sel else ""
        glow = f"box-shadow:0 0 0 4px {c}1a;" if sel else "box-shadow:0 1px 3px rgba(0,0,0,0.06);"
        logo = logo_b64(u)
        inner = (
            f'<img src="data:image/png;base64,{logo}" style="width:{sz-6}px;height:{sz-6}px;'
            f'border-radius:50%;object-fit:contain" alt="{escape_html(u)}">'
        ) if logo else (
            f'<span style="font-size:{max(9,sz//3)}px;font-weight:700;color:{c}">{u[:1]}</span>'
        )
        parts.append(
            f'<div class="hk-comp-dot{sel_class}" data-school="{escape_html(u)}" '
            f'data-x="{pos:.1f}" '
            f'title="{escape_html(u)}" '
            f'style="position:absolute;left:{pos}%;top:{50+v_offset}%;'
            f'transform:translate(-50%,-50%);text-align:center;cursor:pointer">'
            f'<div class="hk-comp-circle" style="width:{sz}px;height:{sz}px;background:white;'
            f'border:{ring} solid {c};border-radius:50%;margin:0 auto;display:flex;'
            f'align-items:center;justify-content:center;{glow}">{inner}</div></div>'
        )

    hardest = schools[-1]
    hp = 100 - (safe_float(hardest.get("probability")) - min_p) / p_range * 100 if p_range > 0 else 50
    hp = max(5, min(95, hp))
    pointer = ""
    if hp > s_pos + 5:
        pointer = (
            f'<div style="position:absolute;left:{hp}%;top:-34px">'
            '<div style="background:#ef4444;color:white;font-size:11px;font-weight:600;'
            'padding:3px 10px;border-radius:8px;white-space:nowrap;'
            'box-shadow:0 2px 8px rgba(239,68,68,0.35)">最大挑战</div>'
            '<div style="width:0;height:0;border-left:7px solid transparent;'
            'border-right:7px solid transparent;border-top:7px solid #ef4444;margin:0 auto"></div></div>'
        )

    # Dynamic gradient: zone boundaries float with difficulty-weighted thresholds
    r_pos = max(t_pos - 8, 2)
    bar_style = (
        "position:relative;height:68px;background:linear-gradient(to right,"
        f"rgba(34,197,94,0.12) 0%,rgba(34,197,94,0.12) {t_pos:.0f}%,"
        f"rgba(59,130,246,0.10) {t_pos:.0f}%,rgba(59,130,246,0.10) {s_pos:.0f}%,"
        f"rgba(249,115,22,0.10) {s_pos:.0f}%,rgba(249,115,22,0.10) {r_pos:.0f}%,"
        f"rgba(239,68,68,0.10) {r_pos:.0f}%,rgba(239,68,68,0.10) 100%);"
        "border-radius:12px;margin-bottom:4px"
    )
    zones = (
        '<div style="display:flex;justify-content:space-between;font-size:10px;'
        'color:#94a3b8;padding:0 4px;margin-bottom:14px">'
        '<span style="color:#22c55e">← 保底</span><span style="color:#3b82f6">目标</span>'
        '<span style="color:#f97316">冲刺</span><span style="color:#ef4444">高难 →</span></div>'
    )
    return f'<div data-comp-bar style="{bar_style}">{"".join(parts)}{pointer}</div>{zones}'


def _hover_css():
    return (
        "<style>"
        ".hk-comp-circle{transition:transform .22s ease,box-shadow .22s ease,margin-top .35s ease;}"
        ".hk-comp-circle:hover{transform:scale(1.22)!important;"
        "box-shadow:0 6px 16px rgba(0,0,0,0.18)!important;}"
        ".hk-comp-circle:active{transform:scale(.92)!important;"
        "box-shadow:0 1px 4px rgba(0,0,0,0.1)!important;transition:transform .06s ease;}"
        ".hk-comp-dot{cursor:pointer;}"
        "</style>"
    )


def _click_js():
    return (
        "<script>setTimeout(function(){"
        "var D=window.parent.document;"
        "function S(n){"
        "D.querySelectorAll('.hk-comp-gap').forEach(function(g){"
        "g.style.display=g.getAttribute('data-school')===n?'block':'none';});"
        "D.querySelectorAll('.hk-comp-narrative').forEach(function(t){"
        "t.style.display=t.getAttribute('data-school')===n?'block':'none';});"
        "D.querySelectorAll('.hk-comp-dot').forEach(function(d){"
        "var m=d.getAttribute('data-school')===n;d.classList.toggle('hk-comp-sel',m);"
        "var c=d.querySelector('.hk-comp-circle');if(c){var bc=getComputedStyle(c).borderColor;"
        "c.style.boxShadow=m?'0 0 0 4px '+bc+'1a':'0 1px 3px rgba(0,0,0,0.06)';}});}"
        "D.querySelectorAll('.hk-comp-dot').forEach(function(d){"
        "d.addEventListener('click',function(e){e.stopPropagation();"
        "var s=this.getAttribute('data-school');if(s)S(s);});});"
        "},100);</script>"
    )
