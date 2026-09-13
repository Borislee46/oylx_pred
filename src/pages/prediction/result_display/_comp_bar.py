from __future__ import annotations

from src.pages.prediction.result_display._comp_core import (
    escape_html,
    logo_b64,
    tier_color_for,
)
from src.utils.numeric import clip_probability_coerce

_CLUSTER_THRESHOLD = 2
_DEFAULT_SPREAD = 15
_EXPANDED_SPREAD = 32


def render_bar(schools, profiles, min_p, p_range, selected, tier_map, threshold_positions=None):
    if threshold_positions and p_range > 0:
        safety_vals = [threshold_positions.get(s["university"], (0.55, 0.30))[0] for s in schools]
        target_vals = [threshold_positions.get(s["university"], (0.55, 0.30))[1] for s in schools]
        median_safety = sorted(safety_vals)[len(safety_vals) // 2]
        median_target = sorted(target_vals)[len(target_vals) // 2]
        safety_x = max(5, min(95, 100 - (median_safety - min_p) / p_range * 100))
        target_x = max(5, min(95, 100 - (median_target - min_p) / p_range * 100))
    else:
        safety_x, target_x = 33, 60  # 回退默认值

    positions = []
    for s in schools:
        prob = clip_probability_coerce(s.get("probability"))
        pos = (prob - min_p) / p_range * 100 if p_range > 0 else 50
        pos = max(5, min(95, 100 - pos))
        positions.append(pos)

    groups = []
    cur_group = []
    for i, pos in enumerate(positions):
        if not cur_group:
            cur_group.append(i)
        elif abs(pos - positions[i - 1]) < _CLUSTER_THRESHOLD:
            cur_group.append(i)
        else:
            groups.append(cur_group)
            cur_group = [i]
    if cur_group:
        groups.append(cur_group)

    offsets = {}
    expanded_offsets = {}
    group_map = {}
    for g_idx, group in enumerate(groups):
        n = len(group)
        if n >= 2:
            mid = (n - 1) / 2.0
            for j, idx in enumerate(group):
                offsets[idx] = int((j - mid) * _DEFAULT_SPREAD)
                expanded_offsets[idx] = int((j - mid) * _EXPANDED_SPREAD)
                group_map[idx] = g_idx
        else:
            offsets[group[0]] = 0
            expanded_offsets[group[0]] = 0
            group_map[group[0]] = g_idx

    parts = []
    for i, s in enumerate(schools):
        u = s["university"]
        pos = positions[i]
        v_offset = offsets.get(i, 0)

        c = tier_color_for(tier_map.get(u, "冲刺"))
        n = profiles.get(u, {}).get("n_admitted", 0)
        sz = max(22, min(42, 20 + int(n / 30)))
        hardest = i == len(schools) - 1
        sel = u == selected
        ring = "3px" if hardest else "2px"
        sel_class = " hk-comp-sel" if sel else ""
        glow = f"box-shadow:0 0 0 4px {c}1a;" if sel else "box-shadow:0 1px 3px rgba(0,0,0,0.06);"

        logo = logo_b64(u)
        inner = (
            (
                f'<img src="data:image/png;base64,{logo}" style="width:{sz - 6}px;height:{sz - 6}px;'
                f'border-radius:50%;object-fit:contain" alt="{escape_html(u)}">'
            )
            if logo
            else (
                f'<span style="font-size:{max(9, sz // 3)}px;font-weight:700;color:{c}">'
                f"{escape_html(u[:1])}</span>"
            )
        )

        tier_label = tier_map.get(u, "")
        aria = f"{u} {tier_label}".strip()
        parts.append(
            f'<div class="hk-comp-dot{sel_class}" role="button" tabindex="0" '
            f'aria-label="{escape_html(aria)}" '
            f'data-school="{escape_html(u)}" '
            f'data-tier="{escape_html(tier_label)}" data-tiercolor="{c}" '
            f'data-group="{group_map.get(i, "")}" '
            f'data-default-top="{50 + offsets.get(i, 0)}%" '
            f'data-expanded-top="{50 + expanded_offsets.get(i, 0)}%" '
            f'title="{escape_html(u)}" '
            f'style="position:absolute;left:{pos}%;top:{50 + v_offset}%;'
            f'transform:translate(-50%,-50%);text-align:center;cursor:pointer">'
            f'<div class="hk-comp-circle" style="width:{sz}px;height:{sz}px;background:white;'
            f"border:{ring} solid {c};border-radius:50%;margin:0 auto;display:flex;"
            f'align-items:center;justify-content:center;{glow}">{inner}</div></div>'
        )

    hardest = schools[-1]
    hp = (
        100 - (clip_probability_coerce(hardest.get("probability")) - min_p) / p_range * 100
        if p_range > 0
        else 50
    )
    hp = max(5, min(95, hp))
    pointer = ""
    if hp > target_x + 5:
        pointer = (
            f'<div style="position:absolute;left:{hp}%;top:-34px">'
            '<div style="background:#ef4444;color:white;font-size:11px;font-weight:600;'
            "padding:3px 10px;border-radius:8px;white-space:nowrap;"
            'box-shadow:0 2px 8px rgba(239,68,68,0.35)">最大挑战</div>'
            '<div style="width:0;height:0;border-left:7px solid transparent;'
            'border-right:7px solid transparent;border-top:7px solid #ef4444;margin:0 auto"></div></div>'
        )

    bar_style = (
        "position:relative;height:68px;margin-top:40px;"
        "background:linear-gradient(to right,"
        f"rgba(34,197,94,0.12) 0%,rgba(34,197,94,0.12) {safety_x:.0f}%,"
        f"rgba(59,130,246,0.10) {safety_x:.0f}%,rgba(59,130,246,0.10) {target_x:.0f}%,"
        f"rgba(249,115,22,0.12) {target_x:.0f}%,rgba(239,68,68,0.14) 100%);"
        "border-radius:12px;margin-bottom:4px"
    )

    zones = (
        '<div style="display:flex;justify-content:space-between;font-size:10px;'
        'color:#94a3b8;padding:0 4px;margin-bottom:4px">'
        '<span style="color:#22c55e">← 稳妥</span><span style="color:#3b82f6">匹配</span>'
        '<span style="color:#f97316">冲刺</span><span style="color:#ef4444">高难 →</span></div>'
    )
    hint = (
        '<div style="text-align:center;font-size:10px;color:#cbd5e1;margin-bottom:8px">'
        "点击圆圈或下方院校标签切换</div>"
    )
    chips = _school_chips(schools, tier_map, selected)
    return (
        f'<div data-comp-bar style="{bar_style}">{"".join(parts)}{pointer}</div>'
        f"{zones}{hint}{chips}"
    )


def _school_chips(schools, tier_map, selected):
    chips = []
    for s in schools:
        u = s["university"]
        c = tier_color_for(tier_map.get(u, "冲刺"))
        sel = u == selected
        cls = "hk-comp-chip hk-comp-chip-sel" if sel else "hk-comp-chip"
        chips.append(
            f'<button type="button" class="{cls}" data-school="{escape_html(u)}" '
            f'aria-pressed="{"true" if sel else "false"}" style="--chip-color:{c}">'
            f'<span class="hk-comp-chip-dot" style="background:{c}"></span>{escape_html(u)}</button>'
        )
    return (
        '<div class="hk-comp-chips" role="group" aria-label="切换院校" style="margin-bottom:14px">'
        + "".join(chips)
        + "</div>"
    )
