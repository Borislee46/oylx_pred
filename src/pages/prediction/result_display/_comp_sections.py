"""Pre-rendered gap bars and narratives for all schools (client-side switching)."""

from src.pages.prediction.result_display._comp_core import (
    FEATURE_LABELS,
    PROFILE_FEATURES,
    escape_html,
    safe_float,
    tier_color_for,
)

FMT = {
    "gpa": lambda v: f"{v:.2f}",
    "language_score": lambda v: f"{v:.2f}",
    "research_count": lambda v: f"{v:.0f}段",
    "internship_count": lambda v: f"{v:.0f}段",
}


def gap_sections(schools, profiles, student, default_school, tier_map):
    """Pre-render gap HTML for every school. Only default_school is visible."""
    parts = []
    for s in schools:
        u = s["university"]
        profile = profiles.get(u)
        if not profile:
            continue
        sp = next((safe_float(x.get("probability")) for x in schools if x["university"] == u), 0)
        vis = "block" if u == default_school else "none"
        rows = [
            f'<div class="hk-comp-gap" data-school="{escape_html(u)}" '
            'style="background:#f8fafc;border-radius:12px;padding:14px 16px;'
            f'border:1px solid #e2e8f0;margin-bottom:12px;display:{vis}">'
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">'
            f'<span style="font-size:13px;font-weight:700;color:#1e293b">{u}</span>'
            f'<span style="background:{tier_color_for(tier_map.get(u, "冲刺"))};color:white;font-size:11px;'
            f'font-weight:600;padding:2px 8px;border-radius:10px">{sp:.0%}</span>'
            '<span style="font-size:10px;color:#94a3b8">vs 录取者中位水平</span></div>',
        ]
        for feat in PROFILE_FEATURES:
            fi = profile["features"].get(feat, {})
            p50 = fi.get("p50")
            sv = student.get(feat)
            if p50 is None or sv is None or p50 == 0:
                continue
            ratio = min(1.0, max(0.05, sv / p50))
            gap = sv - p50
            fm = FMT.get(feat, str)
            if gap >= 0:
                c, bg, st_text, sc = "#22c55e", "rgba(34,197,94,0.12)", "达标", "#22c55e"
            elif gap >= -0.2 * p50:
                c, bg, st_text, sc = "#f97316", "rgba(249,115,22,0.10)", f"差 {abs(gap):.1f}", "#f97316"
            else:
                c, bg, st_text, sc = "#ef4444", "rgba(239,68,68,0.10)", f"差 {abs(gap):.1f}", "#ef4444"
            label = FEATURE_LABELS.get(feat, feat)
            rows.append(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
                f'<span style="width:42px;font-size:11px;font-weight:600;color:#64748b;'
                f'text-align:right;flex-shrink:0">{label}</span>'
                f'<div style="flex:1;height:22px;background:#f1f5f9;border-radius:5px;'
                f'position:relative;overflow:hidden">'
                f'<div style="position:absolute;left:0;top:0;height:22px;width:{ratio*100:.0f}%;'
                f'background:{bg};border-radius:5px"></div>'
                f'<div style="position:absolute;left:{ratio*100:.0f}%;top:-1px;width:2px;'
                f'height:24px;background:{c};border-radius:1px"></div></div>'
                f'<span style="width:72px;font-size:10px;font-weight:600;color:{c};flex-shrink:0">'
                f'{fm(sv)} → {fm(p50)}</span>'
                f'<span style="width:40px;font-size:10px;font-weight:600;color:{sc};flex-shrink:0">{st_text}</span>'
                "</div>"
            )
        rows.append("</div>")
        parts.append("".join(rows))
    return "".join(parts)


def narrative_sections(schools, profiles, student, default_school):
    """Pre-render narrative text for every school, only default visible."""
    parts = []
    hardest = schools[-1] if schools else {}
    best = schools[0] if schools else {}
    h_prob = safe_float(hardest.get("probability"))
    b_prob = safe_float(best.get("probability"))
    h_n = profiles.get(hardest.get("university", ""), {}).get("n_admitted", 0)
    for s in schools:
        u = s["university"]
        profile = profiles.get(u)
        if not profile:
            continue
        strengths, weaknesses = [], []
        for feat in PROFILE_FEATURES:
            p50 = (profile.get("features") or {}).get(feat, {}).get("p50")
            sv = student.get(feat)
            if p50 is None or sv is None:
                continue
            gap = sv - p50
            label = FEATURE_LABELS.get(feat, feat)
            if gap >= 0:
                strengths.append((label, gap))
            else:
                weaknesses.append((label, abs(gap)))
        if not weaknesses and not strengths:
            continue
        lines = []
        if strengths:
            names = "、".join(s[0] for s in strengths)
            lines.append(f"你的{names}已达到或超过{u}录取者中位水平，这是你的差异化优势。")
        if weaknesses:
            w = max(weaknesses, key=lambda x: x[1])
            lines.append(f"以你当前水平，最需要提升的是{w[0]}（中位差距 {w[1]:.1f}），建议优先投入。")
            lines.append(
                f"你目前录取概率区间为 {h_prob:.0%} — {b_prob:.0%}，"
                f"提升{w[0]}可获得最大边际增益。"
            )
        if 0 < h_n < 20:
            lines.append(
                f"注意：{hardest.get('university','')} 仅有 {h_n} 条历史录取样本，预测结果仅供参考。"
            )
        vis = "block" if u == default_school else "none"
        parts.append(
            f'<div class="hk-comp-narrative" data-school="{escape_html(u)}" '
            f'style="font-size:12px;color:#64748b;margin-top:4px;display:{vis}">'
            f'{" | ".join(lines)}</div>'
        )
    return "".join(parts)
