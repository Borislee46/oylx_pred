from src.pages.prediction.result_display._comp_core import (
    FEATURE_LABELS,
    PROFILE_FEATURES,
    escape_html,
    tier_color_for,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce, clip_scalar

_logger = setup_logger("page3", "prediction")

FMT = {
    "gpa": lambda v: f"{v:.2f}",
    "language_score": lambda v: f"{v:.2f}",
    "research_count": lambda v: f"{v:.0f}段",
    "paper_count": lambda v: f"{v:.0f}篇",
    "internship_count": lambda v: f"{v:.0f}段",
    "award_count": lambda v: f"{v:.0f}项",
}

_AXIS_UNITS = 1.3


def _x(v, p50):
    if p50 <= 0:
        return 0.0
    return clip_scalar(v / p50 / _AXIS_UNITS * 100, 0.0, 100.0)


_COUNT_FEATURES = {"research_count", "paper_count", "internship_count", "award_count"}
_COUNT_FEATURE_NOTE = (
    '<div style="font-size:10px;color:#94a3b8;margin-top:8px;padding-left:4px">'
    "* 科研/实习差距含小数时，为数量与含金量（论文/项目质量、实习影响力）的综合评估，并非单纯计数</div>"
)


def gap_sections(schools, profiles, student, default_school, tier_map):
    _logger.info(
        "gap_sections: %d schools %d profiles default=%s",
        len(schools),
        len(profiles),
        default_school,
    )
    parts = []
    show_count_note = False
    skipped_no_profile = 0
    for s in schools:
        u = s["university"]
        profile = profiles.get(u)
        if not profile:
            skipped_no_profile += 1
            continue

        sp = clip_probability_coerce(s.get("probability"))
        vis = "block" if u == default_school else "none"

        rows = [
            f'<div class="hk-comp-gap" data-school="{escape_html(u)}" '
            'style="background:var(--hk-surface-elevated);border-radius:12px;padding:14px 16px;'
            f'border:1px solid var(--hk-border);margin-bottom:12px;display:{vis}">'
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">'
            f'<span style="font-size:13px;font-weight:700;color:var(--hk-slate-800)">{u}</span>'
            f'<span style="background:{tier_color_for(tier_map.get(u, "冲刺"))};color:white;font-size:11px;'
            f'font-weight:600;padding:2px 8px;border-radius:10px">{sp:.0%}</span>'
            '<span style="font-size:10px;color:#94a3b8">vs 录取者区间（灰带 P25–P75，虚线中位）</span></div>',
        ]

        for feat in PROFILE_FEATURES:
            fi = profile["features"].get(feat, {})
            p50 = fi.get("p50")
            sv = student.get(feat)
            if p50 is None or sv is None or p50 == 0:
                continue
            p25 = fi.get("p25")
            p75 = fi.get("p75")
            if p25 is None:
                p25 = p50
            if p75 is None:
                p75 = p50

            gap = sv - p50
            fm = FMT.get(feat, str)

            if sv >= p75:
                c, bg, st_text = "#34d399", "rgba(34,197,94,0.14)", f"超标 +{gap:.1f}"
            elif sv >= p50:
                c, bg, st_text = "#22c55e", "rgba(34,197,94,0.10)", "达标"
            elif sv >= p25:
                c, bg, st_text = "#22d3ee", "rgba(8,145,178,0.10)", "区间内"
            elif (p75 - p25) > 0 and sv >= p25 - (p75 - p25):
                c, bg, st_text = "#f97316", "rgba(249,115,22,0.10)", f"偏低 {abs(gap):.1f}"
            else:
                c, bg, st_text = "#ef4444", "rgba(239,68,68,0.10)", f"差 {abs(gap):.1f}"

            if feat in _COUNT_FEATURES and gap % 1 != 0:
                show_count_note = True

            x_sv = _x(sv, p50)
            x25 = _x(p25, p50)
            x75 = _x(p75, p50)
            x50 = _x(p50, p50)
            band_w = max(0.0, x75 - x25)
            band = (
                f'<div style="position:absolute;left:{x25:.0f}%;width:{band_w:.0f}%;top:0;height:22px;'
                f'background:rgba(148,163,184,0.16);border:1px dashed rgba(100,116,139,0.45);'
                f'border-radius:4px;box-sizing:border-box"></div>'
                if band_w > 0
                else ""
            )
            label = FEATURE_LABELS.get(feat, feat)
            rows.append(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
                f'<span style="width:42px;font-size:11px;font-weight:600;color:var(--hk-slate-500);'
                f'text-align:right;flex-shrink:0">{label}</span>'
                f'<div style="flex:1;height:22px;background:var(--hk-slate-100);border-radius:5px;'
                f'position:relative;overflow:hidden">'
                f"{band}"
                f'<div style="position:absolute;left:0;top:0;height:22px;width:{x_sv:.0f}%;'
                f'background:{bg};border-radius:5px"></div>'
                f'<div style="position:absolute;left:{x50:.0f}%;top:0;height:22px;width:0;'
                f'border-left:1px dashed #94a3b8"></div>'
                f'<div style="position:absolute;left:{x_sv:.0f}%;top:-1px;width:2px;'
                f'height:24px;background:{c};border-radius:1px"></div></div>'
                f'<span style="width:72px;font-size:10px;font-weight:600;color:{c};flex-shrink:0">'
                f"{fm(sv)} → {fm(p50)}</span>"
                f'<span style="width:40px;font-size:10px;font-weight:600;color:{c};flex-shrink:0">{st_text}</span>'
                "</div>"
            )
        rows.append("</div>")
        parts.append("".join(rows))

    if show_count_note:
        parts.append(_COUNT_FEATURE_NOTE)
    if skipped_no_profile:
        _logger.info("gap_sections: %d schools skipped (no profile data)", skipped_no_profile)
    return "".join(parts)


def narrative_sections(schools, profiles, student, default_school):
    parts = []
    hardest = schools[-1] if schools else {}
    best = schools[0] if schools else {}
    _h_prob = clip_probability_coerce(hardest.get("probability"))
    _b_prob = clip_probability_coerce(best.get("probability"))

    for s in schools:
        u = s["university"]
        profile = profiles.get(u)
        if not profile:
            continue
        n_admitted = profile.get("n_admitted", 0)
        strengths, weaknesses = [], []
        for feat in PROFILE_FEATURES:
            p50 = (profile.get("features") or {}).get(feat, {}).get("p50")
            sv = student.get(feat)
            if p50 is None or sv is None:
                continue
            gap = sv - p50
            label = FEATURE_LABELS.get(feat, feat)
            if gap >= 0:
                strengths.append((label, gap, feat))
            else:
                weaknesses.append((label, abs(gap), feat))
        if not weaknesses and not strengths:
            continue

        lines = []
        if strengths:
            names = "、".join(s[0] for s in strengths)
            lines.append(f"你的{names}已达到或超过{u}录取者中位水平，这是你的差异化优势。")
        if weaknesses:
            w = max(weaknesses, key=lambda x: x[1])
            feat_note = (
                "（含金量差异，不限于数量）" if w[2] in _COUNT_FEATURES and w[1] % 1 != 0 else ""
            )
            lines.append(
                f"以你当前水平，最需要提升的是{w[0]}（中位差距 {w[1]:.1f}）{feat_note}，建议优先投入。"
            )
        if 0 < n_admitted < 20:
            lines.append(f"注意：{u} 仅有 {n_admitted} 条历史录取样本，预测仅供参考。")

        vis = "block" if u == default_school else "none"
        parts.append(
            f'<div class="hk-comp-narrative" data-school="{escape_html(u)}" '
            f'style="font-size:12px;color:var(--hk-slate-500);margin-top:4px;display:{vis}">'
            f'{" | ".join(lines)}</div>'
        )
    return "".join(parts)
