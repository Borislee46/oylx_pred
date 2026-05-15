"""Hero summary banner above the prediction tables.

Renders a row of soft-colored overlapping school logos (Linear/Stripe-style
"customer wall" idiom) plus a tier breakdown line. Acts as the first thing
users see when results land — turning a cold table into a "the AI did this
for you" moment without waiting on ExplainAgent.

When ExplainAgent later finishes, ``overview`` text is rendered separately
in ``ai_report_sections`` so this banner stays static and instant.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

from src.agent.schemas import compute_tiers
from src.pages.prediction.admission_probability_calculator_components.school_logo_loader import (
    get_logo_path,
    get_school_url,
)
from src.pages.prediction.result_display._comp_core import compute_school_difficulty

_MAX_VISIBLE_LOGOS = 5


@lru_cache(maxsize=64)
def _logo_base64(school: str) -> str | None:
    p = Path(get_logo_path(school))
    if not p.exists() or p.name == "product_logo.png":
        return None
    return base64.b64encode(p.read_bytes()).decode()


def _dedupe_by_university(candidates: list[dict]) -> list[dict]:
    """Keep the highest-probability entry per university, sorted desc."""
    best: dict[str, dict] = {}
    for c in candidates:
        u = c.get("university") or ""
        if not u:
            continue
        cur_p = float(c.get("probability") or 0)
        prev = best.get(u)
        if prev is None or cur_p > float(prev.get("probability") or 0):
            best[u] = c
    return sorted(
        best.values(),
        key=lambda r: float(r.get("probability") or 0),
        reverse=True,
    )


def _render_logo(school: dict) -> str:
    name = school.get("university") or ""
    b64 = _logo_base64(name)
    url = get_school_url(name)
    img = f'<img src="data:image/png;base64,{b64}" alt="">' if b64 else ""
    inner = img if img else (name or "?")[:1]
    classes = "hk-hero-logo" if img else "hk-hero-logo hk-hero-logo-initial"
    if url:
        return (
            f'<a class="{classes}" href="{url}" target="_blank" rel="noopener noreferrer" title="{name}">'
            f"{inner}</a>"
        )
    return f'<span class="{classes}" title="{name}">{inner}</span>'


def _render_breakdown(tier_counts: dict[str, int], total: int) -> str:
    parts: list[str] = []
    for tier, label, css in (
        ("保底", "保底", "hk-tier-safety-num"),
        ("适中", "目标", "hk-tier-target-num"),
        ("冲刺", "冲刺", "hk-tier-reach-num"),
    ):
        n = tier_counts.get(tier, 0)
        if n:
            parts.append(f'<b class="{css}">{n}</b> {label}')
    if not parts:
        return f"{total} 所院校"
    return '<span class="hk-hero-dot">·</span>'.join(parts)


def render_hero_summary(all_candidates: list[dict], cases_df=None) -> None:
    """Render the hero summary banner.

    No-op when candidate list is empty so this is safe to call at the top of
    every render path.  When *cases_df* is provided, uses difficulty-weighted
    tier thresholds per school.
    """
    if not all_candidates:
        return

    schools = _dedupe_by_university(all_candidates)
    if not schools:
        return

    total = len(schools)
    probs = [float(s.get("probability") or 0) for s in schools]

    difficulties: dict[int, float] | None = None
    if cases_df is not None:
        diff_map = compute_school_difficulty(cases_df)
        if diff_map:
            difficulties = {i: diff_map.get(s.get("university", ""), 0.5)
                            for i, s in enumerate(schools)}

    tier_labels = compute_tiers(probs, difficulties=difficulties)
    tier_counts: dict[str, int] = {"保底": 0, "适中": 0, "冲刺": 0}
    for label in tier_labels:
        tier_counts[label] += 1

    visible = schools[:_MAX_VISIBLE_LOGOS]
    extra = max(0, total - _MAX_VISIBLE_LOGOS)

    logos = "".join(_render_logo(s) for s in visible)
    if extra > 0:
        logos += f'<span class="hk-hero-logo hk-hero-logo-more">+{extra}</span>'

    breakdown = _render_breakdown(tier_counts, total)

    st.html(
        '<div class="hk-hero-summary">'
        f'<div class="hk-hero-logos">{logos}</div>'
        '<div class="hk-hero-text">'
        f'<div class="hk-hero-headline">Signals 系统已为您筛选 <b>{total}</b> 所院校</div>'
        f'<div class="hk-hero-breakdown">{breakdown}</div>'
        "</div>"
        "</div>"
    )
