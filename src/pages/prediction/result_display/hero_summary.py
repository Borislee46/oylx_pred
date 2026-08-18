from __future__ import annotations

import base64
import hashlib
import html
import json
from functools import lru_cache
from pathlib import Path

import streamlit as st

from src.agent.schemas import compute_tiers
from src.pages.prediction.data_facts import N_SAMPLES
from src.pages.prediction.school_logo_loader import (
    get_logo_path,
    get_school_url,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce, prob_round, prob_to_pct

_logger = setup_logger("page3", "prediction")

_MAX_VISIBLE_LOGOS = 5


@lru_cache(maxsize=64)
def _logo_base64(school: str) -> str | None:
    p = Path(get_logo_path(school))
    if not p.exists() or p.name == "product_logo.png":
        return None
    return base64.b64encode(p.read_bytes()).decode()


def _dedupe_by_university(candidates: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for c in candidates:
        u = c.get("university") or ""
        if not u:
            continue
        cur_p = clip_probability_coerce(c.get("probability"))
        prev = best.get(u)
        if prev is None or cur_p > clip_probability_coerce(prev.get("probability")):
            best[u] = c
    deduped = sorted(
        best.values(),
        key=lambda r: clip_probability_coerce(r.get("probability")),
        reverse=True,
    )
    if len(candidates) != len(deduped):
        _logger.info("hero dedup: %d candidates → %d unique schools", len(candidates), len(deduped))
    return deduped


def canonical_school_tiers(unified_results: list[dict]) -> dict[str, str]:
    if not unified_results:
        return {}

    best: dict[str, float] = {}
    for r in unified_results:
        uni = (r.get("university") or "").strip()
        if not uni:
            continue
        prob = clip_probability_coerce(r.get("probability"))
        if uni not in best or prob > best[uni]:
            best[uni] = prob

    fp_items = sorted((u, prob_round(p)) for u, p in best.items())
    fp = hashlib.md5(json.dumps(fp_items, sort_keys=True).encode()).hexdigest()

    cache: dict[str, dict[str, str]] = st.session_state.setdefault("_hk_canonical_tiers", {})
    if fp in cache:
        return cache[fp]

    schools = sorted(best.keys())
    probs = [best[u] for u in schools]
    labels = compute_tiers(probs)
    result = dict(zip(schools, labels, strict=False))

    _logger.info(
        "canonical_school_tiers: computed for %d schools — 保底=%d 适中=%d 冲刺=%d",
        len(schools),
        sum(1 for v in result.values() if v == "保底"),
        sum(1 for v in result.values() if v == "适中"),
        sum(1 for v in result.values() if v == "冲刺"),
    )
    cache[fp] = result
    return result


def _render_logo(school: dict) -> str:
    name = school.get("university") or ""
    name_esc = html.escape(str(name))
    b64 = _logo_base64(name)
    url = get_school_url(name)
    img = f'<img src="data:image/png;base64,{b64}" alt="">' if b64 else ""
    inner = img if img else html.escape((name or "?")[:1])
    classes = "hk-hero-logo" if img else "hk-hero-logo hk-hero-logo-initial"
    if url:
        return (
            f'<a class="{classes}" href="{url}" target="_blank" rel="noopener noreferrer" '
            f'title="{name_esc}">'
            f"{inner}</a>"
        )
    return f'<span class="{classes}" title="{name_esc}">{inner}</span>'


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


def _hero_tier_counts(
    candidates: list[dict],
    canonical_tiers: dict[str, str] | None = None,
) -> tuple[list[dict], int, dict[str, int]]:
    schools = _dedupe_by_university(candidates)
    if not schools:
        return [], 0, {"保底": 0, "适中": 0, "冲刺": 0}
    total = len(schools)
    tier_counts: dict[str, int] = {"保底": 0, "适中": 0, "冲刺": 0}
    if canonical_tiers:
        for s in schools:
            uni = s.get("university") or ""
            label = canonical_tiers.get(uni, "")
            if label in tier_counts:
                tier_counts[label] += 1
    else:
        probs = [clip_probability_coerce(s.get("probability")) for s in schools]
        tier_labels = compute_tiers(probs)
        for label in tier_labels:
            tier_counts[label] += 1
    return schools, total, tier_counts


def build_hero_logos_html(
    candidates: list[dict],
    canonical_tiers: dict[str, str] | None = None,
) -> tuple[str, int, dict[str, int]]:
    schools, total, tier_counts = _hero_tier_counts(candidates, canonical_tiers)
    if not schools:
        return "", 0, tier_counts
    visible = schools[:_MAX_VISIBLE_LOGOS]
    extra = max(0, total - _MAX_VISIBLE_LOGOS)
    logos = "".join(_render_logo(s) for s in visible)
    if extra > 0:
        logos += f'<span class="hk-hero-logo hk-hero-logo-more">+{extra}</span>'
    return logos, total, tier_counts


def _render_sales_tier_bar(tier_counts: dict[str, int], total: int) -> str:
    if total <= 0:
        return ""
    segs: list[str] = []
    labels: list[str] = []
    for tier, label, color in (
        ("保底", "保底", "#22c55e"),
        ("适中", "目标", "#3b82f6"),
        ("冲刺", "冲刺", "#f97316"),
    ):
        n = tier_counts.get(tier, 0)
        if n <= 0:
            continue
        pct = n / total * 100
        segs.append(
            f'<div class="hk-sales-tierbar-seg" style="flex:{pct:.1f};background:{color}"></div>'
        )
        labels.append(
            f'<span class="hk-sales-tierbar-label"><b style="color:{color}">{n}</b> {label}</span>'
        )
    if not segs:
        return ""
    return (
        '<div class="hk-sales-tierbar">'
        '<div class="hk-sales-tierbar-track">' + "".join(segs) + "</div>"
        '<div class="hk-sales-tierbar-labels">' + "".join(labels) + "</div>"
        "</div>"
    )


def build_sales_social_proof_html(candidates: list[dict]) -> str:
    total_records = 0
    combos_with_history = 0
    for r in candidates or []:
        n = int(r.get("_baseline_sample_count", 0) or 0)
        if n > 0:
            total_records += n
            combos_with_history += 1
    if total_records <= 0:
        return ""
    return (
        '<div class="hk-sales-proof">'
        '<span class="hk-sales-proof-dot"></span>'
        f"已有 <b>{total_records}</b> 份与你目标一致的真实申请记录，"
        f"支撑了其中 {combos_with_history} 个院校专业组合的测算"
        "</div>"
    )


def build_sales_hero_html(
    candidates: list[dict],
    *,
    combo_count: int,
    best_prob: float,
    encourage: str,
    canonical_tiers: dict[str, str] | None = None,
    include_prob_grid: bool = True,
) -> str:
    logos, total, tier_counts = build_hero_logos_html(candidates, canonical_tiers)
    tier_bar = _render_sales_tier_bar(tier_counts, total)

    primary_pct = f"{prob_to_pct(best_prob)}%"
    primary_label = "最高录取概率"
    primary_foot = "基于历史同类申请估算"

    safe_encourage = html.escape(encourage)
    logos_block = f'<div class="hk-hero-logos hk-sales-hero-logos">{logos}</div>' if logos else ""
    samples_str = f"{N_SAMPLES / 10000:.0f} 万+"
    prob_grid = (
        '<div class="hk-sales-prob-grid">'
        f'<div class="hk-sales-prob-card is-primary">'
        f'<div class="hk-sales-prob-num">{primary_pct}</div>'
        f'<div class="hk-sales-prob-label">{primary_label}</div>'
        f'<div class="hk-sales-prob-foot">{primary_foot}</div>'
        f"</div>"
        f'<div class="hk-sales-prob-card">'
        f'<div class="hk-sales-prob-num">{total}</div>'
        f'<div class="hk-sales-prob-label">推荐院校</div>'
        f"</div>"
        f'<div class="hk-sales-prob-card">'
        f'<div class="hk-sales-prob-num">{combo_count}</div>'
        f'<div class="hk-sales-prob-label">专业组合</div>'
        f"</div>"
        "</div>"
        if include_prob_grid
        else ""
    )

    return (
        '<div class="hk-sales-hero">'
        f'<div class="hk-sales-hero-eyebrow">Signals 已比对 {samples_str} 份真实亚英申请，'
        f"为你定位最匹配的院校组合</div>"
        f"{logos_block}"
        f'<div class="hk-hero-headline">为你匹配了 <b>{combo_count}</b> 个名校专业组合</div>'
        f"{tier_bar}"
        f'<div class="hk-sales-encourage">{safe_encourage}</div>'
        f"{prob_grid}"
    )


def fmt_pp_range(lo: float, hi: float) -> str:
    lo_pp = prob_to_pct(lo)
    hi_pp = prob_to_pct(hi)
    if hi_pp <= 0:
        return "—"
    if lo_pp == hi_pp:
        return f"+{hi_pp}pp"
    return f"+{lo_pp}~{hi_pp}pp"


_QUALITY_FIELD_ORDER: list[tuple[str, str]] = [
    ("paper_details", "论文"),
    ("award_details", "奖项"),
    ("internship_details", "实习"),
    ("research_details", "科研"),
]

_QUALITY_LEVEL_META: dict[str, dict[str, str]] = {
    "high": {"color": "#22c55e", "label": "可信"},
    "medium": {"color": "#f59e0b", "label": "待确认"},
    "low": {"color": "#94a3b8", "label": "弱信号"},
    "invalid": {"color": "#ef4444", "label": "不可用"},
}


def build_quality_badge_html(candidates: list[dict]) -> str:
    qs_raw: dict | None = None
    for c in candidates or []:
        qs = (c.get("_adjustment_trace") or {}).get("quality_signals")
        if qs:
            qs_raw = qs
            break
    if not qs_raw or not isinstance(qs_raw, dict):
        return ""

    raw_tags: dict[str, list[str]] = qs_raw.get("raw_tags", {}) if isinstance(qs_raw, dict) else {}
    llm_verified: dict[str, dict] = (
        qs_raw.get("llm_verified", {}) if isinstance(qs_raw, dict) else {}
    )

    cards: list[str] = []
    has_any_data = False
    for field, label in _QUALITY_FIELD_ORDER:
        lv = llm_verified.get(field) if isinstance(llm_verified, dict) else None
        if isinstance(lv, dict) and lv.get("verified_tags"):
            tags = lv["verified_tags"]
            quality_level = lv.get("quality_level", "")
        else:
            tags = raw_tags.get(field) or [] if isinstance(raw_tags, dict) else []
            quality_level = ""

        meta = _QUALITY_LEVEL_META.get(quality_level, {})
        accent_color = meta.get("color", "#e2e8f0")
        level_label = meta.get("label", "")

        if tags:
            has_any_data = True
            tag_text = html.escape(", ".join(tags))
            level_pill = (
                f'<span class="hk-quality-level-pill" style="background:{accent_color}15;color:{accent_color}">'
                f"{level_label}</span>"
                if level_label
                else ""
            )
            cards.append(
                f'<div class="hk-quality-card" style="--hk-quality-accent:{accent_color}">'
                f'<div class="hk-quality-card-head">'
                f'<span class="hk-quality-label">{label}</span>'
                f"{level_pill}"
                f"</div>"
                f'<div class="hk-quality-card-tags">{tag_text}</div>'
                f"</div>"
            )
        else:
            cards.append(
                '<div class="hk-quality-card is-empty">'
                f'<div class="hk-quality-card-head">'
                f'<span class="hk-quality-label">{label}</span>'
                f"</div>"
                f'<div class="hk-quality-card-empty">暂无</div>'
                f"</div>"
            )

    if not has_any_data:
        return ""

    concerns: list[str] = []
    for _field, lv in (llm_verified or {}).items():
        if isinstance(lv, dict) and lv.get("concern"):
            concerns.append(lv["concern"])

    concern_html = ""
    if concerns:
        concern_items = "".join(
            f'<span class="hk-quality-concern-item">{html.escape(c)}</span>' for c in concerns[:2]
        )
        concern_html = (
            '<div class="hk-quality-concerns">'
            f'<span class="hk-quality-concern-icon">⚠</span>'
            f'<span class="hk-quality-concern-text">{concern_items}</span>'
            "</div>"
        )

    return (
        '<div class="hk-quality-row">'
        '<div class="hk-quality-head-row">'
        '<span class="hk-quality-head">经历含金量</span>'
        '<span class="hk-quality-head-meta">AI 校验 · 仅供参考</span>'
        "</div>"
        f'<div class="hk-quality-grid">{"".join(cards)}</div>'
        f"{concern_html}"
        '<div class="hk-quality-disclaimer">'
        "含金量≠录取概率。经历标签经 AI 校验，系统无法量化其对录取的具体影响。"
        "</div>"
        "</div>"
    )


def render_hero_summary(
    all_candidates: list[dict],
    canonical_tiers: dict[str, str] | None = None,
) -> None:
    if not all_candidates:
        _logger.info("render_hero_summary: no candidates, skipping")
        return

    logos, total, tier_counts = build_hero_logos_html(all_candidates, canonical_tiers)
    if not logos:
        _logger.warning(
            "render_hero_summary: no logos generated for %d candidates", len(all_candidates)
        )
        return

    _logger.info(
        "render_hero_summary: %d schools — 保底=%d 目标=%d 冲刺=%d",
        total,
        tier_counts.get("保底", 0),
        tier_counts.get("适中", 0),
        tier_counts.get("冲刺", 0),
    )
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
