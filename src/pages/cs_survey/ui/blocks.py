from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

import streamlit as st

from ..layout import get_page_layout, layout_get


def render_html_block(markup: str) -> None:
    html_renderer = getattr(st, "html", None)
    if callable(html_renderer):
        html_renderer(markup)
        return
    st.markdown(markup, unsafe_allow_html=True)


def render_page_hero(
    title: str,
    subtitle: str,
    *,
    badge: str | None = None,
    eyebrow: str | None = None,
    summary: str | None = None,
    chips: Sequence[str] | None = None,
    stats: Sequence[dict[str, str]] | None = None,
    logo_path: str | None = None,
    logo_alt: str = "Company logo",
) -> None:
    chip_html = "".join(
        f'<span class="cs-hero-chip">{item}</span>' for item in (chips or []) if item
    )
    stat_html = "".join(
        f"""
        <div class="cs-hero-stat{' cs-hero-stat-accent' if item.get('accent') else ''}">
            <div class="cs-hero-stat-label">{item['label']}</div>
            <div class="cs-hero-stat-value">{item['value']}</div>
            <div class="cs-hero-stat-caption">{item.get('caption', '')}</div>
        </div>
        """
        for item in (stats or [])
    )
    eyebrow_html = f'<div class="cs-hero-eyebrow">{eyebrow}</div>' if eyebrow else ""
    summary_html = f'<p class="cs-hero-summary">{summary}</p>' if summary else ""
    chips_block = f'<div class="cs-hero-chip-row">{chip_html}</div>' if chip_html else ""
    stats_block = f'<div class="cs-hero-stat-grid">{stat_html}</div>' if stat_html else ""
    badge_html = f'<div class="cs-hero-badge">{badge}</div>' if badge else ""
    if logo_path:
        overview_layout = get_page_layout("overview")
        hero_columns = layout_get(overview_layout, "hero_columns", default=[1.55, 0.72])
        hero_gap = layout_get(overview_layout, "hero_gap", default="medium")
        left, right = st.columns(hero_columns, gap=hero_gap)
        with left:
            render_html_block(
                f"""
                <section class="cs-page-hero">
                    <div class="cs-page-hero-main">
                        <div>
                            {eyebrow_html}
                            <h1 class="cs-page-hero-title">{title}</h1>
                            <p class="cs-page-hero-subtitle">{subtitle}</p>
                            {summary_html}
                        </div>
                    </div>
                    {chips_block}
                    {stats_block}
                </section>
                """
            )
        with right:
            with st.container(border=True):
                render_html_block('<div class="cs-hero-brand-anchor"></div>')
                render_html_block(
                    f"""
                    <div class="cs-hero-brand-head">
                        <div class="cs-hero-brand-kicker">Brand Signature</div>
                        {badge_html}
                    </div>
                    """
                )
                st.image(logo_path, width="stretch")
                render_html_block(
                    f"""
                    <div class="cs-hero-brand-caption">{logo_alt}</div>
                    """
                )
        return

    render_html_block(
        f"""
        <section class="cs-page-hero">
            <div class="cs-page-hero-main">
                <div>
                    {eyebrow_html}
                    <h1 class="cs-page-hero-title">{title}</h1>
                    <p class="cs-page-hero-subtitle">{subtitle}</p>
                    {summary_html}
                </div>
                {badge_html}
            </div>
            {chips_block}
            {stats_block}
        </section>
        """
    )


def render_section_header(kicker: str, title: str, description: str | None = None) -> None:
    desc_html = f'<p class="cs-section-desc">{description}</p>' if description else ""
    render_html_block(
        f"""
        <div class="cs-section-header">
            <div class="cs-section-kicker">{kicker}</div>
            <h2 class="cs-section-title">{title}</h2>
            {desc_html}
        </div>
        """
    )


def render_info_cards(cards: Sequence[dict[str, str]], columns: int | None = None) -> None:
    if not cards:
        return
    count = columns or len(cards)
    cols = st.columns(count)
    for idx, card in enumerate(cards):
        with cols[idx % count]:
            tone = card.get("tone", "neutral")
            value = card.get("value")
            value_html = (
                f'<div class="cs-info-card-value">{value}</div>' if value is not None else ""
            )
            meta = card.get("meta")
            meta_html = f'<div class="cs-info-card-meta">{meta}</div>' if meta else ""
            badge = card.get("badge")
            badge_html = f'<div class="cs-info-card-badge">{badge}</div>' if badge else ""
            priority = card.get("priority")
            priority_html = (
                f'<div class="cs-info-card-priority">{priority}</div>' if priority else ""
            )
            render_html_block(
                f"""
                <div class="cs-info-card cs-info-card-{tone}">
                    <div class="cs-info-card-top">
                        <div class="cs-info-card-label">{card['label']}</div>
                        {badge_html}
                    </div>
                    {value_html}
                    <div class="cs-info-card-body">{card['body']}</div>
                    {priority_html}
                    {meta_html}
                </div>
                """
            )


@contextmanager
def panel(title: str, subtitle: str | None = None, *, badge: str | None = None):
    with st.container(border=True):
        badge_html = f'<div class="cs-panel-badge">{badge}</div>' if badge else ""
        subtitle_html = f'<p class="cs-panel-subtitle">{subtitle}</p>' if subtitle else ""
        render_html_block('<div class="cs-panel-anchor"></div>')
        render_html_block(
            f"""
            <div class="cs-panel-header">
                <div>
                    <p class="cs-panel-title">{title}</p>
                    {subtitle_html}
                </div>
                {badge_html}
            </div>
            """
        )
        yield
