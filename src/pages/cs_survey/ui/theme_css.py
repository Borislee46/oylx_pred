from __future__ import annotations

import streamlit as st

from ..layout import get_global_layout, layout_get

THEME = {
    "primary": "#006B67",
    "primary_dark": "#00514E",
    "accent": "#F5A623",
    "success": "#0F766E",
    "danger": "#B91C1C",
    "text": "#111827",
    "muted": "#6B7280",
    "border": "#D5E3E1",
    "divider": "#E3ECEB",
    "card": "#FFFFFF",
    "bg": "#F4F8F7",
    "bg_soft": "#F8FBFA",
    "shadow": "0 10px 24px rgba(15, 23, 42, 0.06)",
}


def _g(*keys: str, default: str) -> str:
    return str(layout_get(get_global_layout(), *keys, default=default))


def _base_css() -> str:
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --cs-primary: {THEME["primary"]};
    --cs-primary-dark: {THEME["primary_dark"]};
    --cs-accent: {THEME["accent"]};
    --cs-success: {THEME["success"]};
    --cs-danger: {THEME["danger"]};
    --cs-text: {THEME["text"]};
    --cs-muted: {THEME["muted"]};
    --cs-border: {THEME["border"]};
    --cs-divider: {THEME["divider"]};
    --cs-card: {THEME["card"]};
    --cs-bg: {THEME["bg"]};
    --cs-bg-soft: {THEME["bg_soft"]};
    --cs-shadow: {THEME["shadow"]};
}}

html, body, [class*="css"] {{
    font-family: "Inter", "Segoe UI", "Microsoft YaHei", sans-serif;
    color: var(--cs-text);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {{
    background: linear-gradient(180deg, #F8FAFC 0%, {THEME["bg"]} 100%) !important;
}}

[data-testid="stMainBlockContainer"] {{
    padding: {_g("main_container", "padding", default="1rem 1.2rem 1.1rem 1.2rem")} !important;
    max-width: {_g("main_container", "max_width", default="100%")} !important;
}}

[data-testid="stCaptionContainer"], .stCaption {{
    font-size: 11.5px !important;
    color: var(--cs-muted) !important;
    line-height: 1.55 !important;
}}

[data-testid="stMarkdownContainer"] h5 {{
    font-size: 12px !important;
    font-weight: 700 !important;
    color: var(--cs-primary-dark) !important;
    margin: 0 0 0.35rem 0 !important;
}}

div[data-testid="stHorizontalBlock"] {{
    gap: {_g("horizontal_block", "gap", default="0.9rem")} !important;
    align-items: stretch !important;
}}

div[data-testid="stColumn"] > div {{
    width: 100% !important;
}}

div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label p {{
    font-size: 10.5px !important;
    font-weight: 600 !important;
    color: var(--cs-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

div[data-baseweb="select"] > div {{
    min-height: 38px !important;
    border-radius: 10px !important;
    border: 1px solid var(--cs-border) !important;
    background: rgba(255, 255, 255, 0.92) !important;
    box-shadow: none !important;
}}

div[data-baseweb="select"] span {{
    font-size: 12.5px !important;
    color: var(--cs-text) !important;
}}

div[data-testid="stRadio"] > div {{
    gap: 0.4rem !important;
}}

div[data-testid="stRadio"] label {{
    min-height: 36px !important;
    border-radius: 999px !important;
    border: 1px solid var(--cs-border) !important;
    background: rgba(255, 255, 255, 0.78) !important;
    padding: 0.15rem 0.9rem !important;
}}

div[data-testid="stRadio"] label:has(input:checked) {{
    border-color: rgba(0, 107, 103, 0.22) !important;
    background: linear-gradient(180deg, rgba(0, 107, 103, 0.12) 0%, rgba(0, 107, 103, 0.04) 100%) !important;
}}

[data-testid="stMetric"] {{
    background: rgba(255, 255, 255, 0.92) !important;
    border: 1px solid var(--cs-border) !important;
    border-radius: 16px !important;
    padding: 0.75rem 0.95rem !important;
    box-shadow: var(--cs-shadow) !important;
}}

[data-testid="stMetricLabel"] {{
    color: var(--cs-muted) !important;
    font-size: 10.5px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

[data-testid="stMetricValue"] {{
    color: var(--cs-text) !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.05 !important;
}}

[data-testid="stTabs"] {{
    margin-top: 0.2rem;
}}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 0.45rem;
}}

[data-testid="stTabs"] [data-baseweb="tab"] {{
    min-height: 38px;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    border: 1px solid var(--cs-border);
    background: rgba(255, 255, 255, 0.8);
    color: var(--cs-muted);
    font-size: 11.5px;
    font-weight: 600;
}}

[data-testid="stTabs"] [aria-selected="true"] {{
    border-color: rgba(0, 107, 103, 0.18);
    background: rgba(0, 107, 103, 0.08);
    color: var(--cs-primary-dark);
}}

[data-testid="stDataFrame"] {{
    border: 1px solid var(--cs-border);
    border-radius: 14px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.92);
}}

[data-testid="stDataFrame"] div[role="columnheader"] {{
    background: var(--cs-bg-soft) !important;
    color: var(--cs-text) !important;
    font-weight: 700 !important;
    font-size: 10.5px !important;
    border-bottom: 1px solid var(--cs-divider) !important;
}}

[data-testid="stDataFrame"] div[role="gridcell"] {{
    font-size: 11.5px !important;
    line-height: 1.5 !important;
}}

.cs-page-hero {{
    padding: {_g("hero", "padding", default="1.15rem 1.2rem 1rem 1.2rem")};
    border: 1px solid rgba(0, 107, 103, 0.12);
    border-radius: 24px;
    background:
        radial-gradient(circle at top right, rgba(0, 107, 103, 0.12), transparent 34%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(244, 247, 251, 0.98) 100%);
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    margin-bottom: {_g("hero", "margin_bottom", default="0.95rem")};
}}

.cs-page-hero-main {{
    display: flex;
    justify-content: space-between;
    gap: {_g("hero_main", "gap", default="1rem")};
    align-items: flex-start;
}}

.cs-hero-eyebrow {{
    font-size: 11px;
    font-weight: 700;
    color: var(--cs-primary-dark);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}}

.cs-page-hero-title {{
    margin: 0;
    color: var(--cs-text);
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.04em;
}}

.cs-page-hero-subtitle {{
    margin: 0.4rem 0 0 0;
    color: var(--cs-muted);
    font-size: 12.5px;
    line-height: 1.55;
}}

.cs-hero-summary {{
    margin: {_g("hero_summary", "margin_top", default="0.7rem")} 0 0 0;
    max-width: {_g("hero_summary", "max_width", default="880px")};
    color: var(--cs-text);
    font-size: 12.5px;
    line-height: 1.65;
}}

.cs-hero-badge {{
    display: inline-flex;
    align-items: center;
    min-height: 34px;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    border: 1px solid rgba(0, 107, 103, 0.14);
    background: rgba(255, 255, 255, 0.88);
    color: var(--cs-primary-dark);
    font-size: 10.5px;
    font-weight: 700;
    white-space: nowrap;
}}

.cs-hero-chip-row,
.cs-hero-stat-grid {{
    display: grid;
    gap: {_g("hero_chip_row", "gap", default="0.7rem")};
}}

.cs-hero-chip-row {{
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    margin-top: {_g("hero_chip_row", "margin_top", default="0.9rem")};
}}

.cs-hero-chip {{
    display: inline-flex;
    align-items: center;
    min-height: 38px;
    padding: 0.5rem 0.85rem;
    border-radius: 14px;
    border: 1px solid rgba(0, 107, 103, 0.1);
    background: rgba(255, 255, 255, 0.78);
    color: var(--cs-text);
    font-size: 11.5px;
    font-weight: 600;
}}

.cs-hero-stat-grid {{
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: {_g("hero_stat_grid", "gap", default="0.7rem")};
    margin-top: {_g("hero_stat_grid", "margin_top", default="0.9rem")};
}}

.cs-hero-stat {{
    min-height: 114px;
    padding: 0.8rem 0.9rem;
    border-radius: 18px;
    border: 1px solid rgba(214, 222, 232, 0.9);
    background: rgba(255, 255, 255, 0.86);
}}

.cs-hero-stat-accent {{
    border-color: rgba(0, 107, 103, 0.16);
    background: linear-gradient(180deg, rgba(236, 253, 250, 0.95) 0%, rgba(255, 255, 255, 0.92) 100%);
}}

.cs-hero-stat-label,
.cs-info-card-label {{
    color: var(--cs-muted);
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}

.cs-hero-stat-value,
.cs-info-card-value {{
    margin-top: 0.45rem;
    color: var(--cs-text);
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.05;
}}

.cs-hero-stat-caption,
.cs-info-card-meta {{
    margin-top: 0.45rem;
    color: var(--cs-muted);
    font-size: 11px;
    line-height: 1.5;
}}

.cs-section-header {{
    margin: {_g("section_header", "margin", default="1.05rem 0 0.65rem 0")};
}}

.cs-section-kicker {{
    color: var(--cs-primary-dark);
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.cs-section-title {{
    margin: 0.15rem 0 0 0;
    color: var(--cs-text);
    font-size: 1.02rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}}

.cs-section-desc {{
    margin: {_g("section_desc", "margin", default="0.28rem 0 0 0")};
    max-width: {_g("section_desc", "max_width", default="780px")};
    color: var(--cs-muted);
    font-size: 11.5px;
    line-height: 1.6;
}}

.cs-info-card {{
    height: 100%;
    min-height: {_g("info_card", "min_height", default="140px")};
    padding: {_g("info_card", "padding", default="0.9rem 1rem")};
    border-radius: 18px;
    border: 1px solid var(--cs-border);
    background: rgba(255, 255, 255, 0.92);
    box-shadow: var(--cs-shadow);
}}

.cs-info-card-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: {_g("info_card_top", "gap", default="0.6rem")};
}}

.cs-info-card-badge {{
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    padding: 0.12rem 0.55rem;
    border-radius: 999px;
    background: rgba(0, 107, 103, 0.08);
    color: var(--cs-primary-dark);
    font-size: 10px;
    font-weight: 700;
    white-space: nowrap;
}}

.cs-info-card-body {{
    margin-top: {_g("info_card_body", "margin_top", default="0.55rem")};
    color: var(--cs-text);
    font-size: 11.5px;
    line-height: 1.65;
}}

.cs-info-card-priority {{
    margin-top: {_g("info_card_priority", "margin_top", default="0.55rem")};
    color: var(--cs-text);
    font-size: 11px;
    font-weight: 700;
    line-height: 1.5;
}}

.cs-info-card-neutral {{
    background: rgba(255, 255, 255, 0.92);
}}

.cs-info-card-positive {{
    border-color: rgba(15, 118, 110, 0.18);
    background: linear-gradient(180deg, rgba(240, 253, 250, 0.96) 0%, rgba(255, 255, 255, 0.92) 100%);
}}

.cs-info-card-warning {{
    border-color: rgba(245, 158, 11, 0.2);
    background: linear-gradient(180deg, rgba(255, 251, 235, 0.96) 0%, rgba(255, 255, 255, 0.92) 100%);
}}

.cs-info-card-danger {{
    border-color: rgba(185, 28, 28, 0.16);
    background: linear-gradient(180deg, rgba(254, 242, 242, 0.96) 0%, rgba(255, 255, 255, 0.92) 100%);
}}

.pbi-badge,
.cs-detail-badge,
.cs-filter-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    min-height: 30px;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    border: 1px solid rgba(0, 107, 103, 0.12);
    background: rgba(255, 255, 255, 0.75);
    color: var(--cs-primary-dark);
    font-size: 11px;
    font-weight: 600;
}}

.cs-detail-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    justify-content: flex-end;
}}

.pbi-seg-card,
.stPageLink,
.cs-panel-shell {{
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid var(--cs-border);
    border-radius: 16px;
    box-shadow: var(--cs-shadow);
}}

.pbi-seg-card {{
    padding: {_g("segment_card", "padding", default="1rem")};
    height: 100%;
}}

.pbi-seg-title {{
    margin: 0 0 0.55rem 0;
    font-size: 12px;
    font-weight: 700;
    color: var(--cs-text);
}}

.pbi-seg-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: {_g("segment_grid", "gap", default="0.5rem 1rem")};
}}

.pbi-seg-lbl {{
    font-size: 10px;
    color: var(--cs-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.pbi-seg-val {{
    margin-top: 0.15rem;
    font-size: 17px;
    font-weight: 700;
    color: var(--cs-primary-dark);
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-panel-anchor) {{
    background: rgba(255, 255, 255, 0.92) !important;
    border: 1px solid var(--cs-border) !important;
    border-radius: {_g("panel", "border_radius", default="20px")} !important;
    padding: {_g("panel", "padding", default="0.8rem 0.95rem 0.85rem 0.95rem")} !important;
    box-shadow: var(--cs-shadow) !important;
    height: 100% !important;
}}

.cs-panel-header {{
    display: flex;
    justify-content: space-between;
    gap: {_g("panel_header", "gap", default="0.8rem")};
    align-items: flex-start;
    margin-bottom: {_g("panel_header", "margin_bottom", default="0.65rem")};
}}

.cs-panel-title {{
    margin: 0;
    color: var(--cs-text);
    font-size: 12px;
    font-weight: 700;
    line-height: 1.35;
}}

.cs-panel-subtitle {{
    margin: 0.22rem 0 0 0;
    color: var(--cs-muted);
    font-size: 11px;
    line-height: 1.5;
    min-height: 2.8em;
}}

.cs-panel-badge {{
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(0, 107, 103, 0.12);
    background: rgba(236, 253, 250, 0.88);
    color: var(--cs-primary-dark);
    font-size: 10px;
    font-weight: 700;
    white-space: nowrap;
}}

.stPageLink {{
    padding: {_g("view_nav", "padding", default="0.8rem 0.95rem")};
    transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
}}

.stPageLink:hover {{
    transform: translateY(-1px);
    border-color: rgba(0, 107, 103, 0.28);
    background: rgba(236, 253, 250, 0.88);
}}

.stPageLink a {{
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--cs-primary-dark) !important;
}}

.cs-view-nav-title {{
    margin: 0.3rem 0 0.45rem 0;
    color: var(--cs-muted);
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}

div.stButton > button {{
    min-height: 38px !important;
    border-radius: 14px !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    border: 1px solid var(--cs-border) !important;
    background: rgba(255, 255, 255, 0.9) !important;
    box-shadow: none !important;
}}

div.stButton > button[kind="primary"] {{
    border-color: rgba(0, 107, 103, 0.2) !important;
    background: linear-gradient(180deg, rgba(0, 107, 103, 0.14) 0%, rgba(0, 107, 103, 0.08) 100%) !important;
    color: var(--cs-primary-dark) !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-hero-brand-anchor) {{
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 251, 250, 0.98) 100%) !important;
    border: 1px solid rgba(0, 107, 103, 0.12) !important;
    border-radius: 24px !important;
    padding: {_g("hero_brand", "padding", default="0.95rem 1rem 0.85rem 1rem")} !important;
    box-shadow: var(--cs-shadow) !important;
    min-height: 100% !important;
}}

.cs-hero-brand-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: {_g("hero_brand", "gap", default="0.6rem")};
    margin-bottom: {_g("hero_brand", "margin_bottom", default="0.6rem")};
}}

.cs-hero-brand-kicker {{
    color: var(--cs-primary-dark);
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.cs-hero-brand-caption {{
    margin-top: 0.55rem;
    color: var(--cs-muted);
    font-size: 11px;
    line-height: 1.5;
    text-align: left;
}}

.cs-empty-state {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: 92px;
    padding: {_g("empty_state", "padding", default="0.85rem 1rem")};
    border: 1px dashed rgba(107, 114, 128, 0.32);
    border-radius: 14px;
    background: rgba(248, 250, 252, 0.88);
    color: var(--cs-muted);
    font-size: 11.5px;
    text-align: center;
}}
"""


def _overview_css() -> str:
    return f"""
.cs-page-hero {{
    margin-bottom: {_g("overview_css", "hero_margin_bottom", default="1rem")};
}}
"""


def _detail_css() -> str:
    return f"""
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-filter-bar-anchor) {{
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(244, 247, 251, 0.96) 100%) !important;
    border: 1px solid rgba(0, 107, 103, 0.14) !important;
    border-radius: 18px !important;
    padding: {_g("filter_bar", "padding", default="0.7rem 0.9rem 0.6rem 0.9rem")} !important;
    margin-bottom: {_g("filter_bar", "margin_bottom", default="0.8rem")} !important;
    box-shadow: var(--cs-shadow) !important;
}}

.pbi-filter-title {{
    margin: {_g("filter_bar", "title_margin", default="0 0 0.6rem 0")};
    color: var(--cs-primary-dark);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.cs-page-hero {{
    margin-bottom: {_g("detail_css", "hero_margin_bottom", default="0.85rem")};
}}
"""


def _inject(css: str) -> None:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def inject_base_css() -> None:
    _inject(_base_css())


def inject_overview_css() -> None:
    _inject(_overview_css())


def inject_detail_css() -> None:
    _inject(_detail_css())


def format_metric(v: object, fmt: str = "{:.2f}", empty: str = "—") -> str:
    if v is None:
        return empty
    if isinstance(v, float) and v != v:
        return empty
    return fmt.format(v)
