"""HK Dashboard — CSS theme injection. Mirrors hr_dashboard/admin/styles.py."""

import streamlit as st


def inject_hk_dashboard_styles() -> None:
    st.html(f"""
    <style>
    /* ── Page chrome ── */
    .stMainBlockContainer {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}

    /* ── Headings ── */
    h1 {{
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        border-bottom: 2px solid #008a6c;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }}
    h2 {{
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        border-left: 4px solid #008a6c;
        padding-left: 0.75rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }}
    h3 {{
        font-size: 1.15rem;
        font-weight: 600;
        color: #4b5563;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }}

    /* ── Containers ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: #fafafa;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        padding: 1rem;
        margin-bottom: 1rem;
    }}

    /* ── Metrics ── */
    [data-testid="stMetricLabel"] {{
        font-size: 0.85rem;
        font-weight: 600;
        color: #374151;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #008a6c;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 0.78rem;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {{
        font-size: 1rem;
        font-weight: 600;
    }}

    /* ── DataFrames ── */
    .stDataFrame {{
        font-size: 0.82rem;
    }}

    /* ── Metric highlight card ── */
    .hk-metric-card {{
        background: linear-gradient(135deg, #f8fffe 0%, #f0fdf4 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }}
    .hk-metric-card .value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: #008a6c;
    }}
    .hk-metric-card .label {{
        font-size: 0.82rem;
        font-weight: 600;
        color: #374151;
        margin-top: 0.25rem;
    }}

    /* ── Funnel bar ── */
    .hk-funnel-bar {{
        height: 36px;
        border-radius: 6px;
        margin: 4px 0;
        display: flex;
        align-items: center;
        padding-left: 12px;
        color: white;
        font-size: 0.88rem;
        font-weight: 600;
    }}
    </style>
    """)
