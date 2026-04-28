"""HK Dashboard — professional CSS theme."""

import streamlit as st


def inject_hk_dashboard_styles() -> None:
    st.html("""
    <style>
    /* ── Root / page ── */
    .stMainBlockContainer {
        padding: 0.5rem 1.5rem 2rem 1.5rem;
    }
    section[data-testid="stSidebar"] { display: none; }

    /* ── Typography ── */
    h1 {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1e293b;
        padding: 0.5rem 0 0.75rem 0;
        margin: 0 0 1.25rem 0;
        border-bottom: 1px solid #e2e8f0;
        letter-spacing: 0.02em;
    }
    h2 {
        font-size: 1.15rem;
        font-weight: 600;
        color: #334155;
        margin: 1.25rem 0 0.75rem 0;
        padding-left: 0.6rem;
        border-left: 3px solid #2563eb;
    }
    h3 {
        font-size: 0.95rem;
        font-weight: 600;
        color: #475569;
        margin: 0.75rem 0 0.5rem 0;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        font-weight: 500;
        color: #64748b;
        letter-spacing: 0.03em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* ── Custom KPI cards ── */
    .hk-kpi-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 0.5rem 0 1rem 0;
    }
    .hk-kpi-card {
        flex: 1 1 180px;
        min-width: 150px;
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.15s;
    }
    .hk-kpi-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .hk-kpi-card .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .hk-kpi-card .kpi-label {
        font-size: 0.72rem;
        font-weight: 500;
        color: #64748b;
        letter-spacing: 0.04em;
        margin-top: 0.3rem;
    }
    .hk-kpi-card .kpi-delta {
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .hk-kpi-card.accent { border-top: 3px solid #2563eb; }
    .hk-kpi-card.positive { border-top: 3px solid #059669; }
    .hk-kpi-card.warning { border-top: 3px solid #d97706; }

    /* ── Section cards ── */
    .hk-section {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .hk-section h3 { margin-top: 0; border: none; padding: 0; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 500;
        padding: 0.5rem 1rem;
        color: #64748b;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: #2563eb;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        font-size: 0.8rem;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }
    .stDataFrame thead th {
        background: #f8fafc;
        color: #475569;
        font-weight: 600;
        font-size: 0.75rem;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    /* ── Containers ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    /* ── Chart captions ── */
    .stCaption { color: #94a3b8; font-size: 0.72rem; }

    /* ── Search inputs ── */
    input[type="text"] { border: 1px solid #e2e8f0; border-radius: 6px; }
    </style>
    """)
