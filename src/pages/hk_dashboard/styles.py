"""HK Dashboard — premium styling for startup ops dashboard."""

import streamlit as st


def inject_hk_dashboard_styles() -> None:
    st.html("""
    <style>
    /* ── Root ── */
    .stMainBlockContainer { padding: 0.75rem 2rem 2.5rem 2rem; }
    .stApp { background: #f8fafc; }

    /* ── Title ── */
    h1 {
        font-size: 1.4rem; font-weight: 600; color: #0f172a;
        padding: 0.25rem 0 0.5rem 0; margin: 0 0 0.75rem 0;
        border-bottom: 1px solid #e2e8f0; letter-spacing: 0.01em;
    }

    /* ── Section headers ── */
    h2 {
        font-size: 0.95rem; font-weight: 600; color: #334155;
        margin: 1.5rem 0 0.6rem 0; padding: 0;
        text-transform: uppercase; letter-spacing: 0.06em;
    }
    h3 {
        font-size: 0.82rem; font-weight: 600; color: #64748b;
        margin: 0.6rem 0 0.4rem 0; text-transform: uppercase; letter-spacing: 0.05em;
    }

    /* ── KPI row ── */
    .hk-kpi-row {
        display: flex; gap: 14px; flex-wrap: wrap; margin: 0 0 0.5rem 0;
    }
    .hk-kpi-card {
        flex: 1 1 170px; min-width: 140px;
        background: #fff; border: 1px solid #f1f5f9;
        border-radius: 8px; padding: 0.9rem 1.1rem;
    }
    .hk-kpi-card .kpi-value {
        font-size: 1.5rem; font-weight: 700; color: #0f172a; line-height: 1.2;
    }
    .hk-kpi-card .kpi-label {
        font-size: 0.7rem; font-weight: 500; color: #94a3b8;
        letter-spacing: 0.05em; margin-top: 0.25rem; text-transform: uppercase;
    }
    .hk-kpi-card .kpi-sub {
        font-size: 0.72rem; color: #64748b; margin-top: 0.2rem;
    }

    /* ── KPI colors ── */
    .hk-kpi-card.t-blue  { border-left: 3px solid #2563eb; }
    .hk-kpi-card.t-green { border-left: 3px solid #0d9488; }
    .hk-kpi-card.t-amber { border-left: 3px solid #d97706; }
    .hk-kpi-card.t-slate { border-left: 3px solid #64748b; }

    /* ── Metric cards (st.metric) ── */
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #f1f5f9; border-radius: 8px;
        padding: 0.6rem 0.9rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem; font-weight: 500; color: #94a3b8; letter-spacing: 0.04em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem; font-weight: 700; color: #0f172a;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem; font-weight: 500; padding: 0.4rem 1rem;
        color: #64748b; border-radius: 0; margin-right: 0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb; font-weight: 600;
        border-bottom: 2px solid #2563eb;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent; }

    /* ── DataFrames ── */
    .stDataFrame {
        font-size: 0.78rem; border: 1px solid #f1f5f9; border-radius: 6px;
    }
    .stDataFrame thead th {
        background: #f8fafc; color: #64748b; font-weight: 600; font-size: 0.72rem;
        text-transform: uppercase; letter-spacing: 0.03em;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        border: 1px solid #f1f5f9; border-radius: 8px; margin: 0.4rem 0;
        background: #fff;
    }

    /* ── Containers with border ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff; border-radius: 8px; border: 1px solid #f1f5f9;
        padding: 1rem; margin-bottom: 0.5rem;
    }

    /* ── Select boxes ── */
    .stSelectbox [data-baseweb="select"] {
        border: 1px solid #e2e8f0; border-radius: 6px;
    }

    /* ── Inputs ── */
    input[data-baseweb="input"] {
        border: 1px solid #e2e8f0; border-radius: 6px;
    }

    /* ── Chart captions ── */
    .stCaption { color: #94a3b8; font-size: 0.7rem; }

    /* ── Dividers ── */
    hr { border-color: #f1f5f9; margin: 1rem 0; }
    </style>
    """)
