"""HK Dashboard — compact premium styling."""

import streamlit as st


def inject_hk_dashboard_styles() -> None:
    st.html("""
    <style>
    /* ── Root ── */
    .stMainBlockContainer { padding: 0.4rem 1.25rem 1.5rem 1.25rem; }
    .stApp { background: #f8fafc; }

    /* ── Title ── */
    h1 {
        font-size: 1.25rem; font-weight: 600; color: #0f172a;
        padding: 0.15rem 0 0.35rem 0; margin: 0 0 0.5rem 0;
        border-bottom: 1px solid #e2e8f0;
    }

    /* ── Section headers ── */
    h2 {
        font-size: 0.85rem; font-weight: 600; color: #334155;
        margin: 0.8rem 0 0.35rem 0; padding: 0;
        text-transform: uppercase; letter-spacing: 0.06em;
    }
    h3 {
        font-size: 0.75rem; font-weight: 600; color: #64748b;
        margin: 0.4rem 0 0.25rem 0; text-transform: uppercase; letter-spacing: 0.05em;
    }

    /* ── KPI row ── */
    .hk-kpi-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 0 0 0.35rem 0; }
    .hk-kpi-card {
        flex: 1 1 160px; min-width: 130px;
        background: #fff; border: 1px solid #f1f5f9;
        border-radius: 6px; padding: 0.55rem 0.8rem;
    }
    .hk-kpi-card .kpi-value {
        font-size: 1.25rem; font-weight: 700; color: #0f172a; line-height: 1.15;
    }
    .hk-kpi-card .kpi-label {
        font-size: 0.65rem; font-weight: 500; color: #94a3b8;
        letter-spacing: 0.05em; margin-top: 0.15rem; text-transform: uppercase;
    }
    .hk-kpi-card .kpi-sub {
        font-size: 0.66rem; color: #64748b; margin-top: 0.12rem;
    }
    .hk-kpi-card .kpi-formula {
        font-size: 0.6rem; color: #cbd5e1; margin-top: 0.15rem;
        font-family: "SF Mono", "Consolas", monospace;
    }

    /* ── KPI accent bars ── */
    .hk-kpi-card.t-blue  { border-left: 2px solid #2563eb; }
    .hk-kpi-card.t-green { border-left: 2px solid #0d9488; }
    .hk-kpi-card.t-amber { border-left: 2px solid #d97706; }
    .hk-kpi-card.t-slate { border-left: 2px solid #64748b; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #f1f5f9; border-radius: 6px;
        padding: 0.4rem 0.7rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.65rem; font-weight: 500; color: #94a3b8; letter-spacing: 0.04em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.05rem; font-weight: 700; color: #0f172a;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.8rem; font-weight: 500; padding: 0.2rem 0.8rem;
        color: #64748b; border-radius: 0; margin-right: 0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb; font-weight: 600;
        border-bottom: 2px solid #2563eb;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent; }

    /* ── DataFrames ── */
    .stDataFrame {
        font-size: 0.74rem; border: 1px solid #f1f5f9; border-radius: 6px;
    }
    .stDataFrame thead th {
        background: #f8fafc; color: #64748b; font-weight: 600; font-size: 0.68rem;
        text-transform: uppercase; letter-spacing: 0.03em;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        border: 1px solid #f1f5f9; border-radius: 6px; margin: 0.25rem 0;
        background: #fff;
    }
    [data-testid="stExpander"] p { font-size: 0.8rem; }

    /* ── Containers ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff; border-radius: 6px; border: 1px solid #f1f5f9;
        padding: 0.75rem; margin-bottom: 0.35rem;
    }

    /* ── Select & Input ── */
    .stSelectbox [data-baseweb="select"] { border: 1px solid #e2e8f0; border-radius: 4px; }
    input[data-baseweb="input"] { border: 1px solid #e2e8f0; border-radius: 4px; }

    /* ── Captions ── */
    .stCaption { color: #94a3b8; font-size: 0.66rem; }

    /* ── Formula footnote ── */
    .hk-note {
        font-size: 0.62rem; color: #cbd5e1; margin-top: 0.1rem;
        font-family: "SF Mono", "Consolas", monospace;
    }

    hr { border-color: #f1f5f9; margin: 0.5rem 0; }
    </style>
    """)
