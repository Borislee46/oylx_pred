"""HK Dashboard — compact premium styling."""

import streamlit as st


def inject_hk_dashboard_styles() -> None:
    st.html(
        """
    <style>
    /* ── Root ── */
    .stMainBlockContainer { padding: 1.5rem 2rem 2rem 2rem; }
    .stApp { background: #f8fafc; }

    /* ── Title ── */
    h1 {
        font-size: 1.5rem; font-weight: 700; color: #0f172a;
        padding: 0 0 0.5rem 0; margin: 0 0 0.6rem 0;
        border-bottom: 2px solid #e2e8f0;
    }

    /* ── Section headers ── */
    h2 {
        font-size: 1.05rem; font-weight: 600; color: #1e293b;
        margin: 1.1rem 0 0.5rem 0; padding: 0;
    }
    h3 {
        font-size: 0.85rem; font-weight: 600; color: #475569;
        margin: 0.5rem 0 0.35rem 0;
    }

    /* ── KPI row ── */
    .hk-kpi-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 0 0 0.6rem 0; }
    .hk-kpi-card {
        flex: 1 1 160px; min-width: 140px;
        background: #fff; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 0.7rem 0.9rem;
    }
    .hk-kpi-card .kpi-value {
        font-size: 1.4rem; font-weight: 700; color: #0f172a; line-height: 1.2;
    }
    .hk-kpi-card .kpi-label {
        font-size: 0.72rem; font-weight: 500; color: #64748b;
        margin-top: 0.2rem;
    }
    .hk-kpi-card .kpi-sub {
        font-size: 0.72rem; color: #64748b; margin-top: 0.15rem;
    }
    .hk-kpi-card .kpi-formula {
        font-size: 0.65rem; color: #94a3b8; margin-top: 0.2rem;
        font-family: "SF Mono", "Consolas", monospace;
    }

    /* ── KPI accent bars ── */
    .hk-kpi-card.t-blue  { border-left: 3px solid #2563eb; }
    .hk-kpi-card.t-green { border-left: 3px solid #0d9488; }
    .hk-kpi-card.t-amber { border-left: 3px solid #d97706; }
    .hk-kpi-card.t-slate { border-left: 3px solid #64748b; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 0.5rem 0.8rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem; font-weight: 500; color: #64748b;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.15rem; font-weight: 700; color: #0f172a;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem; font-weight: 500; padding: 0.35rem 1rem;
        color: #64748b; border-radius: 0; margin-right: 0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb; font-weight: 600;
        border-bottom: 2px solid #2563eb;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent; }

    /* ── DataFrames ── */
    .stDataFrame {
        font-size: 0.78rem; border: 1px solid #e2e8f0; border-radius: 8px;
    }
    .stDataFrame thead th {
        background: #f8fafc; color: #475569; font-weight: 600; font-size: 0.7rem;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0; border-radius: 8px; margin: 0.35rem 0;
        background: #fff;
    }
    [data-testid="stExpander"] p { font-size: 0.82rem; }

    /* ── Containers ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff; border-radius: 8px; border: 1px solid #e2e8f0;
        padding: 0.85rem; margin-bottom: 0.5rem;
    }

    /* ── Select & Input ── */
    .stSelectbox [data-baseweb="select"] { border: 1px solid #e2e8f0; border-radius: 6px; }
    input[data-baseweb="input"] { border: 1px solid #e2e8f0; border-radius: 6px; }

    /* ── Captions ── */
    .stCaption { color: #94a3b8; font-size: 0.7rem; }

    /* ── Formula footnote (hidden by default, toggle in dev) ── */
    .hk-note {
        display: none;
    }

    hr { border-color: #e2e8f0; margin: 0.6rem 0; }
    </style>
    """
    )
