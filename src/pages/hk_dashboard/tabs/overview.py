"""Tab 1: 综合概览 — 数据健康度 + 核心指标."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import monthly_trend_line, simple_bar
from src.pages.hk_dashboard.components.kpi_cards import render_kpi_row
from src.pages.hk_dashboard.metrics.funnel_metrics import consultant_ranking
from src.pages.hk_dashboard.metrics.renewal_metrics import calculate_renewal_rate
from src.pages.hk_dashboard.metrics.revenue_metrics import cash_income_monthly


def _data_inventory(data: dict) -> None:
    """Compact table inventory — what we have."""
    tables = [
        ("客服资源", data["kehu_ziyuan"], "资源id", None),
        ("TMK处理", data["tmk"], "资源ID", "93%工单失效"),
        ("签约列表", data["qianyue"], "签约单id", "91%资源匹配"),
        ("班级维表", data["class_master"], "班级编码", "241正常/86空班"),
        ("花名册", data["roster"], "学员编号", "367学员"),
        ("收入人次", data["revenue"], None, "57%班级匹配"),
        ("结转收入", data["deferred_revenue"], None, "42%班级匹配"),
    ]

    cols = st.columns(7)
    for i, (name, df, _pk, note) in enumerate(tables):
        with cols[i]:
            empty_flag = " 🔴" if df.empty else ""
            rows = f"{len(df):,}" if not df.empty else "无"
            st.metric(
                label=f"{name}{empty_flag}",
                value=rows,
                delta=note,
                delta_color=(
                    "off" if note and "失效" not in note and "空班" not in note else "inverse"
                ),
            )


def _data_quality(data: dict) -> None:
    """Data quality alerts — what's wrong."""
    revenue = data["revenue"]
    tmk = data["tmk"]
    class_master = data["class_master"]
    roster = data["roster"]
    qianyue = data["qianyue"]

    issues = []

    # 1. Revenue orphans — classes not in master table
    cm_codes = set(class_master["班级编码"].dropna())
    rev_codes = set(revenue["班级编码"].dropna())
    orphan_codes = rev_codes - cm_codes
    if orphan_codes:
        orphan_amt = pd.to_numeric(
            revenue[revenue["班级编码"].isin(orphan_codes)]["现金收入"], errors="coerce"
        ).sum()
        rev_total = pd.to_numeric(revenue["现金收入"], errors="coerce").sum()
        pct = abs(orphan_amt) / abs(rev_total) * 100 if rev_total else 0
        issues.append(
            {
                "level": "warn",
                "text": f"收入孤儿班级: {len(orphan_codes)} 个班级不在维表, "
                f"涉及 ¥{abs(orphan_amt) / 1e4:.0f} 万 ({pct:.0f}% 收入)",
            }
        )

    # 2. Refund rate
    amt = pd.to_numeric(revenue["现金收入"], errors="coerce")
    gross = amt[amt > 0].sum()
    refund = abs(amt[amt < 0].sum())
    refund_rate = refund / gross * 100 if gross else 0
    if refund_rate > 15:
        issues.append(
            {
                "level": "warn",
                "text": f"退费率 {refund_rate:.1f}%: "
                f"流水 ¥{gross / 1e4:.0f} 万, 退费 ¥{refund / 1e4:.0f} 万",
            }
        )

    # 3. Work order pipeline failure
    if "工单状态" in tmk.columns:
        wo_total = tmk["工单状态"].notna().sum()
        wo_deleted = (tmk["工单状态"] == "删除").sum()
        if wo_total > 0 and wo_deleted / wo_total > 0.7:
            issues.append(
                {
                    "level": "error",
                    "text": f"工单管道失效: {wo_deleted}/{wo_total} 已删除 "
                    f"({wo_deleted / wo_total * 100:.0f}%), 仅 {wo_total - wo_deleted} 正常",
                }
            )

    # 4. Zero-student active classes
    cm = class_master.copy()
    cm["当前_n"] = pd.to_numeric(cm["当前人数"], errors="coerce")
    empty_active = len(cm[(cm["班级状态"] == "正常") & (cm["当前_n"] == 0)])
    if empty_active > 10:
        issues.append(
            {
                "level": "warn",
                "text": f"空壳班级: {empty_active} 个班级状态为「正常」但当前人数 = 0",
            }
        )

    # 5. TMK-Signing zero overlap
    qianyue_ids = set(qianyue["资源id"].dropna()) if "资源id" in qianyue.columns else set()
    tmk_ids = set(tmk["资源id"].dropna()) if "资源id" in tmk.columns else set()
    if tmk_ids and qianyue_ids:
        overlap = len(tmk_ids & qianyue_ids)
        if overlap == 0:
            issues.append(
                {
                    "level": "info",
                    "text": "TMK ↔ 签约资源id交集为 0: 两条管道完全独立, 无法计算端到端转化率",
                }
            )

    # 6. Teacher coverage
    tm = class_master[["主带课教师", "教师"]].copy()
    tm["_t"] = tm["主带课教师"]
    mask = tm["_t"].isna()
    tm.loc[mask, "_t"] = tm.loc[mask, "教师"].apply(
        lambda x: str(x).split(",")[0].split("(")[0].strip() if pd.notna(x) else None
    )
    missing_teacher = tm["_t"].isna().sum()
    if missing_teacher > 5:
        issues.append(
            {
                "level": "warn",
                "text": f"教师缺失: {missing_teacher} 个班级无教师归属, 影响续费率和奖金的准确性",
            }
        )

    # 7. Student orphans
    roster_ids = set(roster["学员编号"].dropna())
    rev_student_ids = set(revenue["学员编号"].dropna())
    orphan_students = rev_student_ids - roster_ids
    if orphan_students:
        orphan_s_amt = pd.to_numeric(
            revenue[revenue["学员编号"].isin(orphan_students)]["现金收入"],
            errors="coerce",
        ).sum()
        issues.append(
            {
                "level": "warn",
                "text": f"收入孤儿学员: {len(orphan_students)} 个学员不在花名册, "
                f"涉及 ¥{abs(orphan_s_amt) / 1e4:.0f} 万",
            }
        )

    if not issues:
        st.success("数据质量良好，无异常")
        return

    # Render issues compactly
    errors = [i for i in issues if i["level"] == "error"]
    warns = [i for i in issues if i["level"] == "warn"]
    infos = [i for i in issues if i["level"] == "info"]

    colors = {"error": "#dc2626", "warn": "#d97706", "info": "#64748b"}
    bg_colors = {"error": "#fef2f2", "warn": "#fffbeb", "info": "#f8fafc"}
    icons = {"error": "🔴", "warn": "⚠️", "info": "ℹ️"}

    for level, items in [("error", errors), ("warn", warns), ("info", infos)]:
        for item in items:
            st.html(
                f'<div style="background:{bg_colors[level]};border-left:3px solid {colors[level]};'
                f"padding:0.35rem 0.6rem;margin:0.15rem 0;border-radius:2px;font-size:0.78rem;"
                f'color:#334155;">{icons[level]} {item["text"]}</div>'
            )


def _key_metrics(data: dict) -> None:
    """Core business KPIs in one compact row."""
    revenue = data["revenue"]
    qianyue = data["qianyue"]
    roster = data["roster"]
    class_master = data["class_master"]

    # Revenue
    amt = pd.to_numeric(revenue["现金收入"], errors="coerce")
    net = amt.sum()
    gross = amt[amt > 0].sum()
    refund = abs(amt[amt < 0].sum())

    # Signings
    total_signed = qianyue["签约单id"].notna().sum()

    # Active students
    r = roster.copy()
    r["进班_dt"] = pd.to_datetime(r["进班日期"], errors="coerce")
    r["离班_dt"] = pd.to_datetime(r["离班日期"], errors="coerce")
    now = pd.Timestamp.now()
    active_students = r[
        (r["进班_dt"] <= now)
        & (r["离班_dt"].isna() | (r["离班_dt"] >= now))
        & (r["有效状态"] == "有效")
    ]["学员编号"].nunique()

    # Renewal rate (latest available month)
    renewal_rate_str = "-"
    try:
        from src.pages.hk_dashboard.tabs.renewal import _available_months

        months = _available_months(roster)
        if months:
            latest = months[-1]
            renewal_df = calculate_renewal_rate(roster, class_master, month=latest)
            if not renewal_df.empty and renewal_df["当月学员数"].sum() > 0:
                rate = renewal_df["次月在班数"].sum() / renewal_df["当月学员数"].sum()
                renewal_rate_str = f"{rate:.0%}"
    except Exception:
        pass

    render_kpi_row(
        [
            {
                "value": f"{net / 1e4:.0f} 万",
                "label": "现金收入 (净额)",
                "accent": "green",
                "sub": f"流水 ¥{gross / 1e4:.0f} 万  |  退费 ¥{refund / 1e4:.0f} 万",
                "formula": "SUM(收入人次.现金收入)",
            },
            {
                "value": str(total_signed),
                "label": "累计签约",
                "accent": "blue",
                "formula": "COUNT(签约列表.签约单id)",
            },
            {
                "value": str(active_students),
                "label": "当前在读",
                "accent": "blue",
                "formula": "花名册 | 已进班 未离班 有效",
            },
            {
                "value": renewal_rate_str,
                "label": "上月续费率",
                "accent": (
                    "amber"
                    if renewal_rate_str.replace("%", "").isdigit()
                    and float(renewal_rate_str.replace("%", "")) < 75
                    else "green"
                ),
                "formula": "次月在班数 / 当月学员数",
            },
        ]
    )


def render(data: dict[str, pd.DataFrame]) -> None:
    revenue = data["revenue"]
    kehu = data["kehu_ziyuan"]
    qianyue = data["qianyue"]

    # ── Section 1: Data Inventory ──
    st.html("<h2>数据资产</h2>")
    _data_inventory(data)

    # ── Section 2: Data Quality ──
    with st.expander("数据质量检查", expanded=True):
        _data_quality(data)

    # ── Section 3: Key Metrics ──
    st.html("<h2>核心指标</h2>")
    _key_metrics(data)

    # ── Section 4: Revenue Trend ──
    c1, c2 = st.columns([3, 2])
    with c1:
        with st.container(border=True):
            st.html("<h3>月度现金收入</h3>")
            monthly = cash_income_monthly(revenue)
            if not monthly.empty:
                monthly_trend_line(monthly, "月份", "现金收入", currency=True, height=200)
    with c2:
        with st.container(border=True):
            st.html("<h3>顾问签约 Top 10</h3>")
            ranking = consultant_ranking(qianyue)
            if not ranking.empty:
                simple_bar(ranking.head(10), "顾问姓名", "签约数", horizontal=True, height=200)

    # ── Details in expander ──
    with st.expander("更多明细"):
        c1, c2 = st.columns(2)
        with c1:
            from src.pages.hk_dashboard.metrics.funnel_metrics import channel_breakdown

            st.html("<h3>资源渠道</h3>")
            ch = channel_breakdown(kehu)
            if not ch.empty:
                from src.pages.hk_dashboard.charts import donut_chart

                donut_chart(ch, "渠道", "count", max_categories=5, height=200)
        with c2:
            from src.pages.hk_dashboard.metrics.revenue_metrics import cash_income_by_project

            st.html("<h3>收入 — 产品品类</h3>")
            by_proj = cash_income_by_project(revenue)
            if not by_proj.empty:
                from src.pages.hk_dashboard.charts import simple_bar as sb

                sb(by_proj, "产品品类", "现金收入", fmt=",.0f", height=200)
