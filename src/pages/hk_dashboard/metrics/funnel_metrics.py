"""Conversion funnel & consultant performance calculations."""

import pandas as pd


def calculate_funnel(kehu: pd.DataFrame, tmk: pd.DataFrame, qianyue: pd.DataFrame) -> dict:
    """Compute 4-stage conversion funnel counts."""
    total = len(kehu)
    # Resources that have outbound calls in TMK
    called = tmk["外呼次数"].notna().sum() if "外呼次数" in tmk.columns else 0
    # Resources that generated a work order (工单状态 is not null)
    has_workorder = tmk["工单状态"].notna().sum() if "工单状态" in tmk.columns else 0
    # Resources that resulted in a signed contract
    signed = qianyue["签约单id"].notna().sum() if "签约单id" in qianyue.columns else 0

    return {
        "总资源数": total,
        "已外呼": int(called),
        "有工单": int(has_workorder),
        "已签约": int(signed),
        "外呼率": f"{called / total * 100:.1f}%" if total else "0%",
        "外呼→工单率": f"{has_workorder / called * 100:.1f}%" if called else "0%",
        "工单→签约率": f"{signed / has_workorder * 100:.1f}%" if has_workorder else "0%",
    }


def consultant_ranking(qianyue: pd.DataFrame) -> pd.DataFrame:
    df = qianyue.copy()
    df["学费_num"] = pd.to_numeric(df["学费"], errors="coerce")
    ranking = (
        df.groupby("顾问姓名", as_index=False)
        .agg(签约数=("签约单id", "count"), 总学费=("学费_num", "sum"))
        .sort_values("签约数", ascending=False)
    )
    return ranking


def channel_breakdown(kehu: pd.DataFrame) -> pd.DataFrame:
    if "一级渠道" not in kehu.columns:
        return pd.DataFrame()
    return (
        kehu["一级渠道"].value_counts().reset_index(name="count").rename(columns={"index": "渠道"})
    )


def class_capacity_metrics(class_master: pd.DataFrame) -> dict:
    """班级容量指标。仅统计当前人数 > 0 的正常班级。"""
    cm = class_master.copy()
    cm["标准_n"] = pd.to_numeric(cm["标准人数"], errors="coerce")
    cm["当前_n"] = pd.to_numeric(cm["当前人数"], errors="coerce")
    active = cm[(cm["班级状态"] == "正常") & (cm["当前_n"] > 0)]
    total = len(active)
    full = len(active[active["当前_n"] >= active["标准_n"]])
    avg_ratio = (active["当前_n"] / active["标准_n"].replace(0, pd.NA)).mean()
    avg_current = active["当前_n"].mean()
    return {
        "行课班级数": total,
        "满班班级数": full,
        "满班率": f"{full / total * 100:.0f}%" if total else "-",
        "平均满班率": f"{avg_ratio * 100:.0f}%" if pd.notna(avg_ratio) else "-",
        "平均班容": f"{avg_current:.0f} 人" if pd.notna(avg_current) else "-",
    }


def tmk_processing_stats(tmk: pd.DataFrame) -> dict:
    """TMK processing TAT stats."""
    t = tmk.copy()
    t["分配_dt"] = pd.to_datetime(t["分配时间"], errors="coerce")
    t["处理_dt"] = pd.to_datetime(t["顾问处理时间"], errors="coerce")
    delta = (t["处理_dt"] - t["分配_dt"]).dropna()
    called = t["外呼次数"].notna().sum()
    processed = (t["资源状态"] == "已处理").sum()
    pending = (t["资源状态"] == "待处理").sum()
    within_1d = int((delta <= pd.Timedelta("1 day")).sum()) if len(delta) else 0
    return {
        "已外呼": int(called),
        "已处理": int(processed),
        "待处理": int(pending),
        "1天内处理": within_1d,
        "平均处理时延(天)": f"{delta.dt.total_seconds().mean() / 86400:.1f}" if len(delta) else "-",
    }


def work_order_follow_up_stats(tmk: pd.DataFrame, qianyue: pd.DataFrame) -> dict:
    """Work order follow-up stats: total, normal, deleted, dead-marked."""
    t = tmk.copy()
    q = qianyue.copy()

    col_wid = "工单id" if "工单id" in t.columns else "工单ID"
    wo_total = t[col_wid].notna().sum() if col_wid in t.columns else 0
    wo_normal = (t["工单状态"] == "正常").sum() if "工单状态" in t.columns else 0
    wo_deleted = (t["工单状态"] == "删除").sum() if "工单状态" in t.columns else 0
    wo_dead = t["死单类型"].notna().sum() if "死单类型" in t.columns else 0

    signed_with_consultant = 0
    if "工单跟进顾问" in q.columns:
        signed_with_consultant = q[q["签约单id"].notna() & q["工单跟进顾问"].notna()].shape[0]

    return {
        "工单总数": int(wo_total),
        "正常": int(wo_normal),
        "已删除": int(wo_deleted),
        "死单标记": int(wo_dead),
        "有顾问跟进签约": int(signed_with_consultant),
    }


def consultant_detail(kehu: pd.DataFrame, qianyue: pd.DataFrame) -> pd.DataFrame:
    """Per-consultant detail: signings, students, total revenue, yesterday income."""
    q = qianyue.copy()
    if "顾问姓名" not in q.columns:
        return pd.DataFrame()

    q["学费_n"] = pd.to_numeric(q["学费"], errors="coerce")
    detail = (
        q.groupby("顾问姓名", as_index=False)
        .agg(签约数=("签约单id", "count"), 总学费=("学费_n", "sum"))
        .sort_values("签约数", ascending=False)
    )

    # Student headcount
    if "资源姓名" in q.columns:
        student_count = q.groupby("顾问姓名")["资源姓名"].nunique().reset_index(name="人头数")
        detail = detail.merge(student_count, on="顾问姓名", how="left")

    # Signed rate: signed / total resources assigned
    if "资源分配顾问" in kehu.columns and "资源id" in q.columns and "资源id" in kehu.columns:
        assigned = kehu["资源分配顾问"].value_counts().reset_index(name="分配资源数")
        assigned.columns = ["顾问姓名", "分配资源数"]
        signed_res = (
            q[q["签约单id"].notna()]
            .groupby("顾问姓名")["资源id"]
            .nunique()
            .reset_index(name="签约使用资源数")
        )
        detail = detail.merge(assigned, on="顾问姓名", how="left")
        detail = detail.merge(signed_res, on="顾问姓名", how="left")
        detail["资源签约率"] = (
            detail["签约使用资源数"] / detail["分配资源数"].replace(0, pd.NA) * 100
        ).round(1)

    return detail


def monthly_signing_trend(qianyue: pd.DataFrame) -> pd.DataFrame:
    """Monthly signing count trend from 签约时间."""
    q = qianyue.copy()
    if "签约时间" not in q.columns:
        return pd.DataFrame()
    q["签约_dt"] = pd.to_datetime(q["签约时间"], errors="coerce")
    q["月份"] = q["签约_dt"].dt.to_period("M").dt.to_timestamp()
    return (
        q.dropna(subset=["月份"])
        .groupby("月份", as_index=False)
        .agg(
            签约数=("签约单id", "count"),
            签约金额=("学费", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        )
        .sort_values("月份")
    )


def signing_by_product(qianyue: pd.DataFrame) -> pd.DataFrame:
    """Signing count by 课程产品名称."""
    col = "课程产品名称"
    if col not in qianyue.columns:
        return pd.DataFrame()
    q = qianyue.copy()
    q["学费_n"] = pd.to_numeric(q["学费"], errors="coerce")
    return (
        q.groupby(col, as_index=False)
        .agg(签约数=("签约单id", "count"), 签约金额=("学费_n", "sum"))
        .sort_values("签约数", ascending=False)
    )


def channel_product_cross(qianyue: pd.DataFrame) -> pd.DataFrame:
    """交叉: 一级获取渠道 × 课程产品名称."""
    if "一级获取渠道" not in qianyue.columns or "课程产品名称" not in qianyue.columns:
        return pd.DataFrame()
    q = qianyue.dropna(subset=["一级获取渠道", "课程产品名称"])
    return (
        q.groupby(["一级获取渠道", "课程产品名称"], as_index=False)
        .agg(签约数=("签约单id", "count"))
        .sort_values("签约数", ascending=False)
    )
