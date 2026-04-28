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
        df
        .groupby("顾问姓名", as_index=False)
        .agg(签约数=("签约单id", "count"), 总学费=("学费_num", "sum"))
        .sort_values("签约数", ascending=False)
    )
    return ranking


def channel_breakdown(kehu: pd.DataFrame) -> pd.DataFrame:
    if "一级渠道" not in kehu.columns:
        return pd.DataFrame()
    return kehu["一级渠道"].value_counts().reset_index(name="count").rename(columns={"index": "渠道"})


def class_capacity_metrics(class_master: pd.DataFrame) -> dict:
    cm = class_master.copy()
    cm["标准人数_n"] = pd.to_numeric(cm["标准人数"], errors="coerce")
    cm["当前人数_n"] = pd.to_numeric(cm["当前人数"], errors="coerce")
    cm["最大人数_n"] = pd.to_numeric(cm["最大人数"], errors="coerce")
    total = len(cm[cm["班级状态"] == "正常"])
    full = len(cm[(cm["班级状态"] == "正常") & (cm["当前人数_n"] >= cm["标准人数_n"])])
    avg_ratio = (
        (cm.loc[cm["班级状态"] == "正常", "当前人数_n"] /
         cm.loc[cm["班级状态"] == "正常", "标准人数_n"].replace(0, pd.NA))
        .mean()
    )
    return {
        "行课班级数": total,
        "满班班级数": full,
        "满班率": f"{full / total * 100:.0f}%" if total else "0%",
        "平均满班率": f"{avg_ratio * 100:.0f}%" if pd.notna(avg_ratio) else "0%",
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
