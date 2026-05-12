"""Renewal rate calculation (cohort method) and bonus computation.

Validated against data/hk/旺角2月教学端绩效核对.xlsx (Feb → Mar 2026).
"""

import pandas as pd

from src.pages.hk_dashboard.config import lookup_bonus_rate


def _month_range(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse '2026-02' into (start, end) timestamps for that month."""
    y, m = map(int, month.split("-"))
    start = pd.Timestamp(y, m, 1)
    end = start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return start, end


def calculate_renewal_rate(
    roster: pd.DataFrame,
    class_master: pd.DataFrame,
    month: str = "2026-02",
) -> pd.DataFrame:
    """Cohort renewal rate per teacher for a given month.

    month: 'YYYY-MM' format (e.g. '2026-02').
    Returns DataFrame with columns: 教师, 当月学员数, 次月在班数, 续费率
    """
    # ── Prepare roster date types ──
    r = roster.copy()
    r["进班_dt"] = pd.to_datetime(r["进班日期"], errors="coerce")
    r["离班_dt"] = pd.to_datetime(r["离班日期"], errors="coerce")

    # Parse month
    start, end = _month_range(month)
    y, m = map(int, month.split("-"))
    next_m = (m % 12) + 1
    next_y = y + 1 if m == 12 else y
    next_start, next_end = _month_range(f"{next_y}-{next_m:02d}")

    # ── Active students in target month ──
    # Student is active if 进班_dt <= month_end AND (离班_dt is NaT OR 离班_dt >= month_start)
    active_mask = (r["进班_dt"] <= end) & (r["离班_dt"].isna() | (r["离班_dt"] >= start))
    active = r[active_mask][["学员编号", "班级编码"]].drop_duplicates()

    # ── Active students in next month ──
    retained_mask = (r["进班_dt"] <= next_end) & (
        r["离班_dt"].isna() | (r["离班_dt"] >= next_start)
    )
    retained_students = r[retained_mask]["学员编号"].drop_duplicates().tolist()

    # ── Attach teacher ──
    # 主带课教师 (108/258 non-null) is closest to Excel's manually-assigned primary teacher.
    # Fallback: first name from 教师 column for classes without a designated primary.
    tm = class_master[["班级编码", "主带课教师", "教师"]].copy()
    tm["_teacher"] = tm["主带课教师"]
    mask_nan = tm["_teacher"].isna()
    tm.loc[mask_nan, "_teacher"] = tm.loc[mask_nan, "教师"].apply(
        lambda x: str(x).split(",")[0].split("(")[0].strip() if pd.notna(x) else None
    )
    teacher_map = tm[["班级编码", "_teacher"]].dropna(subset=["_teacher"]).drop_duplicates()
    active = active.merge(teacher_map, on="班级编码", how="inner")

    # ── Per-teacher calc ──
    results = []
    for teacher, grp in active.groupby("_teacher"):
        if pd.isna(teacher):
            continue
        f_total = grp["学员编号"].nunique()
        f_retained = grp[grp["学员编号"].isin(retained_students)]["学员编号"].nunique()
        rate = f_retained / f_total if f_total > 0 else 0.0
        results.append(
            {"教师": teacher, "当月学员数": f_total, "次月在班数": f_retained, "续费率": rate}
        )

    if not results:
        return pd.DataFrame(columns=["教师", "当月学员数", "次月在班数", "续费率"])
    return pd.DataFrame(results).sort_values("续费率", ascending=False)


def calculate_bonus(
    renewal_df: pd.DataFrame,
    class_master: pd.DataFrame,
    month: str = "2026-02",
) -> pd.DataFrame:
    """Calculate teaching bonus based on renewal rate × tiered rate table.

    renewal_df: output from calculate_renewal_rate()
    class_master: for extracting teacher total teaching hours in the month
    """
    # ── Teaching hours: same 主带课教师 → first-teacher fallback as renewal rate ──
    cm = class_master[["主带课教师", "教师", "实际上课时长（去除赠课）", "课次"]].copy()
    cm["_t"] = cm["主带课教师"]
    mask_nan = cm["_t"].isna()
    cm.loc[mask_nan, "_t"] = cm.loc[mask_nan, "教师"].apply(
        lambda x: str(x).split(",")[0].split("(")[0].strip() if pd.notna(x) else None
    )
    teacher_hours = {}
    for _, row in cm.iterrows():
        t = row["_t"]
        if pd.isna(t):
            continue
        h = pd.to_numeric(row.get("实际上课时长（去除赠课）", 0), errors="coerce")
        if pd.isna(h) or h == 0:
            h = pd.to_numeric(row.get("课次", 0), errors="coerce")
        teacher_hours[t] = teacher_hours.get(t, 0) + (h if pd.notna(h) else 0)

    if renewal_df.empty:
        return pd.DataFrame(
            columns=[
                "教师",
                "当月学员数",
                "次月在班数",
                "续费率",
                "单价(HKD/课时)",
                "当月课时",
                "应发奖金(HKD)",
            ]
        )

    # ── Apply tiers ──
    result = renewal_df.copy()
    result["单价(HKD/课时)"] = result.apply(
        lambda r: lookup_bonus_rate(r["续费率"], r["当月学员数"]), axis=1
    )
    result["当月课时"] = result["教师"].map(teacher_hours).fillna(0).astype(int)
    result["应发奖金(HKD)"] = result["单价(HKD/课时)"] * result["当月课时"]

    return result.sort_values("应发奖金(HKD)", ascending=False)
