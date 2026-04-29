"""Revenue & deferred revenue aggregation helpers."""

import pandas as pd


def total_cash_income(revenue: pd.DataFrame) -> float:
    return pd.to_numeric(revenue["现金收入"], errors="coerce").sum()


def cash_income_by_project(revenue: pd.DataFrame) -> pd.DataFrame:
    return (
        revenue.assign(_amt=pd.to_numeric(revenue["现金收入"], errors="coerce"))
        .groupby("产品品类", as_index=False)["_amt"]
        .sum()
        .rename(columns={"_amt": "现金收入"})
        .sort_values("现金收入", ascending=False)
    )


def cash_income_by_quarter(revenue: pd.DataFrame) -> pd.DataFrame:
    df = revenue.assign(_amt=pd.to_numeric(revenue["现金收入"], errors="coerce"))
    return (
        df.groupby(["季度", "业务归属年"], as_index=False)["_amt"]
        .sum()
        .rename(columns={"_amt": "现金收入"})
        .sort_values(["业务归属年", "季度"])
    )


def cash_income_monthly(revenue: pd.DataFrame) -> pd.DataFrame:
    """Monthly cash income trend. Returns DataFrame with '月份' (datetime) and '现金收入'."""
    df = revenue.assign(
        _amt=pd.to_numeric(revenue["现金收入"], errors="coerce"),
        _dt=pd.to_datetime(revenue["业务日期"], errors="coerce"),
    )
    df["月份"] = df["_dt"].dt.to_period("M").dt.to_timestamp()
    return (
        df.groupby("月份", as_index=False)["_amt"]
        .sum()
        .rename(columns={"_amt": "现金收入"})
        .sort_values("月份")
    )


def monthly_deferred_revenue(deferred: pd.DataFrame) -> pd.DataFrame:
    df = deferred.assign(
        _amt=pd.to_numeric(deferred["结转收入(含税)"], errors="coerce"),
        _dt=pd.to_datetime(deferred["月份"].astype(str) + "01", format="%Y%m%d", errors="coerce"),
    )
    df["月份"] = df["_dt"].dt.to_period("M").dt.to_timestamp()
    return (
        df.groupby("月份", as_index=False)["_amt"]
        .sum()
        .rename(columns={"_amt": "结转收入"})
        .sort_values("月份")
    )


def cash_income_by_grade(revenue: pd.DataFrame) -> pd.DataFrame:
    grade_col = "学员年级(自动更新)" if "学员年级(自动更新)" in revenue.columns else "报名时年级"
    if grade_col not in revenue.columns:
        return pd.DataFrame()
    return (
        revenue.assign(_amt=pd.to_numeric(revenue["现金收入"], errors="coerce"))
        .groupby(grade_col, as_index=False)["_amt"]
        .sum()
        .rename(columns={grade_col: "年级", "_amt": "现金收入"})
        .sort_values("现金收入", ascending=False)
    )


def cash_income_by_capacity(revenue: pd.DataFrame, class_master: pd.DataFrame) -> pd.DataFrame:
    """按班容名称统计现金收入。班容信息来自班级维表，通过班级编码关联。"""
    cap_map = class_master[["班级编码", "班容名称"]].dropna(subset=["班容名称"]).drop_duplicates()
    if cap_map.empty:
        return pd.DataFrame()
    merged = revenue.merge(cap_map, on="班级编码", how="inner")
    merged["_amt"] = pd.to_numeric(merged["现金收入"], errors="coerce")
    return (
        merged.groupby("班容名称", as_index=False)["_amt"]
        .sum()
        .rename(columns={"_amt": "现金收入"})
        .sort_values("现金收入", ascending=False)
    )


def cash_income_by_school(revenue: pd.DataFrame) -> pd.DataFrame:
    col = "学校名称" if "学校名称" in revenue.columns else None
    if col is None:
        return pd.DataFrame()
    return (
        revenue.assign(_amt=pd.to_numeric(revenue["现金收入"], errors="coerce"))
        .groupby(col, as_index=False)["_amt"]
        .sum()
        .rename(columns={col: "学校", "_amt": "现金收入"})
        .sort_values("现金收入", ascending=False)
    )


def deferred_by_teacher(deferred: pd.DataFrame, class_master: pd.DataFrame) -> pd.DataFrame:
    """Join deferred revenue → class master on 班级编号 to get teacher capacity."""
    master = class_master[["班级编码", "主带课教师"]].dropna(subset=["主带课教师"])
    merged = deferred.merge(master, left_on="班级编号", right_on="班级编码", how="left")
    merged["_amt"] = pd.to_numeric(merged["结转收入(含税)"], errors="coerce")
    return (
        merged.groupby("主带课教师", as_index=False)["_amt"]
        .sum()
        .rename(columns={"_amt": "结转收入"})
        .sort_values("结转收入", ascending=False)
    )


def cash_income_by_subject(revenue: pd.DataFrame) -> pd.DataFrame:
    if "科目" not in revenue.columns:
        return pd.DataFrame()
    return (
        revenue.assign(_amt=pd.to_numeric(revenue["现金收入"], errors="coerce"))
        .groupby("科目", as_index=False)["_amt"]
        .sum()
        .rename(columns={"_amt": "现金收入"})
        .sort_values("现金收入", ascending=False)
    )


def new_old_student_breakdown(revenue: pd.DataFrame) -> pd.DataFrame:
    col = "集团口径是否新老生"
    if col not in revenue.columns:
        return pd.DataFrame()
    return (
        revenue.assign(_amt=pd.to_numeric(revenue["现金收入"], errors="coerce"))
        .groupby(col, as_index=False)["_amt"]
        .sum()
        .rename(columns={col: "类型", "_amt": "现金收入"})
        .sort_values("现金收入", ascending=False)
    )


def mom_yoy_kpi(revenue: pd.DataFrame) -> dict:
    """Month-over-month and year-over-year cash income comparison."""
    df = revenue.assign(
        _amt=pd.to_numeric(revenue["现金收入"], errors="coerce"),
        _dt=pd.to_datetime(revenue["业务日期"], errors="coerce"),
    )
    df["月份"] = df["_dt"].dt.to_period("M")
    monthly = df.groupby("月份")["_amt"].sum().sort_index()
    if len(monthly) < 2:
        return {"latest_month": "", "latest_amt": 0, "mom_pct": None, "yoy_pct": None}

    latest_month = str(monthly.index[-1])
    latest_amt = monthly.iloc[-1]
    prev_amt = monthly.iloc[-2]
    mom_pct = (latest_amt - prev_amt) / prev_amt * 100 if prev_amt else None

    yoy_month = monthly.index[-1] - 12
    yoy_pct = None
    if yoy_month in monthly.index:
        yoy_amt = monthly[yoy_month]
        yoy_pct = (latest_amt - yoy_amt) / yoy_amt * 100 if yoy_amt else None

    return {
        "latest_month": latest_month,
        "latest_amt": latest_amt,
        "prev_amt": prev_amt,
        "mom_pct": mom_pct,
        "yoy_pct": yoy_pct,
    }
