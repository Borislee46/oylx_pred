"""Revenue & deferred revenue aggregation helpers."""

import pandas as pd


def total_cash_income(revenue: pd.DataFrame) -> float:
    return pd.to_numeric(revenue["现金收入"], errors="coerce").sum()


def cash_income_by_project(revenue: pd.DataFrame) -> pd.DataFrame:
    return (
        revenue
        .assign(_amt=pd.to_numeric(revenue["现金收入"], errors="coerce"))
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


def deferred_by_teacher(deferred: pd.DataFrame, class_master: pd.DataFrame) -> pd.DataFrame:
    """Join deferred revenue → class master on 班级编号 to get teacher capacity."""
    master = class_master[["班级编码", "主带课教师"]].dropna(subset=["主带课教师"])
    merged = deferred.merge(master, left_on="班级编号", right_on="班级编码", how="left")
    merged["_amt"] = pd.to_numeric(merged["结转收入(含税)"], errors="coerce")
    return (
        merged
        .groupby("主带课教师", as_index=False)["_amt"]
        .sum()
        .rename(columns={"_amt": "结转收入"})
        .sort_values("结转收入", ascending=False)
    )
