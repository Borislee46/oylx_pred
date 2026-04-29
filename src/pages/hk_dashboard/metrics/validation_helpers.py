"""Validation helpers — cross-reference checkers and comparison tools."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

TOLERANCE = 0.01  # 1% default


def compare_series(
    sys_series: pd.Series,
    excel_series: pd.Series,
    label: str = "",
    tolerance: float = TOLERANCE,
) -> pd.DataFrame:
    """Compare two numeric series, returning a DataFrame with diff columns."""
    df = pd.DataFrame(
        {
            "index": sys_series.index.astype(str),
            f"系统{label}": sys_series.values,
            f"Excel{label}": excel_series.values,
        }
    )
    denom = df[f"Excel{label}"].replace(0, pd.NA)
    df["差异%"] = ((df[f"系统{label}"] - df[f"Excel{label}"]) / denom * 100).round(1)
    df["差异"] = df[f"系统{label}"] - df[f"Excel{label}"]
    return df


def highlight_validation_table(df: pd.DataFrame, diff_col: str = "差异%") -> pd.DataFrame:
    """Apply conditional formatting: green for pass, red for fail."""

    def _color(val):
        if pd.isna(val):
            return ""
        if abs(val) < 1.0:
            return "color: #0d9488"
        return "color: #dc2626; font-weight: 600"

    return df.style.map(_color, subset=[diff_col])


def render_cross_validation_expander(
    title: str,
    sys_df: pd.DataFrame,
    excel_df: pd.DataFrame | None,
    sys_label: str,
    excel_label: str,
    key: str = "",
) -> None:
    """Render a cross-validation expander comparing system vs Excel data.

    If excel_df is None, renders a file uploader for the user to upload Excel.
    """
    with st.expander(title):
        if excel_df is None:
            uploaded = st.file_uploader(
                "上传 Excel 对照文件",
                type=["xlsx", "xls", "csv"],
                key=f"upload_{key}",
            )
            if uploaded is None:
                st.caption("等待上传 Excel 对照数据...")
                return
            try:
                if uploaded.name.endswith(".csv"):
                    excel_df = pd.read_csv(uploaded, encoding="utf-8-sig")
                else:
                    excel_df = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"读取文件失败: {e}")
                return

        st.caption(f"{sys_label} vs {excel_label}")

        # System side
        c1, c2 = st.columns(2)
        with c1:
            st.html("<h3>系统计算</h3>")
            st.dataframe(sys_df, width="stretch", hide_index=True)
        with c2:
            st.html("<h3>对照数据 (Excel)</h3>")
            st.dataframe(excel_df, width="stretch", hide_index=True)

        # Diff summary
        if not sys_df.empty and not excel_df.empty:
            draw_diff_summary(sys_df, excel_df, key)


def draw_diff_summary(sys_df: pd.DataFrame, excel_df: pd.DataFrame, key: str = "") -> None:
    """Print diff summary line."""
    st.divider()
    try:
        sys_sum = sys_df.select_dtypes(include="number").sum().sum()
        excel_sum = excel_df.select_dtypes(include="number").sum().sum()
        if excel_sum != 0:
            diff_pct = (sys_sum - excel_sum) / abs(excel_sum) * 100
            color = "#dc2626" if abs(diff_pct) > 2 else "#0d9488"
            st.html(
                f'<p style="font-size:0.85rem">'
                f'汇总差异: <span style="color:{color};font-weight:700">{diff_pct:+.1f}%</span>'
                f" (系统 {sys_sum:,.0f} vs Excel {excel_sum:,.0f})"
                f"</p>"
            )
    except Exception:
        pass


def build_cross_ref_matrix(
    all_data: dict[str, pd.DataFrame],
    key_col: str,
    datasets: list[str],
) -> pd.DataFrame:
    """Build cross-reference matrix: how many `key_col` values intersect across datasets.

    Returns a matrix with columns like: [key_col, source1_count, source2_count, intersection_size, overlap%]
    """
    sets = {}
    for name in datasets:
        df = all_data.get(name, pd.DataFrame())
        if df.empty or key_col not in df.columns:
            sets[name] = set()
        else:
            sets[name] = set(df[key_col].dropna())

    names = datasets
    rows = []
    for i, a in enumerate(names):
        row = {"来源": a}
        row["唯一值数"] = len(sets[a])
        for j, b in enumerate(names):
            if i == j:
                row[b] = f"{len(sets[a])} (100%)"
            else:
                inter = len(sets[a] & sets[b])
                pct = inter / len(sets[a]) * 100 if sets[a] else 0
                row[b] = f"{inter} ({pct:.0f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def export_excel_download(df: pd.DataFrame, filename: str = "validation_diff.xlsx") -> None:
    """Add a download button for the DataFrame as Excel."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    st.download_button(
        label=f"下载 {filename}",
        data=buf.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
