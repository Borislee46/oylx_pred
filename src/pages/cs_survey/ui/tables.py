from __future__ import annotations

import pandas as pd
import streamlit as st


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def _mix_with_white(color: str, strength: float) -> str:
    r, g, b = _hex_to_rgb(color)
    base = max(0.0, min(1.0, strength))
    rr = round(255 - (255 - r) * base)
    gg = round(255 - (255 - g) * base)
    bb = round(255 - (255 - b) * base)
    return f"background-color: rgb({rr}, {gg}, {bb}); color: #0F172A; font-weight: 600;"


def _coerce_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keys = ("评分均值", "意见率", "表扬率", "评分量", "意见量", "表扬量", "分数")
    for c in out.columns:
        if c in keys or "均值" in str(c) or str(c).endswith("率"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _column_tone(name: object) -> str | None:
    text = str(name)
    if text in ("评分均值", "分数") or "均值" in text or "表扬率" in text or "表扬" in text:
        return "good"
    if "意见率" in text or "意见量" in text or "投诉" in text or "负向" in text:
        return "bad"
    if text == "评分量":
        return "volume"
    return None


def _style_series(series: pd.Series) -> list[str]:
    if not pd.api.types.is_numeric_dtype(series):
        return [""] * len(series)
    tone = _column_tone(series.name)
    if tone is None:
        return [""] * len(series)
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return [""] * len(series)
    lo = float(valid.min())
    hi = float(valid.max())
    span = hi - lo
    if span <= 0:
        base = 0.38
        if tone == "bad":
            return [_mix_with_white("#FCA5A5", base) if pd.notna(v) else "" for v in numeric]
        if tone == "good":
            return [_mix_with_white("#86EFAC", base) if pd.notna(v) else "" for v in numeric]
        return [_mix_with_white("#93C5FD", 0.3) if pd.notna(v) else "" for v in numeric]

    styles: list[str] = []
    for value in numeric:
        if pd.isna(value):
            styles.append("")
            continue
        ratio = (float(value) - lo) / span
        strength = 0.2 + ratio * 0.55
        if tone == "bad":
            styles.append(_mix_with_white("#FCA5A5", strength))
        elif tone == "good":
            styles.append(_mix_with_white("#86EFAC", strength))
        else:
            styles.append(_mix_with_white("#93C5FD", 0.18 + ratio * 0.36))
    return styles


def _style_dataframe(df: pd.DataFrame):
    df = _coerce_metric_columns(df)
    cols = list(df.columns)
    mean_cols = [c for c in cols if c in ("评分均值", "分数") or "均值" in str(c)]
    pct_cols = [c for c in cols if c in ("意见率", "表扬率")]
    count_cols = [c for c in cols if str(c).endswith("量")]

    fmt = {}
    for c in mean_cols:
        fmt[c] = "{:.3f}"
    for c in pct_cols:
        fmt[c] = "{:.1f}%"
    for c in count_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            fmt[c] = "{:.0f}"

    sty = df.style.hide(axis="index")
    if fmt:
        sty = sty.format(fmt, na_rep="")
    highlight_cols = [c for c in cols if _column_tone(c) is not None]
    if highlight_cols:
        sty = sty.apply(_style_series, subset=highlight_cols)
    return sty


def _drop_hidden_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if str(c).replace("\ufeff", "").strip() == "序列"]
    return df.drop(columns=drop, errors="ignore")


def render_table(df: pd.DataFrame, height: int) -> None:
    if df is None or df.empty:
        st.caption("暂无数据")
        return
    disp = _drop_hidden_columns(df)
    st.dataframe(
        _style_dataframe(disp),
        width="stretch",
        height=height,
        hide_index=True,
    )


def theme_without_other(df: pd.DataFrame, col: str = "反馈类型") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    s = df[col].astype(str).str.strip()
    return df[~s.isin(["其他", "其它"])].copy()
