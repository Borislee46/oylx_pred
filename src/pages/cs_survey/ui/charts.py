from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"

THEME = {
    "text": "#0F172A",
    "muted": "#64748B",
    "border": "#CBD5E1",
    "axis": "#94A3B8",
    "chart_grid": "#E2E8F0",
    "bg": "#FFFFFF",
    "accent": "#006B67",
    "accent_soft": "#83C8BF",
    "accent_deep": "#00514E",
    "opinion": "#F5A623",
    "praise": "#006B67",
    "tooltip_bg": "rgba(255,255,255,0.96)",
}

COL_OPINION = THEME["opinion"]
COL_PRAISE = THEME["praise"]
DETAIL_LINE = THEME["accent"]
DETAIL_FILL = "rgba(0, 107, 103, 0.10)"


def _chart_id(key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return f"cs_e_{h}"


def _json_for_script(obj) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _render_html(html: str, height: int) -> None:
    st.iframe(html, height=height)


def _render_empty_state(height: int, text: str = "当前筛选条件下暂无可展示数据") -> None:
    h = max(height, 96)
    st.markdown(
        f'<div class="cs-empty-state" style="min-height:{h}px">{text}</div>',
        unsafe_allow_html=True,
    )


def _auto_chart_height(count: int, base: int, per_item: int, max_height: int = 560) -> int:
    if count <= 0:
        return base
    return max(base, min(max_height, count * per_item + 80))


def _top_n_rows(
    df: pd.DataFrame,
    value_col: str,
    top_n: int | None,
    *,
    ascending: bool = False,
) -> pd.DataFrame:
    if top_n is None or top_n <= 0 or value_col not in df.columns:
        return df
    sub = df.copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[value_col])
    if sub.empty:
        return sub
    return sub.nlargest(top_n, value_col) if not ascending else sub.nsmallest(top_n, value_col)


def _base_text_style(font_size: int = 10) -> dict:
    return {
        "color": THEME["muted"],
        "fontSize": font_size,
        "fontFamily": "Inter, Microsoft YaHei, sans-serif",
    }


def _tooltip() -> dict:
    return {
        "trigger": "axis",
        "backgroundColor": THEME["tooltip_bg"],
        "borderColor": "rgba(203, 213, 225, 0.9)",
        "borderWidth": 1,
        "textStyle": {"color": THEME["text"], "fontSize": 11},
        "padding": [8, 10],
        "extraCssText": "box-shadow: 0 8px 24px rgba(15,23,42,0.10); border-radius: 10px;",
    }


def _axis_label(*, width: int | None = None, rotate: int = 0) -> dict:
    out: dict = {"color": THEME["muted"], "fontSize": 10}
    if width is not None:
        out["width"] = width
        out["overflow"] = "truncate"
    if rotate:
        out["rotate"] = rotate
    return out


def _category_axis(data: list[str], *, rotate: int = 0) -> dict:
    return {
        "type": "category",
        "data": data,
        "axisLine": {"lineStyle": {"color": THEME["axis"], "width": 1}},
        "axisTick": {"show": False},
        "axisLabel": _axis_label(rotate=rotate),
    }


def _value_axis(*, min_value=None, max_value=None, min_interval: int | None = None) -> dict:
    axis = {
        "type": "value",
        "axisLine": {"show": False},
        "axisTick": {"show": False},
        "splitLine": {"lineStyle": {"color": THEME["chart_grid"], "type": "dashed"}},
        "axisLabel": _axis_label(),
    }
    if min_value is not None:
        axis["min"] = min_value
    if max_value is not None:
        axis["max"] = max_value
    if min_interval is not None:
        axis["minInterval"] = min_interval
    return axis


def _bar_series(
    data: list[float] | list[int],
    *,
    color: str,
    horizontal: bool = False,
    name: str | None = None,
    radius: list[int] | None = None,
    show_label: bool = True,
) -> dict:
    label_position = "right" if horizontal else "top"
    series = {
        "type": "bar",
        "data": data,
        "barMaxWidth": 20 if horizontal else 28,
        "label": {
            "show": show_label,
            "position": label_position,
            "color": THEME["muted"],
            "fontSize": 10,
        },
        "itemStyle": {
            "color": color,
            "borderRadius": radius or ([0, 8, 8, 0] if horizontal else [8, 8, 0, 0]),
        },
    }
    if name is not None:
        series["name"] = name
    return series


def _legend(data: list[str]) -> dict:
    return {
        "data": data,
        "top": 6,
        "itemWidth": 10,
        "itemHeight": 10,
        "textStyle": {"color": THEME["muted"], "fontSize": 11},
    }


def render_echarts_option(option: dict, height: int, key: str) -> None:
    cid = _chart_id(key)
    j = _json_for_script(option)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><script src="{ECHARTS_CDN}"></script></head><body style="margin:0;padding:0;background:transparent;">
<div id="{cid}" style="width:100%;height:{height}px;"></div>
<script>
(function() {{
  const el = document.getElementById("{cid}");
  const chart = echarts.init(el);
  const resize = function() {{ chart.resize(); }};
  chart.setOption({j});
  if (window.ResizeObserver) {{
    const ro = new ResizeObserver(resize);
    ro.observe(document.body);
    ro.observe(el);
  }}
  window.addEventListener('load', resize);
  window.addEventListener('resize', resize);
  setTimeout(resize, 60);
}})();
</script></body></html>"""
    _render_html(html, height)


def render_bar_dist(dist: pd.DataFrame, height: int, key: str) -> None:
    if dist.empty or "分值" not in dist.columns or "数量" not in dist.columns:
        _render_empty_state(height)
        return
    x = [str(int(x)) for x in dist["分值"].tolist()]
    y = [int(v) for v in dist["数量"].tolist()]
    opt = {
        "color": [THEME["accent"]],
        "animationDuration": 500,
        "textStyle": _base_text_style(10),
        "grid": {"left": "8%", "right": "4%", "bottom": "12%", "top": "16%", "containLabel": True},
        "xAxis": _category_axis(x),
        "yAxis": _value_axis(min_interval=1),
        "series": [_bar_series(y, color=THEME["accent"])],
        "tooltip": _tooltip(),
    }
    render_echarts_option(opt, height, key)


def render_hbar_feedback(
    labels: list[str],
    values: list[float],
    color: str,
    height: int,
    key: str,
    top_n: int | None = None,
) -> None:
    if not labels:
        _render_empty_state(height)
        return
    if top_n is not None and top_n > 0:
        pairs = sorted(zip(labels, values, strict=False), key=lambda item: item[1], reverse=True)[
            :top_n
        ]
        pairs = list(reversed(pairs))
        labels = [item[0] for item in pairs]
        values = [item[1] for item in pairs]
    chart_h = _auto_chart_height(len(labels), height, 30)
    opt = {
        "color": [color],
        "animationDuration": 500,
        "textStyle": _base_text_style(10),
        "grid": {"left": "7%", "right": "8%", "bottom": "6%", "top": "6%", "containLabel": True},
        "xAxis": _value_axis(min_interval=1),
        "yAxis": {
            "type": "category",
            "data": labels,
            "axisLine": {"show": False},
            "axisTick": {"show": False},
            "axisLabel": _axis_label(width=148),
        },
        "series": [_bar_series(values, color=color, horizontal=True)],
        "tooltip": {**_tooltip(), "axisPointer": {"type": "shadow"}},
    }
    render_echarts_option(opt, chart_h, key)


def render_area_detail(df: pd.DataFrame, height: int, key: str) -> None:
    if df.empty or "明细" not in df.columns:
        _render_empty_state(height)
        return
    sub = df.copy()
    sub["评分均值"] = pd.to_numeric(sub["评分均值"], errors="coerce")
    sub = sub.dropna(subset=["评分均值"])
    if sub.empty:
        _render_empty_state(height)
        return
    x = [str(v) for v in sub["明细"].tolist()]
    y = [round(float(v), 3) for v in sub["评分均值"].tolist()]
    opt = {
        "color": [DETAIL_LINE],
        "animationDuration": 550,
        "textStyle": _base_text_style(10),
        "grid": {"left": "7%", "right": "5%", "bottom": "18%", "top": "10%", "containLabel": True},
        "xAxis": _category_axis(x, rotate=18),
        "yAxis": _value_axis(min_value=0, max_value=5.5),
        "series": [
            {
                "type": "line",
                "smooth": True,
                "symbol": "circle",
                "symbolSize": 7,
                "data": y,
                "lineStyle": {"width": 2.5, "color": DETAIL_LINE},
                "itemStyle": {"color": "#fff", "borderColor": DETAIL_LINE, "borderWidth": 2},
                "areaStyle": {"color": DETAIL_FILL},
            }
        ],
        "tooltip": _tooltip(),
    }
    render_echarts_option(opt, height, key)


def render_three_pillar_bars(
    matrix: pd.DataFrame, height: int, key: str, pillar_names: list[str]
) -> None:
    if matrix.empty or "维度" not in matrix.columns:
        _render_empty_state(height)
        return
    sub = matrix.copy()
    sub["意见量"] = pd.to_numeric(sub["意见量"], errors="coerce").fillna(0).astype(int)
    sub["表扬量"] = pd.to_numeric(sub["表扬量"], errors="coerce").fillna(0).astype(int)
    mapping = sub.set_index("维度")[["意见量", "表扬量"]].to_dict("index")
    dims = pillar_names or sub["维度"].astype(str).tolist()
    opinion = [int(mapping.get(dim, {}).get("意见量", 0)) for dim in dims]
    praise = [int(mapping.get(dim, {}).get("表扬量", 0)) for dim in dims]
    opt = {
        "color": [COL_OPINION, COL_PRAISE],
        "animationDuration": 500,
        "legend": _legend(["意见", "表扬"]),
        "textStyle": _base_text_style(10),
        "grid": {"left": "8%", "right": "4%", "bottom": "10%", "top": "20%", "containLabel": True},
        "xAxis": _category_axis(dims),
        "yAxis": _value_axis(min_interval=1),
        "series": [
            _bar_series(opinion, name="意见", color=COL_OPINION),
            _bar_series(praise, name="表扬", color=COL_PRAISE),
        ],
        "tooltip": {**_tooltip(), "axisPointer": {"type": "shadow"}},
    }
    render_echarts_option(opt, height, key)


def render_grouped_opinion_praise(
    matrix: pd.DataFrame, height: int, key: str, top_n: int | None = None
) -> None:
    if matrix.empty or "维度" not in matrix.columns:
        _render_empty_state(height)
        return
    sub = matrix.copy()
    sub["意见量"] = pd.to_numeric(sub["意见量"], errors="coerce").fillna(0).astype(int)
    sub["表扬量"] = pd.to_numeric(sub["表扬量"], errors="coerce").fillna(0).astype(int)
    sub["总反馈量"] = sub["意见量"] + sub["表扬量"]
    sub = _top_n_rows(sub, "总反馈量", top_n)
    sub = sub.sort_values(["总反馈量", "意见量"], ascending=True)
    dims = sub["维度"].astype(str).tolist()
    op = sub["意见量"].tolist()
    pr = sub["表扬量"].tolist()
    chart_h = _auto_chart_height(len(dims), height + 50, 34)
    opt = {
        "color": [COL_OPINION, COL_PRAISE],
        "animationDuration": 500,
        "legend": _legend(["意见", "表扬"]),
        "textStyle": _base_text_style(10),
        "grid": {"left": "26%", "right": "8%", "bottom": "6%", "top": "18%", "containLabel": True},
        "xAxis": _value_axis(min_interval=1),
        "yAxis": {
            "type": "category",
            "data": dims,
            "axisLine": {"show": False},
            "axisTick": {"show": False},
            "axisLabel": _axis_label(width=138),
        },
        "series": [
            _bar_series(op, name="意见", color=COL_OPINION, horizontal=True),
            _bar_series(pr, name="表扬", color=COL_PRAISE, horizontal=True),
        ],
        "tooltip": {**_tooltip(), "axisPointer": {"type": "shadow"}},
    }
    render_echarts_option(opt, chart_h, key)


def render_ranked_detail_scores(
    df: pd.DataFrame, height: int, key: str, top_n: int | None = None
) -> None:
    if df.empty or "明细" not in df.columns or "评分均值" not in df.columns:
        _render_empty_state(height)
        return
    sub = df.copy()
    sub["评分均值"] = pd.to_numeric(sub["评分均值"], errors="coerce")
    sub = sub.dropna(subset=["评分均值"])
    sub = _top_n_rows(sub, "评分均值", top_n)
    sub = sub.sort_values("评分均值", ascending=True)
    if sub.empty:
        _render_empty_state(height)
        return
    labels = sub["明细"].astype(str).tolist()
    values = [round(float(v), 3) for v in sub["评分均值"].tolist()]
    chart_h = _auto_chart_height(len(labels), height + 50, 34)
    opt = {
        "color": [DETAIL_LINE],
        "animationDuration": 500,
        "textStyle": _base_text_style(10),
        "grid": {"left": "26%", "right": "10%", "bottom": "6%", "top": "8%", "containLabel": True},
        "xAxis": _value_axis(min_value=0, max_value=5.5),
        "yAxis": {
            "type": "category",
            "data": labels,
            "axisLine": {"show": False},
            "axisTick": {"show": False},
            "axisLabel": _axis_label(width=138),
        },
        "series": [
            {
                "type": "bar",
                "data": values,
                "barMaxWidth": 20,
                "label": {
                    "show": True,
                    "position": "right",
                    "color": THEME["muted"],
                    "fontSize": 10,
                    "formatter": "{c}",
                },
                "itemStyle": {
                    "color": DETAIL_LINE,
                    "borderRadius": [0, 8, 8, 0],
                },
            }
        ],
        "tooltip": {**_tooltip(), "axisPointer": {"type": "shadow"}},
    }
    render_echarts_option(opt, chart_h, key)
