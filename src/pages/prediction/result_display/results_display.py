import pandas as pd
import streamlit as st

from src.pages.prediction.core.utils import get_school_major_details
from src.pages.prediction.data_sort_config import (
    TOP_CROSS_RESULT_UI_CONFIG,
    TOP_SIM_RESULT_UI_CONFIG,
    UNIVERSITY_ORDER_MAP,
    UNIVERSITY_SORT_ORDER,
)
from src.pages.prediction.handler_config import DEFAULT_FORM_KEYS, DEFAULT_UI_KEYS
from src.pages.prediction.result_display.hero_summary import render_hero_summary
from src.pages.prediction.result_display.trace_display import render_trace_for_results
from src.pages.prediction.result_modifier.config import TOP_N_RECOMMENDATIONS
from src.pages.prediction.result_modifier.utils import get_probability
from src.utils.session_manager import SessionManager


def _delta_cell_style(v: str) -> str:
    if not isinstance(v, str):
        return ""
    if v.startswith("+"):
        return "color: #059669; font-weight: 600;"
    if v.startswith("-"):
        return "color: #dc2626; font-weight: 600;"
    if v == "NEW":
        return "color: #2563eb; font-weight: 600;"
    return ""


def get_column_config(
    df: pd.DataFrame, column_widths: dict | None = None, label_map: dict | None = None
):
    column_widths = column_widths or {}
    label_map = label_map or {}

    if "推荐专业详情" in df.columns and "推荐专业详情" not in column_widths:
        column_widths["推荐专业详情"] = "large"

    column_config = {}
    for col_name in df.columns:
        width = column_widths.get(col_name, "small")
        label = label_map.get(col_name)

        if col_name == "推荐专业详情":
            column_config[col_name] = st.column_config.TextColumn(label=label, width=width)
        elif col_name == "录取概率":
            column_config[col_name] = st.column_config.ProgressColumn(
                label=label,
                width=width,
                min_value=0,
                max_value=1,
                format=" ",
                pinned=True,
                color="#06b6d4",
            )
        elif col_name == "±%":
            column_config[col_name] = st.column_config.TextColumn(
                label=label,
                width=width,
                pinned=True,
                help="与上次预测的概率变化：▲上涨 ▼下跌 —不变 NEW新增",
            )
        else:
            column_config[col_name] = st.column_config.TextColumn(label=label, width=width)

    return column_config


class ResultsDisplay:
    def __init__(
        self,
        top_similarity_results=None,
        top_cross_major_results=None,
        user_specified_results=None,
        prev_prob_map: dict | None = None,
        delta_calculator=None,
    ):
        self.top_similarity_results = top_similarity_results or []
        self.top_cross_major_results = top_cross_major_results or []
        self.user_specified_results = user_specified_results or []
        self.prev_prob_map = prev_prob_map
        self.delta_calculator = delta_calculator

        session_manager = SessionManager()
        is_cross_faculty = session_manager.get(DEFAULT_UI_KEYS.cross_faculty_confirmed, False)

        sim_title = "相似（相对）专业" if is_cross_faculty else "相似专业"
        cross_title = "潜力跨（相对）专业" if is_cross_faculty else "潜力跨专业"

        self.result_types = {
            "similarity": {
                "results": self.top_similarity_results,
                "title": sim_title,
                "config": TOP_SIM_RESULT_UI_CONFIG,
            },
            "cross_major": {
                "results": self.top_cross_major_results,
                "title": cross_title,
                "config": TOP_CROSS_RESULT_UI_CONFIG,
            },
            "user_specified": {
                "results": self.user_specified_results,
                "title": "指定专业",
                "config": TOP_SIM_RESULT_UI_CONFIG,
            },
        }

    def display_dataframe(
        self, df: pd.DataFrame, column_widths: dict | None = None, result_type: str | None = None
    ):
        if df.empty:
            st.info("暂无可展示内容")
            return

        label_map = {}
        if result_type in self.result_types:
            title = self.result_types[result_type]["title"]
            label_map["推荐专业"] = title if title == "指定专业" else f"{title}推荐"

        styled_df = df
        major_col = next((c for c in df.columns if c.startswith("推荐专业")), None)
        has_new_major = bool(major_col and df[major_col].str.contains("(new!)", regex=False).any())
        has_delta_col = "±%" in df.columns

        if has_new_major or has_delta_col:
            styled_df = df.style
            if has_new_major:
                styled_df = styled_df.map(
                    lambda v: "color: #06b6d4;" if isinstance(v, str) and "(new!)" in v else "",
                    subset=[major_col],
                )
            if has_delta_col:
                styled_df = styled_df.map(_delta_cell_style, subset=["±%"])

        st.dataframe(
            styled_df,
            hide_index=True,
            column_config=get_column_config(df, column_widths, label_map=label_map),
            key=f"prediction_result_df_{result_type or 'default'}",
        )

    def get_result_dataframe(self, result_type: str, max_items: int | None = None) -> pd.DataFrame:
        results = self.result_types[result_type]["results"]
        if not results:
            return pd.DataFrame(columns=["推荐院校", "推荐专业", "录取概率", "推荐专业详情"])

        sorted_results = sorted(
            results,
            key=lambda x: (
                UNIVERSITY_ORDER_MAP.get(x.get("university"), len(UNIVERSITY_SORT_ORDER)),
                -get_probability(x),
            ),
        )

        if max_items:
            sorted_results = sorted_results[:max_items]

        data: dict[str, list] = {
            "推荐院校": [r["university"] for r in sorted_results],
            "推荐专业": [
                f"{r['major']}(new!)" if r.get("is_new_major") else r["major"]
                for r in sorted_results
            ],
            "录取概率": [get_probability(r) for r in sorted_results],
        }

        if self.prev_prob_map and self.delta_calculator:
            data["±%"] = [
                self.delta_calculator.calculate_delta(r, self.prev_prob_map) for r in sorted_results
            ]

        data["推荐专业详情"] = [
            get_school_major_details(r.get("university"), r.get("major")) or ""
            for r in sorted_results
        ]

        return pd.DataFrame(data)

    def display(self):
        session_manager = SessionManager()
        selected_majors = session_manager.get(DEFAULT_FORM_KEYS.selected_target_majors, [])
        has_user_specified = bool(selected_majors) and bool(self.user_specified_results)
        has_similarity = bool(self.top_similarity_results)
        has_cross_major = bool(self.top_cross_major_results)

        if not (has_user_specified or has_similarity or has_cross_major):
            st.info("无推荐结果。")
            return

        all_candidates = sorted(
            (
                (self.top_similarity_results or [])
                + (self.top_cross_major_results or [])
                + (self.user_specified_results or [])
            ),
            key=lambda r: float(r.get("probability", 0.0) or 0.0),
            reverse=True,
        )
        render_hero_summary(all_candidates)

        if has_user_specified:
            self._display_table("user_specified")
        elif has_similarity and has_cross_major:
            col1, col2 = st.columns(2)
            with col1:
                self._display_table("similarity")
            with col2:
                self._display_table("cross_major")
        elif has_similarity:
            self._display_table("similarity")
        elif has_cross_major:
            self._display_table("cross_major")

        st.html(
            '<div class="hk-disclaimer">'
            "机器学习算法未将时政变化、最新校方招生政策等作为特征因子，预测的录取概率仅供参考。"
            "</div>"
        )

        with st.expander("这分数怎么来的？（top 3 召回算法链路trace）", key="trace_expander"):
            render_trace_for_results(all_candidates)

    def _display_table(self, result_type: str):
        max_items = None if result_type == "user_specified" else TOP_N_RECOMMENDATIONS
        df = self.get_result_dataframe(result_type, max_items=max_items)
        self.display_dataframe(
            df, self.result_types[result_type]["config"], result_type=result_type
        )
