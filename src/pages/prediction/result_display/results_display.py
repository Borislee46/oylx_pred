import pandas as pd
import streamlit as st

from src.pages.prediction.core.utils import get_school_major_details
from src.pages.prediction.data_sort_config import (
    TOP_CROSS_RESULT_UI_CONFIG,
    TOP_SIM_RESULT_UI_CONFIG,
    UNIVERSITY_ORDER_MAP,
    UNIVERSITY_SORT_ORDER,
)
from src.pages.prediction.result_modifier.config import TOP_N_RECOMMENDATIONS
from src.pages.prediction.result_modifier.utils import get_probability
from src.utils.session_manager import SessionManager


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
        else:
            column_config[col_name] = st.column_config.TextColumn(label=label, width=width)

    return column_config


class ResultsDisplay:
    def __init__(
        self,
        top_similarity_results=None,
        top_cross_major_results=None,
        user_specified_results=None,
    ):
        self.top_similarity_results = top_similarity_results or []
        self.top_cross_major_results = top_cross_major_results or []
        self.user_specified_results = user_specified_results or []

        session_manager = SessionManager()
        is_cross_faculty = session_manager.get("cross_faculty_confirmed", False)

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
        if major_col and df[major_col].str.contains("(new!)", regex=False).any():
            styled_df = df.style.map(
                lambda v: "color: #06b6d4;" if isinstance(v, str) and "(new!)" in v else "",
                subset=[major_col],
            )

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

        data = {
            "推荐院校": [r["university"] for r in sorted_results],
            "推荐专业": [
                f"{r['major']}(new!)" if r.get("is_new_major") else r["major"]
                for r in sorted_results
            ],
            "录取概率": [get_probability(r) for r in sorted_results],
            "推荐专业详情": [
                get_school_major_details(r.get("university"), r.get("major")) or ""
                for r in sorted_results
            ],
        }
        return pd.DataFrame(data)

    def display(self):
        st.html(
            '<p style="font-size:0.68rem;color:var(--hk-slate-400);'
            'margin-top:0.5rem;text-align:center">'
            "机器学习算法未将时政变化、最新校方招生政策等作为特征因子，预测的录取概率仅供参考。"
            "</p>"
        )
        return  # ── 完整列表暂时隐藏 ──

        session_manager = SessionManager()
        selected_majors = session_manager.get("selected_target_majors", [])
        has_user_specified = bool(selected_majors) and bool(self.user_specified_results)
        has_similarity = bool(self.top_similarity_results)
        has_cross_major = bool(self.top_cross_major_results)

        if not (has_user_specified or has_similarity or has_cross_major):
            st.info("无推荐结果。")
            return

        if has_user_specified:
            with st.expander("查看完整列表"):
                self._display_type("user_specified")
        elif has_similarity and has_cross_major:
            with st.expander("查看完整列表"):
                col1, col2 = st.columns(2)
                with col1:
                    self._display_type("similarity")
                with col2:
                    self._display_type("cross_major")
        elif has_similarity:
            with st.expander("查看完整列表"):
                self._display_type("similarity")
        elif has_cross_major:
            with st.expander("查看完整列表"):
                self._display_type("cross_major")

    def _display_type(self, result_type: str):
        max_items = None if result_type == "user_specified" else TOP_N_RECOMMENDATIONS
        df = self.get_result_dataframe(result_type, max_items=max_items)
        self.display_dataframe(
            df, self.result_types[result_type]["config"], result_type=result_type
        )
