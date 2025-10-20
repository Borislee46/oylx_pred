import pandas as pd
import streamlit as st

from src.pages.prediction.data_sort_config.top_result_school_order import (
    UNIVERSITY_ORDER_MAP,
    UNIVERSITY_SORT_ORDER,
)
from src.pages.prediction.data_sort_config.top_result_ui_config import (
    TOP_CROSS_RESULT_UI_CONFIG,
    TOP_SIM_RESULT_UI_CONFIG,
)
from src.pages.prediction.result_modifier.config import TOP_N_RECOMMENDATIONS
from src.utils.session_manager import SessionManager


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

        self.result_types = {
            "similarity": {
                "results": self.top_similarity_results,
                "title": f"相似专业录取率 Top {TOP_N_RECOMMENDATIONS}",
                "config": TOP_SIM_RESULT_UI_CONFIG,
            },
            "cross_major": {
                "results": self.top_cross_major_results,
                "title": f"潜力跨专业方向 Top {TOP_N_RECOMMENDATIONS}",
                "config": TOP_CROSS_RESULT_UI_CONFIG,
            },
            "user_specified": {
                "results": self.user_specified_results,
                "title": "您指定的目标学校专业预测",
                "config": TOP_SIM_RESULT_UI_CONFIG,
            },
        }

    def _get_probability_value(self, probability):
        return float(probability) if probability is not None else 0.0

    def _create_styled_dataframe(self, df):
        if df.empty:
            return df

        style_needed = False
        style_columns = []

        if "目标专业" in df.columns:
            if df["目标专业"].astype(str).str.contains("(New!)", regex=False).any():
                style_needed = True
                style_columns.append("目标专业")

        if "变化" in df.columns:
            if df["变化"].astype(str).str.strip().ne("").any():
                style_needed = True
                style_columns.append("变化")

        if not style_needed:
            return df

        def style_cells(val):
            if isinstance(val, str):
                if "(New!)" in val:
                    return "color: #FF4B4B; font-weight: bold;"
                elif val.startswith("+"):
                    return "color: #28a745; font-weight: bold;"
                elif val.startswith("-"):
                    return "color: #dc3545; font-weight: bold;"
            return ""

        return df.style.map(style_cells, subset=style_columns)

    def _get_column_config(self, df, column_widths=None):
        column_widths = column_widths or {}

        if "专业详情" in df.columns and "专业详情" not in column_widths:
            column_widths["专业详情"] = "large"
        if "变化" in df.columns and "变化" not in column_widths:
            column_widths["变化"] = "small"

        column_config = {}
        for col_name in df.columns:
            width = column_widths.get(col_name, "small")

            if col_name == "专业详情":
                column_config[col_name] = st.column_config.TextColumn(
                    width=width, help="学校和专业的详细信息", max_chars=None
                )
            elif col_name == "录取概率":
                column_config[col_name] = st.column_config.ProgressColumn(
                    width=width, help="录取概率", min_value=0, max_value=1, format=" "
                )
            elif col_name == "变化":
                column_config[col_name] = st.column_config.TextColumn(
                    width=width, help="相对上次的概率变化（±%）"
                )
            else:
                column_config[col_name] = st.column_config.TextColumn(width=width)

        return column_config

    def _display_dataframe(self, df, column_widths=None):
        if df.empty:
            st.info("没有可显示的预测结果")
            return

        df = self._clean_and_reorder_columns(df)

        styled_df = self._create_styled_dataframe(df)

        column_config = self._get_column_config(df, column_widths)

        st.data_editor(styled_df, hide_index=True, column_config=column_config, disabled=True)

    def _clean_and_reorder_columns(self, df):
        if "变化" in df.columns and df["变化"].astype(str).str.strip().eq("").all():
            df = df.drop(columns=["变化"])
            return df

        if "变化" in df.columns and "录取概率" in df.columns:
            cols = list(df.columns)
            cols.remove("变化")
            insert_pos = cols.index("录取概率") + 1
            cols = cols[:insert_pos] + ["变化"] + cols[insert_pos:]
            df = df[cols]

        return df

    def _calculate_delta(self, result, prev_prob_map):
        key = (result.get("university"), result.get("major"))
        prev_p = (
            float(prev_prob_map.get(key, 0.0))
            if prev_prob_map and prev_prob_map.get(key) is not None
            else None
        )
        cur_p = float(result.get("probability", 0.0) or 0.0)

        if prev_p is None:
            return ""

        diff_pct = (cur_p - prev_p) * 100.0
        if abs(diff_pct) < 0.05:
            return ""
        elif diff_pct > 0:
            return f"+{diff_pct:.1f}%"
        else:
            return f"{diff_pct:.1f}%"

    def _create_results_dataframe(
        self,
        results: list,
        prev_prob_map: dict | None = None,
        show_delta: bool = False,
        max_items: int | None = None,
    ):
        if not results:
            return pd.DataFrame(columns=["目标院校", "目标专业", "录取概率", "专业中文名称"])

        results.sort(
            key=lambda item: (
                UNIVERSITY_ORDER_MAP.get(item.get("university"), len(UNIVERSITY_SORT_ORDER)),
                -self._get_probability_value(item.get("probability")),
            )
        )

        if isinstance(max_items, int) and max_items > 0:
            results = results[:max_items]

        data = {
            "目标院校": [result["university"] for result in results],
            "目标专业": [
                (
                    f"{result['major']} (New!)"
                    if result.get("is_new_major", False)
                    else result["major"]
                )
                for result in results
            ],
            "录取概率": [
                self._get_probability_value(result.get("probability")) for result in results
            ],
            "专业中文名称": [result.get("chinese_name", "") for result in results],
        }

        if show_delta:
            data["变化"] = [self._calculate_delta(result, prev_prob_map) for result in results]

        return pd.DataFrame(data)

    def _get_result_dataframe(
        self, result_type, prev_prob_map=None, show_delta=False, max_items=None
    ):
        config = self.result_types[result_type]
        return self._create_results_dataframe(
            results=config["results"],
            prev_prob_map=prev_prob_map,
            show_delta=show_delta,
            max_items=max_items,
        )

    def _should_show_delta(
        self,
        target_universities,
        target_majors,
        background_university,
        background_major,
    ):
        session_manager = SessionManager()
        prev_context_key = session_manager.get("previous_context_key")
        prev_prob_map = session_manager.get("previous_prob_map", {})
        form_data_changed = session_manager.get("form_data_changed", False)

        prob_map_to_use = (
            session_manager.get("prev_prev_prob_map", {}) if form_data_changed else prev_prob_map
        )

        target_unis_sorted = tuple(sorted(target_universities)) if target_universities else ()
        target_majs_sorted = tuple(sorted(target_majors)) if target_majors else ()
        cur_context_key = (
            background_university,
            background_major,
            target_unis_sorted,
            target_majs_sorted,
        )

        return (
            isinstance(prev_context_key, tuple)
            and prev_context_key == cur_context_key
            and isinstance(prob_map_to_use, dict)
            and bool(prob_map_to_use)
        ), prob_map_to_use

    def _display_results_layout(
        self, has_user_specified, has_similarity, has_cross_major, pool_is_large
    ):
        if pool_is_large:
            self._display_large_pool_layout(has_similarity, has_cross_major)
        else:
            self._display_normal_layout(has_user_specified, has_similarity, has_cross_major)

    def _display_large_pool_layout(self, has_similarity, has_cross_major):
        if not (has_similarity or has_cross_major):
            st.info("没有可显示的预测结果。")
            return

        if has_similarity and has_cross_major:
            col1, col2 = st.columns(2)
            with col1:
                self._display_result_type("similarity")
            with col2:
                self._display_result_type("cross_major")
        elif has_similarity:
            self._display_result_type("similarity")
        else:
            self._display_result_type("cross_major")

    def _display_normal_layout(self, has_user_specified, has_similarity, has_cross_major):
        if has_user_specified:
            self._display_result_type("user_specified")
        elif has_similarity and has_cross_major:
            col1, col2 = st.columns(2)
            with col1:
                self._display_result_type("similarity")
            with col2:
                self._display_result_type("cross_major")
        elif has_similarity:
            self._display_result_type("similarity")
        elif has_cross_major:
            self._display_result_type("cross_major")
        else:
            st.info("没有可显示的预测结果。")

    def _display_result_type(self, result_type):
        config = self.result_types[result_type]
        st.success(config["title"])

        max_items = None if result_type == "user_specified" else TOP_N_RECOMMENDATIONS

        df = self._get_result_dataframe(
            result_type,
            prev_prob_map=self.prob_map_to_use,
            show_delta=self.show_delta,
            max_items=max_items,
        )
        self._display_dataframe(df, config["config"])

    def display(
        self,
        target_universities,
        target_majors,
        background_university=None,
        background_major=None,
    ):
        session_manager = SessionManager()

        if not any(
            [
                self.top_similarity_results,
                self.top_cross_major_results,
                self.user_specified_results,
            ]
        ):
            combination_count = session_manager.get("combination_count", 0)
            if combination_count > 0:
                st.info("结果生成中，请稍候…")
            else:
                st.info("没有可显示的预测结果。")
            return

        self.show_delta, self.prob_map_to_use = self._should_show_delta(
            target_universities, target_majors, background_university, background_major
        )

        combination_count = session_manager.get("combination_count", 0)
        pool_is_large = isinstance(combination_count, int) and combination_count > 100

        has_user_specified = (not pool_is_large) and bool(self.user_specified_results)
        has_similarity = bool(self.top_similarity_results)
        has_cross_major = bool(self.top_cross_major_results)

        self._display_results_layout(
            has_user_specified, has_similarity, has_cross_major, pool_is_large
        )
