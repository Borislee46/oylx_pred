import streamlit as st

from src.pages.prediction.data_sort_config.top_result_ui_config import (
    TOP_CROSS_RESULT_UI_CONFIG,
    TOP_SIM_RESULT_UI_CONFIG,
)
from src.pages.prediction.result_modifier.config import TOP_N_RECOMMENDATIONS
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

from .dataframe_builder import DataFrameBuilder
from .dataframe_styler import DataFrameStyler
from .layout_manager import LayoutManager

logger = setup_logger("page3", "prediction")


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
                "title": "相似专业 Top N",
                "config": TOP_SIM_RESULT_UI_CONFIG,
            },
            "cross_major": {
                "results": self.top_cross_major_results,
                "title": "跨专业 Top N",
                "config": TOP_CROSS_RESULT_UI_CONFIG,
            },
            "user_specified": {
                "results": self.user_specified_results,
                "title": "指定专业",
                "config": TOP_SIM_RESULT_UI_CONFIG,
            },
        }

        self.dataframe_builder = DataFrameBuilder()
        self.dataframe_styler = DataFrameStyler()
        self.layout_manager = LayoutManager(self)

    def _display_dataframe(self, df, column_widths=None, result_type=None):
        if df.empty:
            st.info("暂无可展示内容")
            return

        label_map = {}
        if result_type and result_type in self.result_types and "推荐专业" in df.columns:
            title = self.result_types[result_type]["title"]
            if title == "指定专业":
                label_map["推荐专业"] = title
            else:
                label_map["推荐专业"] = f"{title} 推荐"

        styled_df = self.dataframe_styler.create_styled_dataframe(df)
        column_config = self.dataframe_styler.get_column_config(
            df, column_widths, label_map=label_map
        )
        st.data_editor(
            styled_df,
            hide_index=True,
            column_config=column_config,
            disabled=True,
            key=f"prediction_result_editor_{result_type or 'default'}",
        )

    def _get_result_dataframe(self, result_type, max_items=None):
        config = self.result_types[result_type]
        return self.dataframe_builder.create_results_dataframe(
            results=config["results"],
            max_items=max_items,
        )

    def _display_result_type(self, result_type):
        config = self.result_types[result_type]

        max_items = None if result_type == "user_specified" else TOP_N_RECOMMENDATIONS

        df = self._get_result_dataframe(
            result_type,
            max_items=max_items,
        )
        self._display_dataframe(df, config["config"], result_type=result_type)

    def display(self):
        session_manager = SessionManager()
        has_results = any(
            [
                self.top_similarity_results,
                self.top_cross_major_results,
                self.user_specified_results,
            ]
        )

        if not has_results:
            combination_count = session_manager.get("combination_count", 0)
            st.info("无推荐结果。")
            return

        combination_count = session_manager.get("combination_count", 0)
        pool_is_large = isinstance(combination_count, int) and combination_count > 10
        has_user_specified = (not pool_is_large) and bool(self.user_specified_results)
        has_similarity = bool(self.top_similarity_results)
        has_cross_major = bool(self.top_cross_major_results)

        self.layout_manager.display_results_layout(
            has_user_specified, has_similarity, has_cross_major, pool_is_large
        )
