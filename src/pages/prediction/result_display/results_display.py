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
from .delta_calculator import DeltaCalculator
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

        self.dataframe_builder = DataFrameBuilder()
        self.dataframe_styler = DataFrameStyler()
        self.delta_calculator = DeltaCalculator()
        self.layout_manager = LayoutManager(self)

    def _display_dataframe(self, df, column_widths=None, result_type=None):
        if df.empty:
            st.info("没有可显示的预测结果")
            return

        df = self.dataframe_builder.clean_and_reorder_columns(df)
        styled_df = self.dataframe_styler.create_styled_dataframe(df)
        column_config = self.dataframe_styler.get_column_config(df, column_widths)
        st.data_editor(
            styled_df,
            hide_index=True,
            column_config=column_config,
            disabled=True,
            key=f"prediction_result_editor_{result_type or 'default'}",
        )

    def _get_result_dataframe(
        self, result_type, prev_prob_map=None, show_delta=False, max_items=None
    ):
        config = self.result_types[result_type]
        return self.dataframe_builder.create_results_dataframe(
            results=config["results"],
            prev_prob_map=prev_prob_map,
            show_delta=show_delta,
            max_items=max_items,
            delta_calculator=self.delta_calculator,
        )

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
        self._display_dataframe(df, config["config"], result_type=result_type)

    def display(
        self,
        target_universities,
        target_majors,
        background_university=None,
        background_major=None,
        input_data=None,
    ):
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
            st.info("结果生成中，请稍候…" if combination_count > 0 else "没有可显示的预测结果。")
            return

        self.show_delta, self.prob_map_to_use = self.delta_calculator.should_show_delta(
            target_universities, target_majors, background_university, background_major
        )

        combination_count = session_manager.get("combination_count", 0)
        pool_is_large = isinstance(combination_count, int) and combination_count > 100
        has_user_specified = (not pool_is_large) and bool(self.user_specified_results)
        has_similarity = bool(self.top_similarity_results)
        has_cross_major = bool(self.top_cross_major_results)

        self.layout_manager.display_results_layout(
            has_user_specified, has_similarity, has_cross_major, pool_is_large
        )
