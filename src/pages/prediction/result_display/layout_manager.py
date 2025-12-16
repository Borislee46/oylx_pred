import streamlit as st


class LayoutManager:
    def __init__(self, results_display):
        self.results_display = results_display

    def display_results_layout(
        self, has_user_specified, has_similarity, has_cross_major, pool_is_large
    ):
        if has_user_specified and not pool_is_large:
            self.results_display._display_result_type("user_specified")
            return

        self._display_recommendation_layout(has_similarity, has_cross_major)

    def _display_recommendation_layout(self, has_similarity, has_cross_major):
        if has_similarity and has_cross_major:
            col1, col2 = st.columns(2)
            with col1:
                self.results_display._display_result_type("similarity")
            with col2:
                self.results_display._display_result_type("cross_major")
        elif has_similarity:
            self.results_display._display_result_type("similarity")
        elif has_cross_major:
            self.results_display._display_result_type("cross_major")
        else:
            st.info("暂无推荐结果")
