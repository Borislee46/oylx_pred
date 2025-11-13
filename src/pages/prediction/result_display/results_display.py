import streamlit as st

from src.agent.single_pred_shap_agent import SinglePredShapAgent
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
from .shap_explainer import ShapExplainer

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
        self.shap_explainer = None
        self.shap_agent = None

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

    def _display_shap_force_plot(
        self, input_data, prediction_model, feature_names, single_result, background_major=None
    ):
        try:
            import shap
            import streamlit_shap as st_shap
        except ImportError:
            st.warning("请安装 streamlit-shap: pip install streamlit-shap")
            return

        target_university = single_result.get("university")
        target_major = single_result.get("major")
        if not target_university or not target_major:
            return

        session_manager = SessionManager()
        if self.shap_explainer is None:
            self.shap_explainer = ShapExplainer(
                prediction_model.model, feature_names, prediction_model=prediction_model
            )

        result = self.shap_explainer.create_force_plot(
            input_data, target_university, target_major, prediction_model, session_manager
        )
        if result is None:
            st.info("无法生成特征影响分析图")
            return

        shap_values, raw_feature_values, expected_value, feature_names_filtered = result

        if len(shap_values) != len(raw_feature_values) or len(shap_values) != len(
            feature_names_filtered
        ):
            logger.error(
                f"SHAP数据长度不匹配: shap_values={len(shap_values)}, raw_feature_values={len(raw_feature_values)}, feature_names_filtered={len(feature_names_filtered)}"
            )
            st.error("SHAP数据长度不匹配，无法显示图表")
            return

        feature_names_display = self.shap_explainer._get_feature_display_names(
            feature_names_filtered
        )

        if len(feature_names_display) != len(shap_values):
            logger.error(f"特征名称长度不匹配: {len(feature_names_display)} != {len(shap_values)}")
            st.error("特征名称长度不匹配，无法显示图表")
            return

        try:
            import pandas as pd

            raw_values_series = pd.Series(raw_feature_values, index=feature_names_filtered)

            shap_values_reversed = -shap_values
            expected_value_reversed = 1.0 - expected_value

            force_plot_obj = shap.force_plot(
                expected_value_reversed,
                shap_values_reversed,
                raw_values_series,
                feature_names=feature_names_display,
                matplotlib=False,
                show=False,
            )

            st.markdown('<div style="max-height: 300px; overflow: auto;">', unsafe_allow_html=True)
            st_shap.st_shap(force_plot_obj, height=150)
            st.markdown("</div>", unsafe_allow_html=True)

            if self.shap_agent is None:
                self.shap_agent = SinglePredShapAgent()

            with st.status("正在分析特征影响...", expanded=True) as status:
                try:
                    explanation_stream = self.shap_agent.explain_shap_values_stream(
                        target_university=target_university,
                        target_major=target_major,
                        background_major=background_major or "",
                        feature_names=feature_names_display,
                        shap_values=shap_values_reversed,
                        raw_feature_values=raw_feature_values,
                        expected_value=expected_value_reversed,
                    )
                    st.markdown('<div style="font-size: 0.9em;">', unsafe_allow_html=True)
                    st.write_stream(explanation_stream)
                    st.markdown("</div>", unsafe_allow_html=True)
                    status.update(label="分析完成", state="complete")
                except Exception as e:
                    logger.error(f"流式生成SHAP解释失败: {e}", exc_info=True)
                    status.update(label="分析失败，尝试备用方案...", state="error")
                    explanation = self.shap_agent.explain_shap_values(
                        target_university=target_university,
                        target_major=target_major,
                        background_major=background_major or "",
                        feature_names=feature_names_display,
                        shap_values=shap_values_reversed,
                        raw_feature_values=raw_feature_values,
                        expected_value=expected_value_reversed,
                    )
                    if explanation:
                        st.markdown('<div style="font-size: 0.9em;">', unsafe_allow_html=True)
                        st.markdown(explanation)
                        st.markdown("</div>", unsafe_allow_html=True)
                        status.update(label="分析完成", state="complete")
                    else:
                        st.info("无法生成特征影响分析，请稍后重试")
                        status.update(label="无法生成分析", state="error")
        except Exception as e:
            logger.error(f"显示SHAP图表失败: {e}", exc_info=True)
            st.error(f"无法显示特征影响分析图: {e}")

    def display(
        self,
        target_universities,
        target_majors,
        background_university=None,
        background_major=None,
        input_data=None,
        prediction_model=None,
        feature_names=None,
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

        show_shap = (
            has_user_specified
            and len(self.user_specified_results) == 1
            and input_data
            and prediction_model
            and feature_names
        )

        if show_shap:
            self._display_result_type("user_specified")
            self._display_shap_force_plot(
                input_data,
                prediction_model,
                feature_names,
                self.user_specified_results[0],
                background_major,
            )
        else:
            self.layout_manager.display_results_layout(
                has_user_specified, has_similarity, has_cross_major, pool_is_large
            )
