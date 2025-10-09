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
from src.pages.prediction.prediction_utils import (
    format_school_major_details_from_row,
    get_school_major_details,
)
from src.pages.prediction.result_modifier.config import TOP_N_RECOMMENDATIONS
from src.utils.session_manager import SessionManager


class ResultsDisplay:
    def __init__(
        self, top_similarity_results=None, top_cross_major_results=None, user_specified_results=None
    ):
        self.top_similarity_results = top_similarity_results or []
        self.top_cross_major_results = top_cross_major_results or []
        self.user_specified_results = user_specified_results or []

    def _get_probability_value(self, probability):
        if probability is None:
            return 0.0
        return float(probability)

    def _display_dataframe(self, df, column_widths=None):
        if df.empty:
            st.info("没有可显示的预测结果")
            return

        apply_styler = False
        try:
            if "目标专业" in df.columns:
                if df["目标专业"].astype(str).str.contains("(New!)", regex=False).any():
                    apply_styler = True
        except Exception:
            apply_styler = False

        if apply_styler:

            def style_new_major(val):
                if isinstance(val, str) and "(New!)" in val:
                    return "color: #FF4B4B; font-weight: bold;"
                return ""

            data_to_render = df.style.map(style_new_major, subset=["目标专业"])
        else:
            data_to_render = df

        if column_widths is None:
            column_widths = {}
        if "专业详情" in df.columns and "专业详情" not in column_widths:
            column_widths["专业详情"] = "large"

        column_config = {}
        for col_name in df.columns:
            width = "small"
            if column_widths and col_name in column_widths:
                width = column_widths[col_name]

            if col_name == "专业详情":
                column_config[col_name] = st.column_config.TextColumn(
                    width=width, help="学校和专业的详细信息", max_chars=None
                )
            elif col_name == "录取概率":
                column_config[col_name] = st.column_config.ProgressColumn(
                    width=width, help="录取概率", min_value=0, max_value=1, format=" "
                )
            else:
                column_config[col_name] = st.column_config.TextColumn(width=width)

        st.data_editor(data_to_render, hide_index=True, column_config=column_config, disabled=True)

    def _get_details_in_batch(self, results, details_df_full=None):
        if not results:
            return []

        query_df = pd.DataFrame(
            [
                {"学校": r["university"], "专业英文名称": r["major"]}
                for r in results
                if isinstance(r, dict)
            ]
        ).drop_duplicates()

        if query_df.empty:
            return ["无详细信息"] * len(results)

        if details_df_full is None:
            details_df_full = get_school_major_details(None, None, return_df=True)
        if details_df_full is None or details_df_full.empty:
            return ["无详细信息"] * len(results)

        merged_df = pd.merge(query_df, details_df_full, on=["学校", "专业英文名称"], how="left")

        merged_df["formatted_details"] = merged_df.apply(
            format_school_major_details_from_row, axis=1
        )

        details_map = pd.Series(
            merged_df.formatted_details.values,
            index=pd.MultiIndex.from_frame(merged_df[["学校", "专业英文名称"]]),
        ).to_dict()

        return [
            details_map.get((r.get("university"), r.get("major")), "无详细信息") for r in results
        ]

    def _get_chinese_names_in_batch(self, results, details_df_full=None):
        if not results:
            return []

        query_df = pd.DataFrame(
            [
                {"学校": r["university"], "专业英文名称": r["major"]}
                for r in results
                if isinstance(r, dict)
            ]
        ).drop_duplicates()

        if query_df.empty:
            return [""] * len(results)

        if details_df_full is None:
            details_df_full = get_school_major_details(None, None, return_df=True)
        if details_df_full is None or details_df_full.empty:
            return [""] * len(results)

        merged_df = pd.merge(query_df, details_df_full, on=["学校", "专业英文名称"], how="left")
        cn_map = pd.Series(
            merged_df.get("专业中文名称", pd.Series([""] * len(merged_df))).values,
            index=pd.MultiIndex.from_frame(merged_df[["学校", "专业英文名称"]]),
        ).to_dict()

        return [cn_map.get((r.get("university"), r.get("major")), "") for r in results]

    def _format_details_for_display(self, details_str):
        return details_str if details_str else "无详细信息"

    def _create_top_similarity_dataframe(
        self, gpa=None, language_score=None, background_university=None, details_df_full=None
    ):
        if not self.top_similarity_results:
            return pd.DataFrame(columns=["目标院校", "目标专业", "录取概率", "专业中文名称"])

        results = self.top_similarity_results

        if results:
            results.sort(
                key=lambda item: (
                    UNIVERSITY_ORDER_MAP.get(item.get("university"), len(UNIVERSITY_SORT_ORDER)),
                    -item.get("probability", 0),
                )
            )

        chinese_names = self._get_chinese_names_in_batch(results, details_df_full=details_df_full)
        return pd.DataFrame(
            {
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
                "专业中文名称": chinese_names,
            }
        )

    def _create_top_cross_major_dataframe(
        self, gpa=None, language_score=None, background_university=None, details_df_full=None
    ):
        if not self.top_cross_major_results:
            return pd.DataFrame(columns=["目标院校", "目标专业", "录取概率", "专业中文名称"])

        results = self.top_cross_major_results

        if results:
            results.sort(
                key=lambda item: (
                    UNIVERSITY_ORDER_MAP.get(item.get("university"), len(UNIVERSITY_SORT_ORDER)),
                    -item.get("probability", 0),
                )
            )

        chinese_names = self._get_chinese_names_in_batch(results, details_df_full=details_df_full)

        return pd.DataFrame(
            {
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
                    self._get_probability_value(result.get("probability", 0)) for result in results
                ],
                "专业中文名称": chinese_names,
            }
        )

    def _create_user_specified_dataframe(
        self,
        gpa=None,
        language_score=None,
        background_university=None,
        max_items=None,
        details_df_full=None,
    ):
        if not self.user_specified_results:
            return pd.DataFrame(columns=["目标院校", "目标专业", "录取概率", "专业中文名称"])

        results = self.user_specified_results

        if results:
            results.sort(
                key=lambda item: (
                    UNIVERSITY_ORDER_MAP.get(item.get("university"), len(UNIVERSITY_SORT_ORDER)),
                    -item.get("probability", 0),
                )
            )

        if isinstance(max_items, int) and max_items > 0:
            results = results[:max_items]

        chinese_names = self._get_chinese_names_in_batch(results, details_df_full=details_df_full)

        return pd.DataFrame(
            {
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
                    self._get_probability_value(result.get("probability", 0)) for result in results
                ],
                "专业中文名称": chinese_names,
            }
        )

    def display(
        self,
        target_universities,
        target_majors,
        gpa=None,
        language_score=None,
        language_type=None,
        background_university=None,
    ):
        session_manager = SessionManager()
        combination_count = session_manager.get("combination_count", 0)
        pool_is_large = isinstance(combination_count, int) and combination_count > 100

        details_df_full = get_school_major_details(None, None, return_df=True)

        df_user_specified = pd.DataFrame()
        has_user_specified = False
        if not pool_is_large:
            df_user_specified = self._create_user_specified_dataframe(
                gpa,
                language_score,
                background_university,
                max_items=None,
                details_df_full=details_df_full,
            )
            has_user_specified = not df_user_specified.empty

        df_similarity = self._create_top_similarity_dataframe(
            gpa, language_score, background_university, details_df_full=details_df_full
        )
        has_similarity = not df_similarity.empty

        df_cross_major = self._create_top_cross_major_dataframe(
            gpa, language_score, background_university, details_df_full=details_df_full
        )
        has_cross_major = not df_cross_major.empty

        if not has_user_specified and not has_similarity and not has_cross_major:
            if pool_is_large or session_manager.get("combination_count", 0) > 0:
                st.info("结果生成中，请稍候…")
            else:
                st.info("没有可显示的预测结果。")
            return

        if pool_is_large:
            if has_similarity or has_cross_major:
                if has_similarity and has_cross_major:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"相似专业录取率 Top {TOP_N_RECOMMENDATIONS}")
                        column_widths = TOP_SIM_RESULT_UI_CONFIG
                        self._display_dataframe(df_similarity, column_widths)

                    with col2:
                        st.success(f"潜力跨专业方向 Top {TOP_N_RECOMMENDATIONS}")
                        column_widths = TOP_CROSS_RESULT_UI_CONFIG
                        self._display_dataframe(df_cross_major, column_widths)
                elif has_similarity:
                    st.success(f"相似专业录取率 Top {TOP_N_RECOMMENDATIONS}")
                    column_widths = TOP_SIM_RESULT_UI_CONFIG
                    self._display_dataframe(df_similarity, column_widths)
                elif has_cross_major:
                    st.success(f"潜力跨专业方向 Top {TOP_N_RECOMMENDATIONS}")
                    column_widths = TOP_CROSS_RESULT_UI_CONFIG
                    self._display_dataframe(df_cross_major, column_widths)
            else:
                st.info("没有可显示的预测结果。")
            return

        if has_user_specified:
            st.success("您指定的目标学校专业预测")
            column_widths_user = TOP_SIM_RESULT_UI_CONFIG
            self._display_dataframe(df_user_specified, column_widths_user)
        elif has_similarity or has_cross_major:
            if has_similarity and has_cross_major:
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"相似专业录取率 Top {TOP_N_RECOMMENDATIONS}")
                    column_widths = TOP_SIM_RESULT_UI_CONFIG
                    self._display_dataframe(df_similarity, column_widths)

                with col2:
                    st.success(f"潜力跨专业方向 Top {TOP_N_RECOMMENDATIONS}")
                    column_widths = TOP_CROSS_RESULT_UI_CONFIG
                    self._display_dataframe(df_cross_major, column_widths)
            elif has_similarity:
                st.success(f"相似专业录取率 Top {TOP_N_RECOMMENDATIONS}")
                column_widths = TOP_SIM_RESULT_UI_CONFIG
                self._display_dataframe(df_similarity, column_widths)
            else:
                st.success(f"潜力跨专业方向 Top {TOP_N_RECOMMENDATIONS}")
                column_widths = TOP_CROSS_RESULT_UI_CONFIG
                self._display_dataframe(df_cross_major, column_widths)
        else:
            st.info("没有可显示的预测结果。")
        return
