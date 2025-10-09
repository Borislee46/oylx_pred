import pandas as pd
import streamlit as st

from src.pages.case_lib import config, filters


def get_language_score_filter(df, selected_chinese_categories):
    language_type = st.radio("语言成绩类型", ["不限", "雅思", "托福"], horizontal=True)

    ielts_col = (
        config.IELTS_COL if selected_chinese_categories == "本科" else config.IELTS_SCORE_COL
    )
    toefl_col = (
        config.TOEFL_COL if selected_chinese_categories == "本科" else config.TOEFL_SCORE_COL
    )

    if language_type == "雅思" and ielts_col in df.columns:
        score_range = st.slider(
            "雅思分数范围", min_value=0.0, max_value=9.0, value=(0.0, 9.0), step=0.5
        )
        return ielts_col, score_range

    elif language_type == "托福" and toefl_col in df.columns:
        score_range = st.slider("托福分数范围", min_value=0, max_value=120, value=(0, 120), step=1)
        return toefl_col, score_range

    return None, None


def display_filters(df, filter_options, selected_chinese_categories):
    selections = {}
    with st.expander("筛选器", expanded=True):
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            selections["years"] = st.multiselect(
                "学年", filter_options.get("years", []), placeholder="选择学年"
            )

        with col2:
            selections["countries"] = st.multiselect(
                "申请国家/地区",
                filter_options.get("countries", []),
                placeholder="选择申请国家/地区",
            )

        target_unis_options = filters.get_target_unis_options(df, tuple(selections["countries"]))

        with col3:
            if selected_chinese_categories == "本科":
                selections["体系"] = st.multiselect(
                    "申请体系", filter_options.get("体系", []), placeholder="选择申请体系"
                )
            else:
                selections["background_majors"] = st.multiselect(
                    "就读专业",
                    filter_options.get("background_majors", []),
                    placeholder="选择就读专业",
                )

        if "体系" not in selections:
            selections["体系"] = []
        if "background_majors" not in selections:
            selections["background_majors"] = []

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            selections["background_unis"] = st.multiselect(
                "就读院校", filter_options.get("background_unis", []), placeholder="选择就读院校"
            )

            if selected_chinese_categories != "本科":
                selections["school_levels"] = st.multiselect(
                    "学校分层", filter_options.get("school_levels", []), placeholder="选择学校分层"
                )
            else:
                selections["school_levels"] = []

        with col2:
            selections["target_unis"] = st.multiselect(
                "申请院校", target_unis_options, placeholder="选择申请院校"
            )

        with col3:
            st.empty()

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            selections["target_majors"] = st.multiselect(
                "申请专业", filter_options.get("target_majors", []), placeholder="选择申请专业"
            )

        with col2:
            selections["admission_statuses"] = st.multiselect(
                "录取状态", filter_options.get("admission_statuses", []), placeholder="选择录取状态"
            )

        with col3:
            selections["language_filter"] = get_language_score_filter(
                df, selected_chinese_categories
            )

    return selections


def display_results(df, selected_chinese_categories):
    if df.empty:
        st.info("没有找到符合条件的案例。")
        return

    max_display_count = 50

    display_df = df.head(max_display_count).copy()

    if selected_chinese_categories in ["硕士", "博士"]:
        background_cols_map = {
            "工作数量": "工作",
            "发表数量": "论文",
            "科研数量": "科研",
            "实习数量": "实习",
            "活动数量": "活动",
            "获奖数量": "获奖",
        }

        def create_summary(row):
            summary_parts = []
            for col, prefix in background_cols_map.items():
                if col in row.index:
                    value = pd.to_numeric(row[col], errors="coerce")
                    if pd.notna(value) and value > 0:
                        summary_parts.append(f"{prefix}{int(value)}")
            return "".join(summary_parts)

        display_df["background_summary"] = display_df.apply(create_summary, axis=1)

    if selected_chinese_categories == "硕士":

        def create_uni_summary(row):
            parts = []
            domestic = row.get("国本院校分类")
            overseas = row.get("海本QS排名区间")
            if domestic and str(domestic) not in config.INVALID_VALUES and not pd.isna(domestic):
                parts.append(str(domestic))
            if overseas and str(overseas) not in config.INVALID_VALUES and not pd.isna(overseas):
                parts.append(str(overseas))
            return "/".join(parts)

        display_df["uni_classification_summary"] = display_df.apply(create_uni_summary, axis=1)

        def create_gre_gmat_summary(row):
            parts = []
            gre_val = pd.to_numeric(row.get("GRE分数"), errors="coerce")
            gmat_val = pd.to_numeric(row.get("GMAT分数"), errors="coerce")

            if pd.notna(gre_val) and gre_val > 0:
                parts.append(str(int(gre_val)))
            if pd.notna(gmat_val) and gmat_val > 0:
                parts.append(str(int(gmat_val)))

            return "/".join(parts)

        display_df["gre_gmat_score"] = display_df.apply(create_gre_gmat_summary, axis=1)

    display_cols = config.DISPLAY_COLS_CONFIG.get(selected_chinese_categories, [])

    existing_display_cols = [
        original_col
        for original_col, display_name in display_cols
        if original_col in display_df.columns
    ]

    if not existing_display_cols:
        st.warning("没有可供展示的数据列。")
        return

    column_rename_map = {original_col: display_name for original_col, display_name in display_cols}

    final_display_df = display_df[existing_display_cols].copy()
    final_display_df = final_display_df.rename(columns=column_rename_map)

    gpa_cols_to_format = ["GPA", "GPA分制", "GPA（百分制）"]

    for col in final_display_df.columns:
        if col in gpa_cols_to_format:
            numeric_series = pd.to_numeric(final_display_df[col], errors="coerce")
            final_display_df[col] = numeric_series.apply(
                lambda x: "" if pd.isna(x) else str(int(x)) if x == int(x) else f"{x:g}"
            )
        else:
            series = final_display_df[col]
            series = series.astype(object)
            series = series.where(series.notna(), "")
            series = series.astype(str).replace({"None": "", "nan": "", "NaN": ""})
            final_display_df[col] = series

    st.markdown('<div class="case-library-content">', unsafe_allow_html=True)
    st.dataframe(final_display_df, hide_index=True, height=1060)
    st.markdown("</div>", unsafe_allow_html=True)
