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
        if "background_summary_precomputed" in display_df.columns:
            display_df["background_summary"] = display_df["background_summary_precomputed"]
        else:
            background_cols_map = {
                "工作数量": "工作",
                "发表数量": "论文",
                "科研数量": "科研",
                "实习数量": "实习",
                "活动数量": "活动",
                "获奖数量": "获奖",
            }

            summary_parts = []
            for col, prefix in background_cols_map.items():
                if col in display_df.columns:
                    values = pd.to_numeric(display_df[col], errors="coerce")
                    valid_values = values.where((values > 0) & values.notna(), pd.NA)
                    str_values = valid_values.apply(
                        lambda x: f"{prefix}{int(x)}" if pd.notna(x) else ""
                    )
                    summary_parts.append(str_values)

            if summary_parts:
                result = summary_parts[0]
                for part in summary_parts[1:]:
                    result = result + part
                display_df["background_summary"] = result
            else:
                display_df["background_summary"] = ""

    if selected_chinese_categories == "硕士":
        if "uni_classification_summary_precomputed" in display_df.columns:
            display_df["uni_classification_summary"] = display_df[
                "uni_classification_summary_precomputed"
            ]
        else:
            parts = []
            if "国本院校分类" in display_df.columns:
                domestic = display_df["国本院校分类"].astype(str)
                domestic = domestic.where(
                    ~domestic.isin(config.INVALID_VALUES) & domestic.notna(), ""
                )
                parts.append(domestic)

            if "海本QS排名区间" in display_df.columns:
                overseas = display_df["海本QS排名区间"].astype(str)
                overseas = overseas.where(
                    ~overseas.isin(config.INVALID_VALUES) & overseas.notna(), ""
                )
                parts.append(overseas)

            if len(parts) == 2:
                result = parts[0] + "/" + parts[1]
                result = result.str.strip("/").replace("/", "")
                display_df["uni_classification_summary"] = result
            elif len(parts) == 1:
                display_df["uni_classification_summary"] = parts[0]
            else:
                display_df["uni_classification_summary"] = ""

        if "gre_gmat_score_precomputed" in display_df.columns:
            display_df["gre_gmat_score"] = display_df["gre_gmat_score_precomputed"]
        else:
            parts = []
            if "GRE分数" in display_df.columns:
                gre = pd.to_numeric(display_df["GRE分数"], errors="coerce")
                gre_str = gre.where((gre > 0) & gre.notna(), pd.NA).apply(
                    lambda x: str(int(x)) if pd.notna(x) else ""
                )
                parts.append(gre_str)

            if "GMAT分数" in display_df.columns:
                gmat = pd.to_numeric(display_df["GMAT分数"], errors="coerce")
                gmat_str = gmat.where((gmat > 0) & gmat.notna(), pd.NA).apply(
                    lambda x: str(int(x)) if pd.notna(x) else ""
                )
                parts.append(gmat_str)

            if len(parts) == 2:
                result = parts[0] + "/" + parts[1]
                result = result.str.strip("/").replace("/", "")
                display_df["gre_gmat_score"] = result
            elif len(parts) == 1:
                display_df["gre_gmat_score"] = parts[0]
            else:
                display_df["gre_gmat_score"] = ""

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

    gpa_cols_in_df = [col for col in gpa_cols_to_format if col in final_display_df.columns]
    for col in gpa_cols_in_df:
        numeric_series = pd.to_numeric(final_display_df[col], errors="coerce")
        final_display_df[col] = numeric_series.apply(
            lambda x: "" if pd.isna(x) else str(int(x)) if x == int(x) else f"{x:g}"
        )

    other_cols = [col for col in final_display_df.columns if col not in gpa_cols_in_df]
    for col in other_cols:
        final_display_df[col] = (
            final_display_df[col]
            .astype(str)
            .replace({"None": "", "nan": "", "NaN": "", "None": ""})
        )

    st.markdown('<div class="case-library-content">', unsafe_allow_html=True)
    st.dataframe(final_display_df, hide_index=True, height=1060)
    st.markdown("</div>", unsafe_allow_html=True)
