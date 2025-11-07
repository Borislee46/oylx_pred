import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

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

        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            search_clicked = st.button("确定查询", type="primary", use_container_width=True)

    return selections, search_clicked


def _prepare_background_summary(display_df):
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
            numeric_series = pd.to_numeric(display_df[col], errors="coerce")
            mask = (numeric_series.notna()) & (numeric_series > 0)
            summary_parts.append(
                pd.Series("", index=display_df.index).where(
                    ~mask, prefix + numeric_series[mask].astype(int).astype(str)
                )
            )

    if summary_parts:
        return pd.concat(summary_parts, axis=1).fillna("").apply("".join, axis=1)
    return pd.Series("", index=display_df.index)


def _prepare_uni_classification_summary(display_df):
    parts_list = []

    if "国本院校分类" in display_df.columns:
        domestic_series = display_df["国本院校分类"].astype(str)
        valid_domestic = (
            domestic_series.notna()
            & (domestic_series != "")
            & (~domestic_series.isin(config.INVALID_VALUES))
        )
        parts_list.append(domestic_series.where(valid_domestic, ""))

    if "海本QS排名区间" in display_df.columns:
        overseas_series = display_df["海本QS排名区间"].astype(str)
        valid_overseas = (
            overseas_series.notna()
            & (overseas_series != "")
            & (~overseas_series.isin(config.INVALID_VALUES))
        )
        parts_list.append(overseas_series.where(valid_overseas, ""))

    if not parts_list:
        return pd.Series("", index=display_df.index)

    result = parts_list[0]
    for part in parts_list[1:]:
        result = result.astype(str) + "/" + part.astype(str)

    return result.str.replace("^/$", "", regex=True).str.replace("^/|/$", "", regex=True)


def _prepare_gre_gmat_score(display_df):
    gre_parts = pd.Series("", index=display_df.index, dtype=str)
    gmat_parts = pd.Series("", index=display_df.index, dtype=str)

    if "GRE分数" in display_df.columns:
        gre_series = pd.to_numeric(display_df["GRE分数"], errors="coerce")
        gre_mask = gre_series.notna() & (gre_series > 0)
        gre_parts.loc[gre_mask] = gre_series.loc[gre_mask].astype(int).astype(str)

    if "GMAT分数" in display_df.columns:
        gmat_series = pd.to_numeric(display_df["GMAT分数"], errors="coerce")
        gmat_mask = gmat_series.notna() & (gmat_series > 0)
        gmat_parts.loc[gmat_mask] = gmat_series.loc[gmat_mask].astype(int).astype(str)

    result = gre_parts + "/" + gmat_parts
    return result.str.replace("^/$", "", regex=True).str.replace("^/|/$", "", regex=True)


def _format_gpa_column(series):
    numeric_series = pd.to_numeric(series, errors="coerce")
    result = pd.Series("", index=series.index, dtype=str)

    notna_mask = numeric_series.notna()
    if notna_mask.any():
        notna_values = numeric_series[notna_mask]
        int_mask = (notna_values == notna_values.astype(int)) & (notna_values >= 0)
        result.loc[notna_mask & int_mask] = notna_values[int_mask].astype(int).astype(str)

        float_mask = notna_mask & ~int_mask
        if float_mask.any():
            result.loc[float_mask] = numeric_series[float_mask].apply(
                lambda x: f"{x:g}" if pd.notna(x) else ""
            )

    return result


def _format_display_dataframe(display_df, selected_chinese_categories):
    if selected_chinese_categories in ["硕士", "博士"]:
        display_df["background_summary"] = _prepare_background_summary(display_df)

    if selected_chinese_categories == "硕士":
        display_df["uni_classification_summary"] = _prepare_uni_classification_summary(display_df)
        display_df["gre_gmat_score"] = _prepare_gre_gmat_score(display_df)

    display_cols = config.DISPLAY_COLS_CONFIG.get(selected_chinese_categories, [])
    existing_display_cols = [
        original_col
        for original_col, display_name in display_cols
        if original_col in display_df.columns
    ]

    if not existing_display_cols:
        return None, None

    column_rename_map = {
        original_col: display_name
        for original_col, display_name in display_cols
        if original_col in existing_display_cols
    }

    final_display_df = display_df[existing_display_cols].rename(columns=column_rename_map).copy()

    for col in final_display_df.columns:
        if col in config.GPA_COLS_TO_FORMAT:
            final_display_df[col] = _format_gpa_column(final_display_df[col])
        else:
            series = final_display_df[col].astype(object)
            series = series.where(series.notna(), "")
            series = series.astype(str).replace({"None": "", "nan": "", "NaN": ""})
            final_display_df[col] = series

    return final_display_df, column_rename_map


def _render_data_grid(final_display_df):
    gb = GridOptionsBuilder.from_dataframe(final_display_df)

    if "录取状态" in final_display_df.columns:
        admission_status_style = JsCode(
            """
            function(params) {
                if (params.value === '已录取') {
                    return {
                        'backgroundColor': '#d4edda',
                        'color': '#155724'
                    };
                } else if (params.value === '拒录') {
                    return {
                        'backgroundColor': '#f8d7da',
                        'color': '#721c24'
                    };
                }
                return null;
            }
            """
        )
        gb.configure_column("录取状态", cellStyle=admission_status_style)

    gb.configure_default_column(editable=False, resizable=True)
    gridOptions = gb.build()

    st.markdown('<div class="case-library-content">', unsafe_allow_html=True)
    AgGrid(
        final_display_df,
        gridOptions=gridOptions,
        height=config.GRID_HEIGHT,
        theme="streamlit",
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def display_results(df, selected_chinese_categories):
    if df.empty:
        st.info("没有找到符合条件的案例。")
        return

    total_count = len(df)
    session_key = f"case_lib_display_count_{selected_chinese_categories}"

    if session_key not in st.session_state:
        st.session_state[session_key] = config.INITIAL_LOAD_COUNT

    current_display_count = min(
        st.session_state[session_key], config.MAX_DISPLAY_COUNT, total_count
    )

    info_msg = f"共找到 {total_count} 条符合条件的案例，当前展示 {current_display_count} 条"
    if total_count > config.MAX_DISPLAY_COUNT:
        info_msg += f"（最多展示 {config.MAX_DISPLAY_COUNT} 条）"
    st.info(info_msg)

    display_df = df.head(current_display_count).copy()
    final_display_df, column_rename_map = _format_display_dataframe(
        display_df, selected_chinese_categories
    )

    if final_display_df is None:
        st.warning("没有可供展示的数据列。")
        return

    _render_data_grid(final_display_df)

    if current_display_count < min(total_count, config.MAX_DISPLAY_COUNT):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("加载更多", type="primary", use_container_width=True):
                st.session_state[session_key] = min(
                    st.session_state[session_key] + config.LOAD_MORE_COUNT,
                    config.MAX_DISPLAY_COUNT,
                    total_count,
                )
                st.rerun()
