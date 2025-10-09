from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.admission_probability_calculator_components.school_logo_loader import (
    get_logo_path,
)
from src.pages.prediction.prediction_utils import (
    format_school_major_details_from_row,
    get_school_major_details,
)


def _get_probability_value(probability):
    if probability is None:
        return 0.0
    return float(probability)


def visualize_recommendations(
    recommendations: list[dict[str, Any]], adaptive_thresholds: dict[str, float] = None
) -> None:
    font_size_css = """
    <style>
    div[data-testid="stMetricValue"] > div {
        font-size: 0.9rem !important; 
    }
    div[data-testid="stMetricLabel"] p {
        font-weight: bold !important;
    }
    </style>
    """

    if not st.session_state.get("_optimizer_visual_css_injected"):
        st.markdown(font_size_css, unsafe_allow_html=True)
        st.session_state["_optimizer_visual_css_injected"] = True

    if not recommendations:
        st.warning("没有找到符合条件的推荐方案")
        return

    if recommendations and "schools" in recommendations[0] and recommendations[0]["schools"]:
        first_recommendation = recommendations[0]
        st.session_state.recommendation_applied = True
        st.session_state.selected_school_results = first_recommendation["schools"]
        st.session_state.selected_school_probabilities = [
            school.get("probability", 0.0) for school in first_recommendation["schools"]
        ]
    else:
        st.session_state.recommendation_applied = False
        st.session_state.selected_school_results = []
        st.session_state.selected_school_probabilities = []

    tab_titles = [
        recommendation.get("type", f"推荐方案 {i + 1}")
        for i, recommendation in enumerate(recommendations)
    ]
    tabs = st.tabs(tab_titles)

    for i, (tab, recommendation) in enumerate(zip(tabs, recommendations, strict=False)):
        with tab:
            schools = recommendation.get("schools", [])
            metrics = recommendation.get("metrics", {})

            if not schools:
                st.warning("此方案无学校推荐")
                continue

            def _create_risk_gauge_html(probability_value):
                if probability_value is None:
                    return ""
                if probability_value <= 0.15:
                    level, color = "低", "#4CAF50"
                elif probability_value <= 0.40:
                    level, color = "中等", "#FFC107"
                elif probability_value <= 0.70:
                    level, color = "高", "#FF9800"
                else:
                    level, color = "非常高", "#F44336"

                pos = probability_value * 100
                return f"""
                <div style="font-family: sans-serif; font-size: 14px;">
                    <b>组合风险评估</b>
                    <div style="position: relative; width: 100%; margin: 8px 0;">
                        <div style="display: flex; width: 100%; height: 12px; border-radius: 6px;">
                            <div style="flex-basis: 15%; background-color: #4CAF50; border-radius: 6px 0 0 6px;"></div>
                            <div style="flex-basis: 25%; background-color: #FFC107;"></div>
                            <div style="flex-basis: 30%; background-color: #FF9800;"></div>
                            <div style="flex-basis: 30%; background-color: #F44336; border-radius: 0 6px 6px 0;"></div>
                        </div>
                        <div style="position: absolute; top: -4px; left: {min(pos, 98)}%; transform: translateX(-50%); width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 8px solid #333;"></div>
                    </div>
                    <div>风险等级: <strong style="color:{color};">{level}</strong></div>
                </div>"""

            def _create_confidence_gauge_html(probability_value):
                if probability_value is None:
                    return ""
                if probability_value >= 0.85:
                    level, color = "非常高", "#4CAF50"
                elif probability_value >= 0.70:
                    level, color = "高", "#8BC34A"
                elif probability_value >= 0.40:
                    level, color = "中等", "#FFC107"
                else:
                    level, color = "低", "#F44336"

                pos = probability_value * 100
                return f"""
                <div style="font-family: sans-serif; font-size: 14px;">
                    <b>录取信心指数</b>
                    <div style="position: relative; width: 100%; margin: 8px 0;">
                        <div style="display: flex; width: 100%; height: 12px; border-radius: 6px;">
                            <div style="flex-basis: 40%; background-color: #F44336; border-radius: 6px 0 0 6px;"></div>
                            <div style="flex-basis: 30%; background-color: #FFC107;"></div>
                            <div style="flex-basis: 15%; background-color: #8BC34A;"></div>
                            <div style="flex-basis: 15%; background-color: #4CAF50; border-radius: 0 6px 6px 0;"></div>
                        </div>
                        <div style="position: absolute; top: -4px; left: {min(pos, 98)}%; transform: translateX(-50%); width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 8px solid #333;"></div>
                    </div>
                    <div>信心等级: <strong style="color:{color};">{level}</strong></div>
                </div>"""

            col2, col1, col3 = st.columns(3)

            current_thresholds = adaptive_thresholds
            if current_thresholds is None:
                from .optimizer_config import SCHOOL_CATEGORY_THRESHOLDS

                current_thresholds = SCHOOL_CATEGORY_THRESHOLDS

            sim_rejection_prob = metrics.get("simulated_rejection_probability", None)
            sim_admission_prob = metrics.get("simulated_admission_probability", None)

            with col2:
                if sim_admission_prob is not None:
                    confidence_gauge_html = _create_confidence_gauge_html(sim_admission_prob)
                    st.markdown(confidence_gauge_html, unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<b style='font-size: 14px;'>录取信心指数</b>", unsafe_allow_html=True
                    )
                    st.progress(0, text="无法计算")

            with col1:
                if sim_rejection_prob is not None:
                    gauge_html = _create_risk_gauge_html(sim_rejection_prob)
                    st.markdown(gauge_html, unsafe_allow_html=True)
                else:
                    st.markdown("<b>组合风险评估</b>", unsafe_allow_html=True)
                    st.write("无法计算")

            with col3:
                diversity = metrics.get("diversity", "")
                st.markdown(
                    f"""
                <div style="font-family: sans-serif;">
                    <p style="font-size: 14px; font-weight: bold; color: black; margin-bottom: 0.2rem;">学校多样性</p>
                    <div style="font-family: sans-serif; font-size: 14px;">{diversity}所不同大学</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

            try:
                cache_key_df = "_optimizer_details_df_full"
                cache_key_map = "_optimizer_details_map"
                details_df_full = st.session_state.get(cache_key_df)
                details_map = st.session_state.get(cache_key_map)

                if details_df_full is None or details_map is None:
                    base_df = get_school_major_details(None, None, return_df=True)
                    if base_df is not None and not base_df.empty:
                        base_df = base_df.drop_duplicates(subset=["学校", "专业英文名称"])

                        def _norm_text(x: str) -> str:
                            try:
                                return "".join(str(x).split()).lower()
                            except Exception:
                                return str(x).strip().lower()

                        base_df["_key_university"] = base_df["学校"].astype(str).map(_norm_text)
                        base_df["_key_major_en"] = (
                            base_df["专业英文名称"].astype(str).map(_norm_text)
                        )

                        base_df["formatted_details"] = base_df.apply(
                            format_school_major_details_from_row, axis=1
                        )
                        details_map = pd.Series(
                            base_df.formatted_details.values,
                            index=pd.MultiIndex.from_frame(
                                base_df[["_key_university", "_key_major_en"]]
                            ),
                        ).to_dict()
                        st.session_state[cache_key_df] = base_df
                        st.session_state[cache_key_map] = details_map
                    else:
                        details_map = {}

                def _norm_s(v: str) -> str:
                    try:
                        return "".join(str(v).split()).lower()
                    except Exception:
                        return str(v).strip().lower()

                details_list = []
                for s in schools:
                    u = _norm_s(s.get("university", ""))
                    m = _norm_s(s.get("major", ""))
                    details_list.append(details_map.get((u, m), "无详细信息"))
            except Exception:
                details_list = ["无详细信息" for _ in schools]

            df_data = {
                "logo": [get_logo_path(school.get("university", "")) for school in schools],
                "目标院校": [school.get("university", "") for school in schools],
                "目标专业": [
                    (
                        f"{school.get('major', '')} (New!)"
                        if school.get("is_new_major", False)
                        else school.get("major", "")
                    )
                    for school in schools
                ],
                "录取概率": [
                    _get_probability_value(school.get("probability")) for school in schools
                ],
                "概率数值": [school.get("probability", 0.0) for school in schools],
                "专业类型": [school.get("type", "") for school in schools],
                "专业详情": details_list,
                "申请难度": [
                    (
                        "保底"
                        if school.get("probability", 0.0) >= current_thresholds["safety"]
                        else (
                            "目标"
                            if school.get("probability", 0.0) >= current_thresholds["target_lower"]
                            else "冲刺"
                        )
                    )
                    for school in schools
                ],
            }

            df = pd.DataFrame(df_data)

            difficulty_order = {"保底": 1, "目标": 2, "冲刺": 3}
            df["排序权重"] = df["申请难度"].map(difficulty_order)
            df = df.sort_values(["排序权重", "概率数值"], ascending=[True, False])

            df = df[["logo", "目标院校", "目标专业", "录取概率", "申请难度", "专业详情"]]

            column_config = {
                "logo": st.column_config.ImageColumn("logo"),
                "目标院校": st.column_config.TextColumn("目标院校"),
                "目标专业": st.column_config.TextColumn("目标专业"),
                "录取概率": st.column_config.ProgressColumn(
                    "录取概率", min_value=0, max_value=1, format=" "
                ),
                "专业详情": st.column_config.TextColumn(
                    "专业详情", width="small", help="双击查看详细信息"
                ),
                "申请难度": st.column_config.TextColumn("申请难度"),
            }

            def color_difficulty(row):
                if row["申请难度"] == "保底":
                    return ["color: #4CAF50"] * len(row)
                elif row["申请难度"] == "目标":
                    return ["color: #2196F3"] * len(row)
                else:
                    return ["color: #FF9800"] * len(row)

            def style_new_major(val):
                if isinstance(val, str) and "(New!)" in val:
                    return "color: #FF4B4B; font-weight: bold;"
                return ""

            styled_df = df.style.apply(color_difficulty, axis=1).map(
                style_new_major, subset=["目标专业"]
            )

            st.data_editor(styled_df, hide_index=True, disabled=True, column_config=column_config)
