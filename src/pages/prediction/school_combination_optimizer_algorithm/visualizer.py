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
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class RecommendationVisualizer:
    RISK_LEVELS = [
        (0.15, "低", "#4CAF50"),
        (0.40, "中等", "#FFC107"),
        (0.70, "高", "#FF9800"),
        (1.0, "非常高", "#F44336"),
    ]

    CONFIDENCE_LEVELS = [
        (0.40, "低", "#F44336"),
        (0.70, "中等", "#FFC107"),
        (0.85, "高", "#8BC34A"),
        (1.0, "非常高", "#4CAF50"),
    ]

    DIFFICULTY_CONFIG = {
        "safety": ("保底", "#4CAF50"),
        "target_lower": ("目标", "#2196F3"),
        "default": ("冲刺", "#FF9800"),
    }

    def __init__(self, adaptive_thresholds=None):
        self.adaptive_thresholds = adaptive_thresholds or self._get_default_thresholds()
        self._inject_css()

    def _get_default_thresholds(self):
        from src.pages.prediction.school_combination_optimizer_algorithm.config import (
            SCHOOL_CATEGORY_THRESHOLDS,
        )

        return SCHOOL_CATEGORY_THRESHOLDS

    def _inject_css(self):
        if not st.session_state.get("_optimizer_visual_css_injected"):
            css = """
            <style>
            div[data-testid="stMetricValue"] > div { font-size: 0.9rem !important; }
            div[data-testid="stMetricLabel"] p { font-weight: bold !important; }
            </style>
            """
            st.markdown(css, unsafe_allow_html=True)
            st.session_state["_optimizer_visual_css_injected"] = True

    def _get_probability_value(self, probability):
        return 0.0 if probability is None else float(probability)

    def _create_gauge_html(self, value, title, levels, is_confidence=False):
        if value is None:
            return ""

        level_name, level_color = "", ""
        for threshold, name, color in levels:
            if value <= threshold:
                level_name, level_color = name, color
                break

        pos = value * 100
        segments = []
        prev_threshold = 0

        for i, (threshold, _, color) in enumerate(levels):
            segment_width = (threshold - prev_threshold) * 100
            border_radius = (
                "6px 0 0 6px" if i == 0 else "0 6px 6px 0" if i == len(levels) - 1 else ""
            )
            segments.append(
                f'<div style="flex-basis: {segment_width}%; background-color: {color}; border-radius: {border_radius};"></div>'
            )
            prev_threshold = threshold

        level_text = "信心等级" if is_confidence else "风险等级"

        return f"""
        <div style="font-family: sans-serif; font-size: 14px;">
            <b>{title}</b>
            <div style="position: relative; width: 100%; margin: 8px 0;">
                <div style="display: flex; width: 100%; height: 12px; border-radius: 6px;">
                    {"".join(segments)}
                </div>
                <div style="position: absolute; top: -6px; left: {min(pos, 98)}%; transform: translateX(-50%); 
                     width: 0; height: 0; border-left: 8px solid transparent; 
                     border-right: 8px solid transparent; border-top: 12px solid #000; 
                     filter: drop-shadow(0 1px 2px rgba(0,0,0,0.3));"></div>
            </div>
            <div>{level_text}: <strong style="color:{level_color};">{level_name}</strong></div>
        </div>"""

    def _get_school_details_data(self):
        cache_key_df = "_optimizer_details_df_full"
        cache_key_map = "_optimizer_details_map"

        details_df_full = st.session_state.get(cache_key_df)
        details_map = st.session_state.get(cache_key_map)

        if details_df_full is not None and details_map is not None:
            return details_map

        base_df = get_school_major_details(None, None, return_df=True)
        if base_df is None or base_df.empty:
            return {}

        base_df = base_df.drop_duplicates(subset=["学校", "专业英文名称"])

        def _norm_text(x: str) -> str:
            return "".join(str(x).split()).lower()

        base_df["_key_university"] = base_df["学校"].astype(str).map(_norm_text)
        base_df["_key_major_en"] = base_df["专业英文名称"].astype(str).map(_norm_text)
        base_df["formatted_details"] = base_df.apply(format_school_major_details_from_row, axis=1)

        details_map = pd.Series(
            base_df.formatted_details.values,
            index=pd.MultiIndex.from_frame(base_df[["_key_university", "_key_major_en"]]),
        ).to_dict()

        st.session_state[cache_key_df] = base_df
        st.session_state[cache_key_map] = details_map

        return details_map

    def _get_school_difficulty(self, probability):
        if probability >= self.adaptive_thresholds["safety"]:
            return self.DIFFICULTY_CONFIG["safety"]
        elif probability >= self.adaptive_thresholds["target_lower"]:
            return self.DIFFICULTY_CONFIG["target_lower"]
        else:
            return self.DIFFICULTY_CONFIG["default"]

    def _prepare_schools_data(self, schools):
        details_map = self._get_school_details_data()

        def _norm_text(v: str) -> str:
            return "".join(str(v).split()).lower()

        schools_data = []
        for school in schools:
            university = school.get("university", "")
            major = school.get("major", "")
            probability = self._get_probability_value(school.get("probability", 0.0))

            u_key = _norm_text(university)
            m_key = _norm_text(major)
            details = details_map.get((u_key, m_key), "无详细信息")

            difficulty_name, difficulty_color = self._get_school_difficulty(probability)

            major_display = f"{major} (New!)" if school.get("is_new_major", False) else major

            schools_data.append(
                {
                    "logo": get_logo_path(university),
                    "university": university,
                    "major": major_display,
                    "probability": probability,
                    "probability_value": probability,
                    "type": school.get("type", ""),
                    "details": details,
                    "difficulty": difficulty_name,
                    "difficulty_color": difficulty_color,
                }
            )

        return schools_data

    def _create_schools_dataframe(self, schools_data):
        df_data = {
            "logo": [school["logo"] for school in schools_data],
            "目标院校": [school["university"] for school in schools_data],
            "目标专业": [school["major"] for school in schools_data],
            "录取概率": [school["probability"] for school in schools_data],
            "概率数值": [school["probability_value"] for school in schools_data],
            "专业类型": [school["type"] for school in schools_data],
            "专业详情": [school["details"] for school in schools_data],
            "申请难度": [school["difficulty"] for school in schools_data],
            "难度颜色": [school["difficulty_color"] for school in schools_data],
        }

        df = pd.DataFrame(df_data)

        difficulty_order = {"保底": 1, "目标": 2, "冲刺": 3}
        df["排序权重"] = df["申请难度"].map(difficulty_order)
        df = df.sort_values(["排序权重", "概率数值"], ascending=[True, False])

        return df[["logo", "目标院校", "目标专业", "录取概率", "申请难度", "专业详情"]]

    def _style_dataframe(self, df):
        def color_by_difficulty(row):
            color_map = {"保底": "#4CAF50", "目标": "#2196F3", "冲刺": "#FF9800"}
            color = color_map.get(row["申请难度"], "#000000")
            return [f"color: {color}"] * len(row)

        def style_new_major(val):
            if isinstance(val, str) and "(New!)" in val:
                return "color: #FF4B4B; font-weight: bold;"
            return ""

        return df.style.apply(color_by_difficulty, axis=1).map(style_new_major, subset=["目标专业"])

    def _render_metrics_section(self, metrics):
        col2, col1, col3 = st.columns(3)

        sim_rejection_prob = metrics.get("simulated_rejection_probability")
        sim_admission_prob = metrics.get("simulated_admission_probability")
        diversity = metrics.get("diversity", "")

        with col2:
            if sim_admission_prob is not None:
                confidence_html = self._create_gauge_html(
                    sim_admission_prob, "录取信心指数", self.CONFIDENCE_LEVELS, True
                )
                st.markdown(confidence_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    "<b style='font-size: 14px;'>录取信心指数</b>",
                    unsafe_allow_html=True,
                )
                st.progress(0, text="无法计算")

        with col1:
            if sim_rejection_prob is not None:
                risk_html = self._create_gauge_html(
                    sim_rejection_prob, "组合风险评估", self.RISK_LEVELS
                )
                st.markdown(risk_html, unsafe_allow_html=True)
            else:
                st.markdown("<b>组合风险评估</b>", unsafe_allow_html=True)
                st.write("无法计算")

        with col3:
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

    def _update_session_state(self, recommendations):
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

    def visualize(self, recommendations: list[dict[str, Any]]) -> None:
        if not recommendations:
            st.warning("没有找到符合条件的推荐方案")
            return

        self._update_session_state(recommendations)

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

                self._render_metrics_section(metrics)

                schools_data = self._prepare_schools_data(schools)
                df = self._create_schools_dataframe(schools_data)
                styled_df = self._style_dataframe(df)

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

                strategy_type = recommendation.get("type", f"strategy_{i}")
                st.data_editor(
                    styled_df,
                    hide_index=True,
                    disabled=True,
                    column_config=column_config,
                    key=f"school_data_editor_{strategy_type}_{i}",
                )


def visualize_recommendations(
    recommendations: list[dict[str, Any]], adaptive_thresholds: dict[str, float] = None
) -> None:
    visualizer = RecommendationVisualizer(adaptive_thresholds)
    visualizer.visualize(recommendations)
