import streamlit as st
from streamlit_elements import dashboard, elements, mui, nivo, sync

from src.pages.prediction.result_modifier.config import (
    GPA_MINIMUM,
    GPA_PENALTY_MAX_COEFFICIENT,
    GPA_PENALTY_QUADRATIC_COEFFICIENT,
    GPA_PENALTY_SEVERE_THRESHOLD,
    LANGUAGE_MINIMUM,
    LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER,
    LANGUAGE_PENALTY_LEVEL_1_THRESHOLD,
    LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER,
    LANGUAGE_PENALTY_LEVEL_2_THRESHOLD,
    LANGUAGE_PENALTY_LEVEL_3_THRESHOLD,
    LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER,
    LANGUAGE_PENALTY_SEVERE_THRESHOLD,
)
from src.utils.school_level_service import SCHOOL_LEVEL_PRIORITY, get_school_level_service

MAX_GPA = 4.0
MAX_ACADEMIC_ITEMS = 3
MAX_PRACTICAL_ITEMS = 7

DEFAULT_GPA_MEAN = 3.0
DEFAULT_GPA_STD = 0.5
DEFAULT_LANGUAGE_MEAN = 0.75
DEFAULT_LANGUAGE_STD = 0.15


def _calculate_gpa_score(
    gpa: float,
    gpa_mean: float = DEFAULT_GPA_MEAN,
    gpa_std: float = DEFAULT_GPA_STD,
) -> float:
    if gpa < GPA_MINIMUM:
        penalty = GPA_PENALTY_SEVERE_THRESHOLD
        return max(0, min(100, (1 - penalty) * 100))

    base_score = (gpa / MAX_GPA) * 100

    if gpa >= gpa_mean:
        return min(100, base_score)

    gpa_gap = (gpa_mean - gpa) / gpa_std if gpa_std > 0 else 0
    penalty = min(
        GPA_PENALTY_MAX_COEFFICIENT,
        GPA_PENALTY_QUADRATIC_COEFFICIENT * gpa_gap**2,
    )

    adjusted_score = base_score * (1 - penalty)
    return max(0, min(100, adjusted_score))


def _calculate_language_score(
    language_score: float,
    language_mean: float = DEFAULT_LANGUAGE_MEAN,
    language_std: float = DEFAULT_LANGUAGE_STD,
) -> float:
    if language_score < LANGUAGE_MINIMUM:
        penalty = LANGUAGE_PENALTY_SEVERE_THRESHOLD
        return max(0, min(100, (1 - penalty) * 100))

    base_score = language_score * 100

    language_excellent_line = language_mean + 0.5 * language_std
    language_pass_line = language_mean - LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER * language_std

    if language_score >= language_excellent_line:
        return min(100, base_score)
    elif language_score >= language_pass_line:
        gap_ratio = (
            (language_excellent_line - language_score)
            / (language_excellent_line - language_pass_line)
            if (language_excellent_line - language_pass_line) > 0
            else 0
        )
        reduction = LANGUAGE_PENALTY_LEVEL_3_THRESHOLD * gap_ratio * 0.15
        return max(0, min(100, base_score * (1 - reduction)))
    elif language_score < (language_pass_line - LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER * language_std):
        penalty = LANGUAGE_PENALTY_LEVEL_1_THRESHOLD
    elif language_score < (language_pass_line - LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER * language_std):
        penalty = LANGUAGE_PENALTY_LEVEL_2_THRESHOLD
    else:
        penalty = LANGUAGE_PENALTY_LEVEL_3_THRESHOLD

    return max(0, min(100, base_score * (1 - penalty)))


def _calculate_student_values(input_data, school_level_scores_map):
    school_level_str = input_data.get("school_level")
    if not school_level_str:
        background_university = input_data.get("background_university")
        if background_university:
            service = get_school_level_service()
            school_level_str = service.get_school_level(background_university)
        else:
            school_level_str = "未知"
    raw_school_score = school_level_scores_map.get(
        school_level_str, school_level_scores_map.get("未知", 11)
    )

    valid_scores = [s for s in school_level_scores_map.values() if isinstance(s, (int, float))]
    min_raw_score_val = min(valid_scores) if valid_scores else 1
    max_raw_score_val = max(valid_scores) if valid_scores else 11

    if (max_raw_score_val - min_raw_score_val) != 0:
        val_school = (
            (max_raw_score_val - raw_school_score) / (max_raw_score_val - min_raw_score_val)
        ) * 100
    else:
        val_school = 50
    val_school = max(0, min(val_school, 100))

    gpa_value = input_data.get("gpa") or input_data.get("gpa_score", 0)
    if isinstance(gpa_value, str) and (gpa_value == "未填写" or not gpa_value.strip()):
        gpa = 0
    elif gpa_value is None:
        gpa = 0
    else:
        gpa = float(gpa_value)
    val_gpa = _calculate_gpa_score(gpa) if gpa > 0 else 0

    lang_score_value = input_data.get("language_score")
    if lang_score_value is None:
        lang_score = 0
    elif isinstance(lang_score_value, str) and (
        lang_score_value == "未填写" or not lang_score_value.strip()
    ):
        lang_score = 0
    else:
        lang_score = float(lang_score_value)
    val_lang = _calculate_language_score(lang_score) if lang_score > 0 else 0

    research = input_data.get("research_count", 0)
    papers = input_data.get("paper_count", 0)
    academic_sum = research + papers
    val_academic = (
        min(academic_sum / MAX_ACADEMIC_ITEMS, 1.0) * 100 if MAX_ACADEMIC_ITEMS > 0 else 0
    )
    val_academic = max(0, min(val_academic, 100))

    internships = input_data.get("internship_count", 0)
    awards = input_data.get("award_count", 0)
    practical_sum = internships + awards
    val_practical = (
        min(practical_sum / MAX_PRACTICAL_ITEMS, 1.0) * 100 if MAX_PRACTICAL_ITEMS > 0 else 0
    )
    val_practical = max(0, min(val_practical, 100))

    return {
        "School Tier": val_school,
        "GPA": val_gpa,
        "Language Score": val_lang,
        "Academic": val_academic,
        "Practical": val_practical,
    }


def render_student_background_chart(input_data):
    if not input_data:
        return

    session_key = "student_background_chart_visible"
    layout_key = "student_background_chart_layout"
    close_btn_key = "close_chart_btn_clicked"

    if session_key not in st.session_state:
        st.session_state[session_key] = True

    if layout_key not in st.session_state:
        st.session_state[layout_key] = [dashboard.Item("background_chart", 0, 0, 4, 3)]

    if close_btn_key not in st.session_state:
        st.session_state[close_btn_key] = None

    if st.session_state[close_btn_key] == "clicked":
        st.session_state[session_key] = False
        st.session_state[close_btn_key] = None
        st.rerun()

    if not st.session_state[session_key]:
        return

    values = _calculate_student_values(input_data, SCHOOL_LEVEL_PRIORITY)

    chart_data = [
        {"axis": "学校层次", "分数": int(values["School Tier"])},
        {"axis": "GPA", "分数": int(values["GPA"])},
        {"axis": "语言成绩", "分数": int(values["Language Score"])},
        {"axis": "学术经历", "分数": int(values["Academic"])},
        {"axis": "实践经历", "分数": int(values["Practical"])},
    ]

    with elements("student_background_chart"):
        with dashboard.Grid(st.session_state[layout_key], onLayoutChange=sync(layout_key)):
            with mui.Paper(
                key="background_chart",
                sx={
                    "height": "100%",
                    "display": "flex",
                    "flexDirection": "column",
                    "border": "2px solid #007ACC",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 12px rgba(0, 122, 204, 0.15)",
                    "overflow": "hidden",
                    "backgroundColor": "#ffffff",
                },
            ):
                with mui.Box(
                    sx={
                        "backgroundColor": "#007ACC",
                        "color": "#ffffff",
                        "padding": "8px 12px",
                        "borderBottom": "2px solid #005a9e",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                    }
                ):
                    mui.Typography(
                        "学生综合实力",
                        variant="subtitle1",
                        sx={"fontWeight": 600, "fontSize": "14px"},
                    )
                    mui.IconButton(
                        mui.icon.Close,
                        onClick=sync(close_btn_key, "clicked"),
                        sx={
                            "color": "#ffffff",
                            "padding": "4px",
                            "&:hover": {"backgroundColor": "rgba(255, 255, 255, 0.1)"},
                        },
                    )

                with mui.Box(
                    sx={
                        "flex": 1,
                        "padding": "8px",
                        "display": "flex",
                        "flexDirection": "column",
                        "minHeight": 0,
                        "backgroundColor": "#fafafa",
                    }
                ):
                    nivo.Radar(
                        data=chart_data,
                        keys=["分数"],
                        indexBy="axis",
                        margin={"top": 50, "right": 60, "bottom": 50, "left": 60},
                        borderColor={"from": "color"},
                        gridLabelOffset=10,
                        dotSize=6,
                        dotColor={"theme": "background"},
                        dotBorderWidth=2,
                        colors={"scheme": "nivo"},
                        fillOpacity=0.25,
                        blendMode="multiply",
                        animate=True,
                        motionConfig="gentle",
                        isInteractive=True,
                    )
