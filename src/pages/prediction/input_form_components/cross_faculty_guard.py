from functools import lru_cache

import pandas as pd
import streamlit as st

from src.utils.app_data_loader import load_raw_cases_data, load_school_major_details_df
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

guard_logger = setup_logger("page3", "prediction")


@lru_cache(maxsize=1)
def _get_major_category_cache() -> dict[str, str]:
    details_df = load_school_major_details_df()
    if details_df is None or details_df.empty:
        return {}

    required_cols = ["学校", "专业英文名称", "专业大类"]
    if not all(col in details_df.columns for col in required_cols):
        return {}

    df = details_df[required_cols].dropna()
    df = df[~df["专业大类"].astype(str).str.lower().isin(["nan", "none"])]

    return {
        f"{row['学校'].strip()}|{row['专业英文名称'].strip()}": row["专业大类"].strip()
        for _, row in df.iterrows()
    }


def check_cross_faculty_situation(
    background_major: str,
    target_majors: list[str],
    target_universities: list[str],
    cases_df: pd.DataFrame,
) -> tuple[bool, str | None, set[str]]:
    from src.pages.prediction.prediction_utils import get_background_faculty

    background_faculty = get_background_faculty(background_major, cases_df)
    if not background_faculty:
        return False, None, set()

    major_category_cache = _get_major_category_cache()
    target_faculties: set[str] = set()

    for major in target_majors:
        if not major:
            continue
        for university in target_universities:
            if not university:
                continue
            target_faculty = major_category_cache.get(f"{university}|{major}")
            if target_faculty:
                target_faculties.add(target_faculty)

    has_cross_faculty = any(f != background_faculty for f in target_faculties)
    return has_cross_faculty, background_faculty, target_faculties


@st.dialog("提示", width="small")
def cross_faculty_confirm_dialog(
    session_manager: SessionManager, background_faculty: str, target_faculties: set[str]
) -> None:
    target_str = "、".join(sorted(target_faculties))
    st.markdown(
        f"检测到您的背景属于 **{background_faculty}**，而目标专业包含 **{target_str}** 方向。\n\n"
        "这属于跨大类申请，可能面临不同的评估标准，是否继续？"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "继续",
            type="primary",
            width="stretch",
            key="cross_faculty_confirm_btn",
            shortcut="Enter",
        ):
            session_manager.set(
                cross_faculty_confirmed=True,
                cross_faculty_cancelled=False,
                pending_cross_faculty_prediction=True,
            )
            st.rerun()

    with col2:
        if st.button("取消", width="stretch", key="cross_faculty_cancel_btn", shortcut="Esc"):
            session_manager.set(
                cross_faculty_confirmed=False,
                cross_faculty_cancelled=True,
                pending_cross_faculty_prediction=False,
                pending_prediction_data=None,
                prediction_submit_lock=False,
                submitted=False,
            )
            st.rerun()


@lru_cache(maxsize=1)
def _get_major_to_faculty_map() -> dict[str, str]:
    details_df = load_school_major_details_df()
    if details_df is None or details_df.empty or "专业大类" not in details_df.columns:
        return {}

    result: dict[str, str] = {}
    df = details_df.dropna(subset=["专业大类"])

    if "专业英文名称_聚合" in df.columns:
        for major, faculty in zip(
            df["专业英文名称_聚合"].astype(str), df["专业大类"].astype(str), strict=True
        ):
            if major.strip():
                result[major.strip()] = faculty.strip()

    if "专业英文名称" in df.columns:
        for major, faculty in zip(
            df["专业英文名称"].astype(str), df["专业大类"].astype(str), strict=True
        ):
            major_key = major.strip()
            if major_key and major_key not in result:
                result[major_key] = faculty.strip()

    return result


def _check_majors_with_agent(
    background_major: str, majors: list[str], cases_df: pd.DataFrame
) -> bool:
    if not majors:
        return False
    try:
        from src.agent.boundary_case_agent import BoundaryCaseAgent

        cases = []
        for m in majors:
            cases.append(
                {
                    "university": "Target University",
                    "major": m,
                    "similarity": 0.85,
                }
            )

        agent = BoundaryCaseAgent(cases_df=cases_df)
        result = agent.evaluate_boundary_cases(background_major, cases, mode="relax")
        decisions = result.get("decisions", [])

        if any(decisions):
            guard_logger.info(
                f"Agent验证通过跨学院专业: {[m for m, d in zip(majors, decisions, strict=False) if d]}"
            )
            return True
        return False
    except Exception as e:
        guard_logger.error(f"Agent跨学院验证失败: {e}")
        return False


def quick_cross_faculty_check(
    background_major: str | None,
    selected_categories: list[str] | None,
    selected_majors: list[str] | None,
    cases_df: pd.DataFrame | None = None,
) -> tuple[bool, str | None, set[str], bool]:
    selected_categories = selected_categories or []
    selected_majors = selected_majors or []

    if not background_major or (not selected_categories and not selected_majors):
        return False, None, set(), False

    if cases_df is None:
        cases_df = load_raw_cases_data()

    from src.pages.prediction.prediction_utils import get_background_faculty

    background_faculty = get_background_faculty(background_major, cases_df)
    if not background_faculty:
        return False, None, set(), False

    target_faculties: set[str] = set(selected_categories)

    cross_majors = []
    if selected_majors:
        major_to_faculty = _get_major_to_faculty_map()
        for major in selected_majors:
            major_str = str(major).strip()
            faculty = major_to_faculty.get(major_str)
            if faculty:
                target_faculties.add(faculty)
                if faculty != background_faculty:
                    cross_majors.append(major_str)

    has_cross = any(f and f != background_faculty for f in target_faculties)
    agent_approved = False

    if has_cross and cross_majors and cases_df is not None:
        agent_approved = _check_majors_with_agent(background_major, cross_majors, cases_df)

    return has_cross, background_faculty, target_faculties, agent_approved
