from typing import Set, Tuple

import pandas as pd
import streamlit as st

from src.utils.app_data_loader import load_raw_cases_data, load_school_major_details_df
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

guard_logger = setup_logger("page3", "prediction")


def check_cross_faculty_situation(
    background_major: str,
    target_majors: list[str],
    target_universities: list[str],
    cases_df: pd.DataFrame,
) -> Tuple[bool, str | None, Set[str]]:
    background_faculty: str | None = None
    if background_major and cases_df is not None and not cases_df.empty:
        try:
            major_match = cases_df[cases_df["background_major"] == background_major]
            if not major_match.empty and "faculty" in major_match.columns:
                background_faculty = major_match["faculty"].iloc[0]
                if pd.isna(background_faculty) or str(background_faculty).strip() == "":
                    background_faculty = None
        except Exception as e:
            try:
                guard_logger.warning(f"查询背景学院失败: {e}")
            except Exception:
                pass

    if not background_faculty:
        return False, None, set()

    details_df = load_school_major_details_df()
    major_category_cache: dict[str, str] = {}

    if details_df is not None and not details_df.empty:
        required_cols = ["学校", "专业英文名称", "专业大类"]
        if all(col in details_df.columns for col in required_cols):
            try:
                for _, row in details_df.iterrows():
                    uni = str(row.get("学校", "")).strip()
                    maj = str(row.get("专业英文名称", "")).strip()
                    cat = str(row.get("专业大类", "")).strip()
                    if uni and maj and cat and cat.lower() not in ["nan", "none"]:
                        cache_key = f"{uni}|{maj}"
                        major_category_cache[cache_key] = cat
            except Exception as e:
                try:
                    guard_logger.warning(f"构建专业大类缓存失败: {e}")
                except Exception:
                    pass

    target_faculties: Set[str] = set()
    has_cross_faculty = False

    for major in target_majors:
        if not major:
            continue
        for university in target_universities:
            if not university:
                continue
            cache_key = f"{university}|{major}"
            target_faculty = major_category_cache.get(cache_key)
            if target_faculty:
                target_faculties.add(target_faculty)
                if target_faculty != background_faculty:
                    has_cross_faculty = True

    return has_cross_faculty, background_faculty, target_faculties


@st.dialog("提示", width="small")
def cross_faculty_confirm_dialog(
    session_manager: SessionManager, background_faculty: str, target_faculties: Set[str]
) -> None:
    st.write("您明确选择的目标专业包含跨学院方向，是否继续？")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "确定继续预测",
            type="primary",
            use_container_width=True,
            key="cross_faculty_confirm_btn",
        ):
            session_manager.set(
                cross_faculty_confirmed=True,
                cross_faculty_cancelled=False,
                pending_cross_faculty_prediction=True,
            )
            st.rerun()

    with col2:
        if st.button("取消", use_container_width=True, key="cross_faculty_cancel_btn"):
            session_manager.set(
                cross_faculty_confirmed=False,
                cross_faculty_cancelled=True,
                pending_cross_faculty_prediction=False,
                pending_prediction_data=None,
                prediction_submit_lock=False,
                submitted=False,
            )
            st.rerun()


def quick_cross_faculty_check(
    background_major: str | None,
    selected_categories: list[str] | None,
    selected_majors: list[str] | None,
    cases_df: pd.DataFrame | None = None,
    details_df: pd.DataFrame | None = None,
) -> tuple[bool, str | None, set[str]]:
    selected_categories = selected_categories or []
    selected_majors = selected_majors or []

    if not background_major or (not selected_categories and not selected_majors):
        return False, None, set()

    if cases_df is None:
        try:
            cases_df = load_raw_cases_data()
        except Exception:
            cases_df = pd.DataFrame()

    background_faculty: str | None = None
    try:
        if cases_df is not None and not cases_df.empty:
            major_match = cases_df[cases_df["background_major"] == background_major]
            if not major_match.empty and "faculty" in major_match.columns:
                background_faculty = major_match["faculty"].iloc[0]
                if pd.isna(background_faculty) or str(background_faculty).strip() == "":
                    background_faculty = None
    except Exception as e:
        try:
            guard_logger.warning(f"查询背景学院失败: {e}")
        except Exception:
            pass

    if not background_faculty:
        return False, None, set()

    target_faculties: set[str] = set(selected_categories)

    if selected_majors:
        if details_df is None:
            try:
                details_df = load_school_major_details_df()
            except Exception:
                details_df = pd.DataFrame()
        try:
            has_agg_col = "专业英文名称_聚合" in details_df.columns
            has_std_col = "专业英文名称" in details_df.columns
            has_cat_col = "专业大类" in details_df.columns

            if details_df is not None and not details_df.empty and has_cat_col:
                if has_agg_col and has_std_col:
                    df = details_df[["专业英文名称_聚合", "专业英文名称", "专业大类"]].dropna(
                        subset=["专业大类"]
                    )

                    matched_df = df[
                        df["专业英文名称_聚合"].astype(str).isin([str(m) for m in selected_majors])
                    ]
                    if len(matched_df) < len(selected_majors):
                        matched_std = df[
                            df["专业英文名称"].astype(str).isin([str(m) for m in selected_majors])
                        ]
                        matched_df = pd.concat([matched_df, matched_std]).drop_duplicates()
                elif has_std_col:
                    df = details_df[["专业英文名称", "专业大类"]].dropna()
                    matched_df = df[
                        df["专业英文名称"].astype(str).isin([str(m) for m in selected_majors])
                    ]
                else:
                    matched_df = pd.DataFrame()

                target_faculties.update(matched_df["专业大类"].astype(str).str.strip().tolist())
        except Exception as e:
            try:
                guard_logger.error(f"根据目标专业解析专业大类失败: {e}", exc_info=True)
            except Exception:
                pass

    has_cross = any(f for f in target_faculties if f and f != background_faculty)
    return has_cross, background_faculty, target_faculties
