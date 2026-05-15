"""
跨学部守卫 — 检测用户背景专业与目标专业是否跨越学部边界。

背景：
  XGBoost 模型在跨学部场景（如理→文）外推极不稳定。
  跨学部惩罚（×0.3）是调整链中最激进的一层。
  与其让模型给出不可靠的预测，不如在前端拦截并告知用户风险。

两阶段检测：
  1. quick_cross_faculty_check：快速比较背景专业 vs 目标专业的学部
  2. cross_faculty_confirm_dialog：用户确认弹窗（Streamlit @st.dialog）

当前状态：
  agent_approved 硬编码为 False（DEC-011 禁用了 LLM 自动审批跨学部）。
  之前用 Agent 判断跨学部的合理性（如理→工应该放行），
  但 Agent 幻觉风险不可接受——可能把不合理跨学部误判为合理。
"""

from functools import lru_cache

import pandas as pd
import streamlit as st

from src.pages.prediction.config.ui_messages import CROSS_FACULTY_MESSAGES
from src.pages.prediction.results_handler import clear_pending_prediction_state
from src.utils.app_data_loader import load_raw_cases_data, load_school_major_details_df
from src.utils.session_manager import SessionManager


# ── 跨学部确认弹窗 ──────────────────────────────────────
@st.dialog(CROSS_FACULTY_MESSAGES["dialog_title"], width="small")
def cross_faculty_confirm_dialog(
    session_manager: SessionManager, background_faculty: str, target_faculties: set[str]
) -> None:
    """Streamlit dialog：提示跨学部风险，用户选择继续或取消。

    确认路径：
      cross_faculty_confirmed=True + pending_cross_faculty_prediction=True
      → st.rerun() → hk.py _dispatch_prediction 检测 pending_cross → 重新提交

    取消路径：
      clear_pending_prediction_state + hk_ui_phase="idle"
      → 回到初始状态，用户可修改目标专业
    """
    target_str = "、".join(sorted(target_faculties))
    st.markdown(
        CROSS_FACULTY_MESSAGES["dialog_body"].format(
            bg_faculty=background_faculty, target_faculties=target_str
        )
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            CROSS_FACULTY_MESSAGES["confirm_button"],
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
        if st.button(
            CROSS_FACULTY_MESSAGES["cancel_button"],
            width="stretch",
            key="cross_faculty_cancel_btn",
            shortcut="Esc",
        ):
            clear_pending_prediction_state(session_manager)
            session_manager.set(
                cross_faculty_confirmed=False,
                cross_faculty_cancelled=True,
                prediction_submit_lock=False,
                submitted=False,
                hk_ui_phase="idle",
                hk_last_error=None,
            )
            st.rerun()


# ── 专业名 → 学部映射（缓存）────────────────────────────
@lru_cache(maxsize=1)
def _get_major_to_faculty_map() -> dict[str, str]:
    """从 school_major_details 表中构建专业名 → 学部映射。

    @lru_cache(maxsize=1)：整个 session 内只计算一次。
    尝试两种列名：专业英文名称_聚合（聚合专业名）优先，专业英文名称（原始专业名）兜底。
    """
    details_df = load_school_major_details_df()
    if details_df is None or details_df.empty or "专业大类" not in details_df.columns:
        return {}

    result: dict[str, str] = {}
    df = details_df.dropna(subset=["专业大类"])

    # 聚合专业名优先（覆盖更广）
    if "专业英文名称_聚合" in df.columns:
        for major, faculty in zip(
            df["专业英文名称_聚合"].astype(str), df["专业大类"].astype(str), strict=True
        ):
            if major.strip():
                result[major.strip()] = faculty.strip()

    # 原始专业名兜底（不覆盖已存在的聚合专业名映射）
    if "专业英文名称" in df.columns:
        for major, faculty in zip(
            df["专业英文名称"].astype(str), df["专业大类"].astype(str), strict=True
        ):
            major_key = major.strip()
            if major_key and major_key not in result:
                result[major_key] = faculty.strip()

    return result


def quick_cross_faculty_check(
    background_major: str | None,
    selected_categories: list[str] | None,
    selected_majors: list[str] | None,
    cases_df: pd.DataFrame | None = None,
) -> tuple[bool, str | None, set[str], bool]:
    """快速检测背景专业与目标专业是否存在跨学部。

    检测逻辑：
    1. 获取背景专业的学部（从 cases_df 的 faculty 列）
    2. 收集目标专业的学部（selected_categories 直接是学部名，selected_majors 需映射）
    3. 任一个目标学部 ≠ 背景学部 → has_cross=True

    Returns:
        (has_cross, background_faculty, target_faculties, agent_approved)
        agent_approved 当前硬编码为 False（DEC-011）
    """
    selected_categories = selected_categories or []
    selected_majors = selected_majors or []

    if not background_major or (not selected_categories and not selected_majors):
        return False, None, set(), False

    if cases_df is None:
        cases_df = load_raw_cases_data()

    from src.pages.prediction.core.utils import get_background_faculty

    background_faculty = get_background_faculty(background_major, cases_df)
    if not background_faculty:
        return False, None, set(), False

    # selected_categories 本身就是学部名
    target_faculties: set[str] = set(selected_categories)

    # selected_majors 需要通过 major_to_faculty 映射
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
    agent_approved = False  # DEC-011：禁用 LLM 自动审批跨学部

    return has_cross, background_faculty, target_faculties, agent_approved
