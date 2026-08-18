from functools import partial

import streamlit as st

from src.pages.prediction.handler_config import DEFAULT_WIDGET_KEYS
from src.pages.prediction.input_form_components.widget_helpers import SelectBoxHelper
from src.utils.schools.data import load_school_base_data

DUAL_ALPHA_MINOR = 0.85


def _log_background_university_change(session_manager, form_state_manager, logger):
    selected_university = session_manager.get_widget_value(
        DEFAULT_WIDGET_KEYS.background_university
    )
    session_manager.set(background_university=selected_university)
    logger.info(f"用户选择背景院校: {selected_university}")
    form_state_manager.on_form_change(session_manager, change_type="select")


def _log_background_major_change(session_manager, form_state_manager, logger):
    selected_major = session_manager.get_widget_value(DEFAULT_WIDGET_KEYS.background_major)
    logger.info(f"用户选择背景专业: {selected_major}")
    form_state_manager.on_form_change(session_manager, change_type="select")


def _log_background_major_2_change(session_manager, form_state_manager, logger):
    selected_major_2 = session_manager.get_widget_value(DEFAULT_WIDGET_KEYS.background_major_2)
    logger.info(f"用户选择第二专业: {selected_major_2}")
    form_state_manager.on_form_change(session_manager, change_type="select")


def _log_dual_degree_change(session_manager, form_state_manager, logger):
    logger.info("双学位/辅修切换")
    form_state_manager.on_form_change(session_manager, change_type="select")


def _generate_university_options(session_manager, cases_df):
    universities: set[str] = set()
    university_counts: dict[str, int] = {}

    if cases_df is not None and "background_university" in cases_df.columns:
        col = cases_df["background_university"]
        universities.update(
            v for v in col.astype(str).unique() if v.strip().lower() not in ("", "nan", "none")
        )
        university_counts = col.value_counts().to_dict()

    if session_manager.get("school_base_df") is None:
        session_manager.set(school_base_df=load_school_base_data())

    school_base_df = session_manager.get("school_base_df")
    if school_base_df is not None and "学校名称" in school_base_df.columns:
        universities.update(
            v for v in school_base_df["学校名称"].astype(str).unique() if v.strip().lower() != "nan"
        )

    return sorted(universities, key=lambda x: university_counts.get(x, 0), reverse=True)


def _generate_major_options(cases_df):
    required_cols = ["background_major_original", "background_major"]
    if cases_df is None or not all(col in cases_df.columns for col in required_cols):
        return {"majors_display": [], "major_map": {}}

    mapping_df = cases_df[required_cols].drop_duplicates()
    major_map = dict(
        zip(mapping_df["background_major_original"], mapping_df["background_major"], strict=True)
    )
    major_counts = cases_df["background_major"].value_counts().to_dict()

    majors_display = [
        m
        for m in mapping_df["background_major_original"].astype(str).unique()
        if m.lower() not in ["none", "nan"]
    ]
    majors_display.sort(key=lambda m: major_counts.get(major_map.get(m, m), 0), reverse=True)

    return {"majors_display": majors_display, "major_map": major_map}


def render_background_section(session_manager, form_state_manager, cases_df, logger):
    st.markdown("**背景信息**")

    helper = SelectBoxHelper(session_manager, form_state_manager, logger)

    background_university = helper.render_cached_selectbox(
        label="背景院校",
        widget_key="background_university_selectbox",
        cache_key="background_universities_cache",
        history_key="background_university",
        options_generator_func=lambda: _generate_university_options(session_manager, cases_df),
        on_change_callback=partial(
            _log_background_university_change,
            session_manager,
            form_state_manager,
            logger,
        ),
    )

    display_label = session_manager.get("background_university_display_label")
    if display_label:
        st.caption(f"识别为**{display_label}**，已为您匹配代表性院校进行预测")

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        selected_background_major_original = helper.render_cached_selectbox(
            label="背景专业",
            widget_key="background_major_selectbox",
            cache_key="background_majors_cache",
            history_key="background_major_original",
            options_generator_func=lambda: _generate_major_options(cases_df),
            on_change_callback=partial(
                _log_background_major_change, session_manager, form_state_manager, logger
            ),
            options_path_in_cache="majors_display",
        )

    major_map = session_manager.get("background_majors_cache", {}).get("major_map", {})
    background_major = (
        major_map.get(selected_background_major_original)
        if selected_background_major_original
        else None
    )

    with col_m2:
        majors_display = session_manager.get("background_majors_cache", {}).get(
            "majors_display", []
        )
        m2_options = ["(无)"] + [
            m for m in majors_display if m != selected_background_major_original
        ]

        saved_m2 = session_manager.get("background_major_2_original")
        m2_default_idx = 0
        if saved_m2 and saved_m2 in m2_options:
            m2_default_idx = m2_options.index(saved_m2)
        elif DEFAULT_WIDGET_KEYS.background_major_2 in st.session_state:
            # 第一专业变化导致已选第二专业不在选项中 → 回退到 "(无)"，避免 Stale 值漂移
            if st.session_state[DEFAULT_WIDGET_KEYS.background_major_2] not in m2_options:
                st.session_state[DEFAULT_WIDGET_KEYS.background_major_2] = "(无)"

        selected_major_2_original = st.selectbox(
            "第二专业",
            options=m2_options,
            index=m2_default_idx,
            key=DEFAULT_WIDGET_KEYS.background_major_2,
            on_change=partial(
                _log_background_major_2_change, session_manager, form_state_manager, logger
            ),
            placeholder="辅修或双学位",
            help="如有辅修或双学位可在此选择，系统将综合评估两个专业的申请竞争力。",
        )

    background_major_2_original = (
        None if selected_major_2_original == "(无)" else selected_major_2_original
    )
    background_major_2 = (
        major_map.get(background_major_2_original) if background_major_2_original else None
    )

    is_dual_degree = False
    dual_alpha = 0.85
    if background_major_2_original:
        if DEFAULT_WIDGET_KEYS.dual_degree_type not in st.session_state:
            st.session_state[DEFAULT_WIDGET_KEYS.dual_degree_type] = "辅修"

        degree_type = st.segmented_control(
            "第二专业类型",
            options=["辅修", "双学位"],
            selection_mode="single",
            key=DEFAULT_WIDGET_KEYS.dual_degree_type,
            on_change=partial(_log_dual_degree_change, session_manager, form_state_manager, logger),
        )
        is_dual_degree = degree_type == "双学位"
        dual_alpha = 1.0 if is_dual_degree else DUAL_ALPHA_MINOR
        session_manager.set(is_dual_degree=is_dual_degree, dual_alpha=dual_alpha)

    return (
        background_university,
        selected_background_major_original,
        background_major,
        background_major_2_original,
        background_major_2,
        is_dual_degree,
        dual_alpha,
    )
