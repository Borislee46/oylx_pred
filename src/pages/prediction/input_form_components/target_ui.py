from functools import partial
from typing import List, Set, Tuple

import pandas as pd
import streamlit as st

from src.pages.prediction.input_form_components.target_options_service import (
    build_target_base_df,
    compute_options,
    compute_selection_cache_key,
    expand_aggregated_majors_for_prediction,
)
from src.pages.prediction.input_form_components.widget_helpers import SelectBoxHelper


def _build_target_cache(session_manager, cases_df) -> pd.DataFrame:
    if session_manager.get("target_section_cache") is None:
        from src.utils.app_data_loader import load_school_major_details_df

        details_df = load_school_major_details_df()
        base_df, university_country_map = build_target_base_df(cases_df, details_df)
        session_manager.set(
            target_section_cache={
                "base_df": base_df,
                "university_country_map": university_country_map,
            }
        )

    target_section_cache = session_manager.get("target_section_cache", {})
    return target_section_cache.get("base_df", pd.DataFrame())


def _get_target_options(
    session_manager,
    base_df: pd.DataFrame,
    selected_countries: Set[str],
    selected_universities: Set[str],
    selected_categories: Set[str],
    selected_majors: Set[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    selection_cache_key = compute_selection_cache_key(
        selected_countries, selected_universities, selected_categories, selected_majors
    )

    target_options_cache = session_manager.get("target_options_cache", {})

    if selection_cache_key in target_options_cache:
        cached_options = target_options_cache[selection_cache_key]
        return (
            cached_options["country"],
            cached_options["university"],
            cached_options["category"],
            cached_options["major"],
        )

    options = compute_options(
        base_df,
        selected_countries,
        selected_universities,
        selected_categories,
        selected_majors,
    )

    target_options_cache[selection_cache_key] = {
        "country": options[0],
        "university": options[1],
        "category": options[2],
        "major": options[3],
    }
    session_manager.set(target_options_cache=target_options_cache)

    return options


def _render_target_multiselects(
    helper: SelectBoxHelper,
    form_state_manager,
    selected_countries: Set[str],
    selected_universities: Set[str],
    selected_categories: Set[str],
    selected_majors: Set[str],
    options_for_country_select: List[str],
    options_for_uni_select: List[str],
    options_for_category_select: List[str],
    options_for_major_select: List[str],
) -> None:
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        helper.render_multiselect(
            "国家/地区（选填）",
            options=options_for_country_select,
            default_selections=list(selected_countries),
            widget_key="target_countries_multiselect",
            on_change_callback=partial(
                form_state_manager.on_target_country_change, helper.session_manager
            ),
        )

    with row1_col2:
        helper.render_multiselect(
            "目标院校（选填）",
            options=options_for_uni_select,
            default_selections=list(selected_universities),
            widget_key="target_universities_multiselect",
            on_change_callback=partial(
                form_state_manager.on_target_university_change, helper.session_manager
            ),
        )

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        helper.render_multiselect(
            "所属学院（选填）",
            options=options_for_category_select,
            default_selections=list(selected_categories),
            widget_key="target_major_categories_multiselect",
            on_change_callback=partial(
                form_state_manager.on_major_category_change, helper.session_manager
            ),
        )

    with row2_col2:
        helper.render_multiselect(
            "目标专业（选填）",
            options=options_for_major_select,
            default_selections=list(selected_majors),
            widget_key="target_majors_multiselect",
            on_change_callback=partial(
                form_state_manager.on_target_major_change, helper.session_manager
            ),
        )


def _calculate_prediction_scope(
    base_df: pd.DataFrame,
    selected_countries: Set[str],
    selected_universities: Set[str],
    selected_categories: Set[str],
    selected_majors: Set[str],
    options_for_uni_select: List[str],
    options_for_major_select: List[str],
) -> Tuple[List[str], List[str]]:
    expand_universities = not selected_universities and (
        selected_countries or selected_categories or selected_majors
    )
    expand_majors = not selected_majors and (
        selected_universities or selected_countries or selected_categories
    )

    prediction_universities = (
        options_for_uni_select if expand_universities else list(selected_universities)
    )

    aggregated_to_use = options_for_major_select if expand_majors else list(selected_majors)
    prediction_majors = expand_aggregated_majors_for_prediction(
        base_df,
        selected_countries,
        selected_universities,
        selected_categories,
        aggregated_to_use,
    )

    return prediction_universities, prediction_majors


def _get_all_targets(base_df: pd.DataFrame, cases_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    all_unis = []
    all_majors = []

    if base_df is not None and not base_df.empty:
        if "target_university" in base_df.columns:
            all_unis = base_df["target_university"].dropna().astype(str).unique().tolist()
        if "target_major" in base_df.columns:
            all_majors = base_df["target_major"].dropna().astype(str).unique().tolist()
    else:
        if cases_df is not None and "target_university" in cases_df.columns:
            all_unis = list(set(cases_df["target_university"].astype(str).unique()))
        if cases_df is not None and "target_major" in cases_df.columns:
            all_majors = list(set(cases_df["target_major"].astype(str).unique()))

    return all_unis, all_majors


def render_target_section(session_manager, form_state_manager, cases_df, logger):
    st.markdown("**申请信息**")

    helper = SelectBoxHelper(session_manager, form_state_manager, logger)

    base_df = _build_target_cache(session_manager, cases_df)

    selected_countries = set(session_manager.get("selected_target_countries", []))
    selected_universities = set(session_manager.get("selected_target_universities", []))
    selected_categories = set(session_manager.get("selected_major_categories", []))
    selected_majors = set(session_manager.get("selected_target_majors", []))

    (
        options_for_country_select,
        options_for_uni_select,
        options_for_category_select,
        options_for_major_select,
    ) = _get_target_options(
        session_manager,
        base_df,
        selected_countries,
        selected_universities,
        selected_categories,
        selected_majors,
    )

    _render_target_multiselects(
        helper,
        form_state_manager,
        selected_countries,
        selected_universities,
        selected_categories,
        selected_majors,
        options_for_country_select,
        options_for_uni_select,
        options_for_category_select,
        options_for_major_select,
    )

    prediction_universities, prediction_majors = _calculate_prediction_scope(
        base_df,
        selected_countries,
        selected_universities,
        selected_categories,
        selected_majors,
        options_for_uni_select,
        options_for_major_select,
    )

    all_unis, all_majors = _get_all_targets(base_df, cases_df)

    return (prediction_universities, prediction_majors, all_unis, all_majors)
