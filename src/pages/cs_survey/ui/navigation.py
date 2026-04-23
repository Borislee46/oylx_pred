from __future__ import annotations

import streamlit as st

from ..schema import SurveyConfig


def navigate_to_view(cfg: SurveyConfig, view_key: str) -> None:
    st.session_state[f"cs_survey_active_view::{cfg.id}"] = view_key
    params = dict(st.query_params)
    params["survey"] = cfg.id
    params["view"] = view_key
    st.query_params.clear()
    st.query_params.update(params)
    st.rerun()


def render_view_nav(
    cfg: SurveyConfig,
    current_view: str,
    *,
    title: str | None = None,
    key_prefix: str = "cs_survey_nav",
    view_keys: list[str] | None = None,
) -> None:
    views = cfg.views
    if view_keys is not None:
        allowed = set(view_keys)
        views = [view for view in cfg.views if view.key in allowed]
    if len(views) <= 1:
        return
    if title:
        st.markdown(f'<p class="cs-view-nav-title">{title}</p>', unsafe_allow_html=True)
    cols = st.columns(len(views))
    for idx, view in enumerate(views):
        with cols[idx]:
            clicked = st.button(
                view.label,
                key=f"{key_prefix}_{view.key}",
                type="primary" if view.key == current_view else "secondary",
                width="stretch",
                disabled=view.key == current_view,
            )
            if clicked:
                navigate_to_view(cfg, view.key)
