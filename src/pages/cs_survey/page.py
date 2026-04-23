from __future__ import annotations

import streamlit as st

from .registry import list_surveys
from .schema import SurveyConfig
from .ui.footer import render_footer
from .ui.theme_css import inject_base_css
from .views import VIEW_REGISTRY

_LEGACY_VIEW_MAP = {"p1": "by_product", "p2": "by_pillar"}


def _get_query(name: str) -> str | None:
    raw = st.query_params.get(name)
    if isinstance(raw, (list, tuple)):
        return raw[0] if raw else None
    return raw


def _resolve_legacy_view() -> str | None:
    raw = _get_query("view")
    if raw in _LEGACY_VIEW_MAP:
        return _LEGACY_VIEW_MAP[raw]
    return raw


def _set_query_param(name: str, value: str) -> None:
    current = _get_query(name)
    if current == value:
        return
    params = dict(st.query_params)
    params[name] = value
    st.query_params.clear()
    st.query_params.update(params)


def _pick_survey(surveys: list[SurveyConfig]) -> SurveyConfig:
    qid = _get_query("survey")
    for s in surveys:
        if s.id == qid:
            return s
    if len(surveys) == 1:
        return surveys[0]
    labels = {s.id: s.title for s in surveys}
    default_id = st.session_state.get("cs_survey_active_id", surveys[0].id)
    if default_id not in labels:
        default_id = surveys[0].id
    chosen = st.selectbox(
        "选择调研",
        list(labels.keys()),
        index=list(labels.keys()).index(default_id),
        format_func=lambda k: labels[k],
        key="cs_survey_active_id",
    )
    _set_query_param("survey", chosen)
    return next(s for s in surveys if s.id == chosen)


def _pick_view(cfg: SurveyConfig) -> str:
    session_key = f"cs_survey_active_view::{cfg.id}"
    raw_q = _get_query("view")
    if raw_q in _LEGACY_VIEW_MAP:
        _set_query_param("view", _LEGACY_VIEW_MAP[raw_q])
    keys = [v.key for v in cfg.views]
    if not keys:
        return ""
    query_view = _resolve_legacy_view()
    if query_view in keys:
        st.session_state[session_key] = query_view
    elif st.session_state.get(session_key) not in keys:
        st.session_state[session_key] = keys[0]
    chosen = st.session_state[session_key]
    _set_query_param("view", chosen)
    return chosen


def render() -> None:
    inject_base_css()
    surveys = list_surveys()
    if not surveys:
        st.warning("未配置任何调研。请在 config/cs_survey/ 下添加 YAML 配置。")
        return

    cfg = _pick_survey(surveys)
    _set_query_param("survey", cfg.id)
    view_key = _pick_view(cfg)

    view_spec = cfg.view(view_key)
    if view_spec is None:
        st.error(f"未找到视图 {view_key}")
        return

    renderer = VIEW_REGISTRY.get(view_spec.type)
    if renderer is None:
        st.error(f"未注册的视图类型 {view_spec.type}")
        return

    renderer(cfg)

    show_back = view_spec.type != "overview"
    render_footer(show_back_to_overview=show_back)
