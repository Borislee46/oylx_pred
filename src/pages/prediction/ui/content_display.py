from typing import Any

import streamlit as st

from src.adjustment.sales_scenario import (
    precompute_product_combinations,
    sync_sales_sim_cache,
)
from src.agent.explain_profiles import classify_profile
from src.pages.prediction.handler_config import (
    DEFAULT_INTERNAL_KEYS,
    DEFAULT_SESSION_KEYS,
    DEFAULT_UI_KEYS,
)
from src.pages.prediction.result_display.competitiveness import (
    render_competitiveness_panel,
)
from src.pages.prediction.result_display.contract_state import (
    CONTRACT_TIER_KEY,
    _default_tier,
)
from src.pages.prediction.result_display.encourage_builder import build_encourage
from src.pages.prediction.result_display.hero_summary import (
    build_quality_badge_html,
    build_sales_hero_html,
    canonical_school_tiers,
)
from src.pages.prediction.result_display.portfolio_views import (
    render_sales_portfolio,
)
from src.pages.prediction.result_display.product_meta import (
    get_blocks_selection,
    normalize_selection,
    set_blocks_selection,
)
from src.pages.prediction.result_display.product_registry import (
    selectable_product_names,
)
from src.pages.prediction.result_display.sales_recommendation import (
    build_matched_products,
)
from src.pages.prediction.result_display.school_explorer import (
    render_school_explorer,
)
from src.pages.prediction.results_handler import reset_prediction_results
from src.pages.prediction.ui.hk_whatif_axis import render_hk_whatif_axis
from src.pages.prediction.ui.probability_blocks import render_probability_blocks
from src.pages.prediction.ui.sales_explain_bridge import build_sales_snapshot
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce
from src.utils.session_manager import SessionManager

_RESET_PREFIXES = (
    "background_",
    "gpa_",
    "language_",
    "lang_",
    "selected_",
    "standardized_",
    "current_exam",
    "dual_",
    "is_dual_degree",
    "research_",
    "internship_",
    "award_",
    "paper_",
    "experience_",
    "target_",
    "form_",
    "lead_in_",
    "_lead_in",
    "_auto_submit",
    "_form_anchor",
    "explain_",
    "_explain_",
    "_hk_pdf",
    "_knn_",
    "hk_ui",
    "hk_run",
    "hk_last",
    "pending_",
    "last_",
    "previous_",
    "input_data",
    "has_predicted",
    "prediction_",
    "processing_lock",
    "lock_start_time",
    "submitted",
    "is_school_selection_submit",
    "cross_faculty_",
    "fresh_prediction",
    "student_background_chart",
    "app_initialized",
    "_hk_edit_background",
    "_hk_canonical",
    "_hk_sales_",
    "_portfolio_",
    "_hk_ar_",
)

content_display_logger = setup_logger("page3", "prediction")


def _check_and_render_fallback_notice(res_model: Any) -> None:
    if res_model is None:
        return
    all_results = (
        (res_model.similarity_results or [])
        + (res_model.cross_major_results or [])
        + (res_model.user_specified_results or [])
    )
    if not all_results:
        return
    any_fallback = any(r.get("_is_fallback") for r in all_results)
    if not any_fallback:
        return
    fallback_level = min(
        (r.get("_fallback_level", 4) for r in all_results if r.get("_is_fallback")),
        default=4,
    )
    sample_counts = [
        r.get("_fallback_sample_count", 0)
        for r in all_results
        if r.get("_is_fallback") and r.get("_fallback_sample_count")
    ]
    min_n = min(sample_counts) if sample_counts else 0

    level_descriptions = {
        0: f"历史数据中有完全匹配的院校-专业组合（n≥{min_n}），以下概率为历史录取率，非模型预测。",
        1: f"基于相同背景院校的数据估算（n≥{min_n}），以下概率为历史录取率，非模型预测。",
        2: f"基于目标院校-专业的历史数据估算（n≥{min_n}），以下概率为历史录取率，非模型预测。",
        3: f"基于目标院校的全局数据估算（n≥{min_n}），以下概率为历史录取率，非模型预测。",
        4: "基于全部历史数据的全局录取率估算，以下概率非个性化模型预测。",
    }
    desc = level_descriptions.get(fallback_level, level_descriptions[4])

    st.warning(
        f"数据不完整，无法运行个性化预测模型。{desc}\n\n"
        "录取概率仅供参考，建议补充完整的GPA和语言成绩信息后重新预测。",
        icon=":material/warning:",
    )


def _reset_for_new_student(session_manager: SessionManager) -> None:
    from src.agent.context import StudentContext
    from src.pages.prediction.form_bridge import reset_lead_in_profile
    from src.pages.prediction.ui.page_state_machine import HKPagePhase, PageStateMachine

    reset_prediction_results(session_manager)
    for prefix in _RESET_PREFIXES:
        session_manager.clear_prefix(prefix)

    ctx = StudentContext()
    reset_lead_in_profile(session_manager, ctx, reset_state_machine=True)

    sm = PageStateMachine(session_manager)
    sm.transition(HKPagePhase.IDLE)
    session_manager.set(
        has_predicted=False,
        input_data=None,
        **{DEFAULT_INTERNAL_KEYS.edit_background: False},
    )


def _render_new_student_reset(session_manager: SessionManager) -> None:
    with st.container(key="hk_reset_cta"):
        if st.button(
            "开始新学生",
            type="tertiary",
            width="stretch",
            icon=":material/person_add:",
            key="hk_reset_btn",
        ):
            _reset_for_new_student(session_manager)
            st.rerun()


def _render_results_toolbar(
    session_manager: SessionManager,
    res_model: Any,
    input_data: dict,
) -> None:
    with st.container(key="hk_results_toolbar"):
        _, c1, _ = st.columns([1, 2, 1])
        with c1:
            _render_new_student_reset(session_manager)


@st.fragment
def _render_profile_card(
    session_manager: SessionManager,
    res_model: Any,
    input_data: dict,
    unified: list,
    has_ai: bool,
    profile: str = "",
) -> None:
    if not has_ai:
        return

    with st.container(key="hk_profile_card"):
        st.html('<div class="biz-section-head">申请画像</div>')

        combo = st.session_state.get("_hk_sales_combo") or []
        precomputed = st.session_state.get("_hk_sales_precomputed") or {}
        selected = get_blocks_selection(session_manager)
        contract_tier = session_manager.get(CONTRACT_TIER_KEY) or _default_tier(session_manager)
        snapshot = (
            build_sales_snapshot(input_data, selected, precomputed, contract_tier=contract_tier)
            if precomputed
            else None
        )
        from src.pages.prediction.ui.pathfinder import render_ai_explanation

        render_ai_explanation(
            res_model,
            input_data,
            portfolio_combo=combo,
            profile=profile,
            sales_snapshot=snapshot,
        )


@st.fragment
def _render_recommendation_card(
    session_manager: SessionManager,
    res_model: Any,
    input_data: dict,
    unified: list,
    page_state: Any,
    canonical_tiers: dict[str, str] | None = None,
) -> None:
    if not unified:
        return

    with st.container(key="hk_recommendation_card"):
        st.html('<div class="biz-section-head">推荐方案</div>')

        from src.pages.prediction.result_display.product_registry import (
            ProductKind,
            registry_by_name,
        )

        by_name = registry_by_name()

        products = build_matched_products(input_data, bool(res_model.cross_major_results))
        st.session_state["_ar_products"] = products
        if not get_blocks_selection(session_manager):
            ordered = selectable_product_names(input_data, products)
            ai_names = {p.get("name") for p in products if p.get("name")}
            if ai_names:
                set_blocks_selection([n for n in ordered if n in ai_names], session_manager)

        precomputed = precompute_product_combinations(page_state, input_data, unified)
        st.session_state["_hk_sales_precomputed"] = precomputed
        contract_tier = session_manager.get(CONTRACT_TIER_KEY) or _default_tier(session_manager)
        selected = render_probability_blocks(
            input_data,
            precomputed,
            page_state=page_state,
            contract_tier=contract_tier,
            session_manager=session_manager,
        )

        for name in normalize_selection(selected):
            p = by_name.get(name)
            if p and p.kind == ProductKind.APPLICATION:
                session_manager.set(**{"hk_contract_tier": p.catalog_id})
                break

        sync_sales_sim_cache(input_data, selected, precomputed)
        render_sales_portfolio(
            session_manager,
            res_model,
            input_data,
            canonical_tiers=canonical_tiers,
            precomputed=precomputed,
        )


def _render_results_view(
    session_manager: SessionManager,
    res_model: Any,
    current_input_data: dict,
    page_state: Any,
    submitted: bool,
    has_ai: bool,
    unified: list,
) -> None:
    all_candidates = (
        (res_model.similarity_results or [])
        + (res_model.cross_major_results or [])
        + (res_model.user_specified_results or [])
    )
    best_prob = max(
        (
            clip_probability_coerce(r.get("probability"))
            for r in all_candidates
            if r.get("probability")
        ),
        default=0,
    )
    _profile = classify_profile(
        {
            "similarity_results": res_model.similarity_results,
            "cross_major_results": res_model.cross_major_results,
            "unified_results": res_model.unified_results,
        }
    )
    _encourage = build_encourage(current_input_data, _profile, all_candidates)

    _canonical = canonical_school_tiers(res_model.unified_results or [])

    st.html(
        build_sales_hero_html(
            all_candidates,
            combo_count=len(all_candidates),
            best_prob=best_prob,
            encourage=_encourage,
            canonical_tiers=_canonical,
            include_prob_grid=False,
        )
    )

    quality_html = build_quality_badge_html(all_candidates)
    if quality_html:
        st.html(quality_html)

    render_hk_whatif_axis(page_state, current_input_data, unified)

    if unified:
        render_school_explorer(unified, current_input_data, canonical_tiers=_canonical)

    _render_recommendation_card(
        session_manager,
        res_model,
        current_input_data,
        unified,
        page_state,
        canonical_tiers=_canonical,
    )

    comp_candidates = (
        (res_model.similarity_results or [])
        + (res_model.cross_major_results or [])
        + (res_model.user_specified_results or [])
    )
    render_competitiveness_panel(comp_candidates, current_input_data, page_state.cases_df)

    _render_profile_card(
        session_manager, res_model, current_input_data, unified, has_ai, profile=_profile
    )

    st.html(
        '<p class="hk-sales-footnote">'
        "以上方案基于历史申请数据生成，录取结果受多种因素影响，仅供参考。"
        "</p>"
    )

    _render_results_toolbar(session_manager, res_model, current_input_data)


def display_content(
    session_manager: SessionManager,
    page_state: Any,
    submitted: bool,
    session_key_has_predicted: str = DEFAULT_SESSION_KEYS.has_predicted,
    session_key_input_data: str = DEFAULT_SESSION_KEYS.input_data,
    session_key_predict_lock: str = DEFAULT_SESSION_KEYS.predict_lock,
    session_key_form_data_changed: str = DEFAULT_SESSION_KEYS.form_data_changed,
) -> None:
    if not session_manager.get(session_key_has_predicted, False):
        return

    current_input_data = session_manager.get(session_key_input_data)
    if not current_input_data:
        content_display_logger.warning("has_predicted 为 True，但 session_state 中缺少输入数据。")
        reset_prediction_results(session_manager)
        session_manager.set(**{session_key_has_predicted: False, session_key_predict_lock: False})
        st.rerun()

    form_changed = session_manager.get(session_key_form_data_changed, False)
    if not submitted and form_changed:
        st.caption("您的输入已更改，当前显示的是先前输入的预测结果。请点击预测按钮获取最新结果。")

    res_model = session_manager.get(DEFAULT_UI_KEYS.prediction_results)

    _check_and_render_fallback_notice(res_model)

    has_ai = bool(res_model and (res_model.similarity_results or res_model.cross_major_results))
    unified = list(res_model.unified_results) if res_model and res_model.unified_results else []

    _render_results_view(
        session_manager, res_model, current_input_data, page_state, submitted, has_ai, unified
    )

    if not submitted and form_changed:
        session_manager.set(**{session_key_form_data_changed: False})
