from __future__ import annotations

from typing import Any

import streamlit as st

from src.pages.prediction.result_display.contract_state import (
    CONTRACT_TIER_KEY,
    _default_tier,
)
from src.pages.prediction.result_display.portfolio_service import (
    build_pool,
    load_correlation,
)
from src.pages.prediction.result_display.product_meta import get_blocks_selection
from src.pages.prediction.result_display.product_registry import (
    ProductKind,
    registry_by_name,
)
from src.portfolio.expected_value import (
    decompose,
)
from src.portfolio.pool_builder import prediction_results_to_schools
from src.portfolio.portfolio_contract import (
    PortfolioContract,
    filter_pool_by_contract,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce, prob_to_pct
from src.utils.session_manager import SessionManager

_logger = setup_logger("page3", "prediction")

_TIER_COLOR = {"保底": "#22c55e", "适中": "#3b82f6", "冲刺": "#f97316"}


def _resolve_contract_from_blocks(
    session_manager: SessionManager,
) -> PortfolioContract:
    selected = get_blocks_selection(session_manager)
    by_name = registry_by_name()
    for name in selected if isinstance(selected, list) else []:
        prod = by_name.get(name)
        if prod is not None and prod.kind == ProductKind.APPLICATION:
            contract = PortfolioContract.from_tier(prod.catalog_id)
            session_manager.set(**{CONTRACT_TIER_KEY: prod.catalog_id})
            return contract
    return PortfolioContract.from_tier(
        session_manager.get(CONTRACT_TIER_KEY) or _default_tier(session_manager)
    )


def _combo_tiers(
    combo: list[dict[str, Any]],
    canonical_tiers: dict[str, str] | None = None,
) -> list[str]:
    if canonical_tiers:
        return [canonical_tiers.get(c.get("university", ""), "适中") for c in combo]
    from src.agent.schemas import compute_tiers

    return compute_tiers([clip_probability_coerce(c.get("probability")) for c in combo])


def render_sales_portfolio(
    session_manager: SessionManager,
    res_model: Any,
    input_data: dict[str, Any],
    canonical_tiers: dict[str, str] | None = None,
    precomputed: dict[str, Any] | None = None,
) -> None:
    from src.adjustment.sales_scenario import affecting_products
    from src.pages.prediction.result_display.portfolio_service import (
        _COMBO_CACHE,
        _pool_fingerprint,
        select_combo_for_pool,
    )

    contract = _resolve_contract_from_blocks(session_manager)

    selected = get_blocks_selection(session_manager)
    affecting = affecting_products(selected)
    pool = None
    if affecting and precomputed:
        combo_key = "|".join(sorted(affecting))
        upgraded = precomputed.get("unified_by_combo", {}).get(combo_key)
        if upgraded:
            _logger.info(
                "render_sales_portfolio: using upgraded unified (products=%s, n=%d)",
                affecting,
                len(upgraded),
            )
            pool = filter_pool_by_contract(
                prediction_results_to_schools(upgraded),
                contract,
            )

    if pool is None:
        pool = build_pool(res_model, contract)

    _logger.info(
        "render_sales_portfolio: pool=%d max_schools=%d regions=%s",
        len(pool),
        contract.max_schools,
        contract.regions,
    )
    if not pool:
        _logger.warning("render_sales_portfolio: empty pool, skipping")
        st.caption("当前预测结果中暂无匹配该合同地区的院校，请先运行预测或切换合同套餐。")
        return

    n_unis = len({s.get("university") for s in pool})
    corr, pair = load_correlation() if n_unis > 2 else (None, None)

    fp = _pool_fingerprint(pool, contract.tier_id, "rec_1.0_0.0")
    if fp in st.session_state.get(_COMBO_CACHE, {}) or n_unis <= 2:
        combo = select_combo_for_pool(pool, contract, corr, pair, risk_weight=1.0)
    else:
        with st.skeleton(height=120):
            combo = select_combo_for_pool(pool, contract, corr, pair, risk_weight=1.0)

    st.session_state["_hk_sales_combo"] = combo  # 供 AI 解释联动
    if combo:
        from src.pages.prediction.ui.explain_cache import (
            _PORTFOLIO_FP_KEY,
            invalidate_pathfinder_pdf,
            portfolio_combo_fingerprint,
        )

        new_fp = portfolio_combo_fingerprint(combo)
        prev_fp = st.session_state.get(_PORTFOLIO_FP_KEY)
        st.session_state[_PORTFOLIO_FP_KEY] = new_fp
        if prev_fp is not None and prev_fp != new_fp:
            invalidate_pathfinder_pdf()
            if "explain_cache" in st.session_state:
                st.session_state["explain_cache"] = {}
    if not combo:
        return

    d = decompose(
        combo,
        contract,
        risk_weight=1.0,
        correlation_matrix=corr,
        pair_weight_matrix=pair,
        compute_ev=False,
    )
    tiers = _combo_tiers(combo, canonical_tiers)
    counts: dict[str, int] = {"保底": 0, "适中": 0, "冲刺": 0}
    for t in tiers:
        counts[t] = counts.get(t, 0) + 1
    admit_at_least_one = (1.0 - d.p_all_reject) * 100

    rows: list[str] = []
    for s, t in sorted(
        zip(combo, tiers, strict=False),
        key=lambda x: -clip_probability_coerce(x[0].get("probability")),
    ):
        color = _TIER_COLOR.get(t, "#3b82f6")
        prob = clip_probability_coerce(s.get("probability"))
        rows.append(
            f'<div class="hk-pf-row">'
            f'<span class="hk-pf-tier" style="background:{color}">{t}</span>'
            f'<span class="hk-pf-uni">{s.get("university", "")}</span>'
            f'<span class="hk-pf-major">{s.get("major", "")}</span>'
            f'<span class="hk-pf-prob" style="color:{color}">{prob_to_pct(prob)}%</span>'
            f"</div>"
        )

    bands = "　·　".join(
        f'<b style="color:{_TIER_COLOR[k]}">{counts[k]}</b> {k}'
        for k in ("冲刺", "适中", "保底")
        if counts.get(k)
    )

    if len(combo) == 1:
        why_html = (
            f'<div class="hk-pf-why">基于你的目标院校，系统为你匹配了 1 个最相关的专业方向，'
            f"估算录取概率约 <b>{admit_at_least_one:.0f}%</b>（基于历史同类申请数据）。"
            "如需扩大选校范围，可以和顾问沟通增加目标院校。</div>"
        )
    elif len(combo) <= 3:
        why_html = (
            f'<div class="hk-pf-why">基于你的目标院校范围，系统推荐上述组合，'
            f"整体<b>全案把握（估算）约 {admit_at_least_one:.0f}%</b>"
            f"（至少一所；按各校概率独立估算，未计入同申相关）。"
            "如需更多备选方案，可以和顾问沟通扩大选校范围。</div>"
        )
    else:
        why_html = (
            f'<div class="hk-pf-why">这套组合兼顾了院校档次与录取稳妥度：既保留冲刺名校的机会，'
            f"也配了保底选项托底，整体<b>全案把握（估算）约 {admit_at_least_one:.0f}%</b>"
            f"（至少一所；按各校概率独立估算，未计入同申相关）。"
            "具体冲/保如何搭配，可以和顾问进一步沟通调整。</div>"
        )

    st.html(
        '<div class="hk-pf-card">'
        f'<div class="hk-pf-sub">合同可申 {contract.max_schools} 所，已在范围内为你优选 '
        f"{len(combo)} 所 —— {bands}</div>"
        f'<div class="hk-pf-list">{"".join(rows)}</div>'
        f"{why_html}"
        "</div>"
    )
