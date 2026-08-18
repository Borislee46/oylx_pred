from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pandas as pd
import streamlit as st

from src.portfolio.expected_value import (
    decompose,
    ev_select,
    frontier_and_nash,
)
from src.portfolio.pool_builder import prediction_results_to_schools
from src.portfolio.portfolio_contract import (
    PortfolioContract,
    filter_pool_by_contract,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce, prob_round, prob_to_pct

_logger = setup_logger("page3", "prediction")

_CORR_PATH = "cache/correlation_matrix.feather"
_PAIR_WEIGHT_PATH = "cache/pair_weight_matrix.feather"

_COMBO_CACHE = "_portfolio_combo_cache"
_FRONTIER_CACHE = "_portfolio_frontier_cache"

_MAX_CACHE_ENTRIES = 16


def _lru_put(cache: dict, key: str, value: object, maxsize: int = _MAX_CACHE_ENTRIES) -> None:
    cache[key] = value
    if len(cache) > maxsize:
        oldest = next(iter(cache))
        del cache[oldest]
        _logger.debug(
            "_lru_put: evicted oldest key=%s (cache size now %d)", oldest[:16], len(cache)
        )


@st.cache_data(show_spinner=False, ttl=3600)
def load_correlation() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    corr = pd.read_feather(_CORR_PATH) if os.path.exists(_CORR_PATH) else None
    pair = pd.read_feather(_PAIR_WEIGHT_PATH) if os.path.exists(_PAIR_WEIGHT_PATH) else None
    return corr, pair


def build_pool(res_model: Any, contract: PortfolioContract) -> list[dict[str, Any]]:
    unified = getattr(res_model, "unified_results", None) or []
    pool = prediction_results_to_schools(unified)
    filtered = filter_pool_by_contract(pool, contract)
    regions = "、".join(contract.regions)
    _logger.info(
        "build_pool: %d unified → %d pool → %d filtered (%s, tier=%s)",
        len(unified),
        len(pool),
        len(filtered),
        regions,
        contract.tier_id,
    )
    return filtered


def _pool_fingerprint(pool: list[dict[str, Any]], tier_id: str, suffix: str = "") -> str:
    key = [(s.get("university"), s.get("major"), prob_round(s.get("probability"))) for s in pool]
    raw = json.dumps([tier_id, suffix, key], sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def cached_recommended(
    pool: list[dict[str, Any]],
    contract: PortfolioContract,
    corr: pd.DataFrame | None,
    pair: pd.DataFrame | None,
    *,
    risk_weight: float = 1.0,
    similarity_floor: float = 0.0,
) -> list[dict[str, Any]]:
    if not pool:
        return []
    fp = _pool_fingerprint(pool, contract.tier_id, f"rec_{risk_weight}_{similarity_floor}")
    cache = st.session_state.setdefault(_COMBO_CACHE, {})
    if fp in cache:
        _logger.info(
            "cached_recommended: cache hit (pool=%d, rw=%.1f, sim≥%.2f)",
            len(pool),
            risk_weight,
            similarity_floor,
        )
    else:
        _logger.info(
            "cached_recommended: cache miss, computing (pool=%d, rw=%.1f, sim≥%.2f)",
            len(pool),
            risk_weight,
            similarity_floor,
        )
        _lru_put(
            cache,
            fp,
            ev_select(
                pool,
                contract,
                risk_weight=risk_weight,
                similarity_floor=similarity_floor,
                correlation_matrix=corr,
                pair_weight_matrix=pair,
            ),
        )
    return cache[fp]


def cached_frontier(
    pool: list[dict[str, Any]],
    contract: PortfolioContract,
    corr: pd.DataFrame | None,
    pair: pd.DataFrame | None,
    *,
    similarity_floor: float = 0.0,
):
    fp = _pool_fingerprint(pool, contract.tier_id, f"frontier_{similarity_floor}")
    cache = st.session_state.setdefault(_FRONTIER_CACHE, {})
    if fp not in cache:
        _lru_put(
            cache,
            fp,
            frontier_and_nash(
                pool,
                contract,
                similarity_floor=similarity_floor,
                correlation_matrix=corr,
                pair_weight_matrix=pair,
            ),
        )
    return cache[fp]


def _admit_one_prob(
    pool: list[dict[str, Any]],
    contract: PortfolioContract,
    corr: pd.DataFrame | None,
    pair: pd.DataFrame | None,
    risk_weight: float,
) -> float | None:
    if not pool:
        return None
    combo = select_combo_for_pool(pool, contract, corr, pair, risk_weight=risk_weight)
    if not combo:
        return None
    d = decompose(
        combo,
        contract,
        risk_weight=risk_weight,
        correlation_matrix=corr,
        pair_weight_matrix=pair,
        compute_ev=False,
    )
    return 1.0 - d.p_all_reject


_SIMPLICITY_UNI_THRESHOLD = 2


def _count_unis(pool: list[dict[str, Any]]) -> int:
    return len({s.get("university") for s in pool})


def pick_by_similarity(pool: list[dict[str, Any]], max_schools: int) -> list[dict[str, Any]]:
    from src.portfolio.config import max_programs_per_school

    by_uni: dict[str, list[dict[str, Any]]] = {}
    for s in pool:
        uni = s.get("university", "")
        by_uni.setdefault(uni, []).append(s)

    result: list[dict[str, Any]] = []
    for uni, entries in by_uni.items():
        max_n = max_programs_per_school(uni)
        if max_n <= 1:
            best = max(
                entries,
                key=lambda s: (
                    clip_probability_coerce(s.get("probability")),
                    float(s.get("similarity", 0.0)),
                ),
            )
            result.append(best)
        else:
            entries.sort(
                key=lambda s: (
                    clip_probability_coerce(s.get("probability")),
                    float(s.get("similarity", 0.0)),
                ),
                reverse=True,
            )
            result.extend(entries[:max_n])

    result.sort(
        key=lambda s: clip_probability_coerce(s.get("probability")),
        reverse=True,
    )
    return result


def select_combo_for_pool(
    pool: list[dict[str, Any]],
    contract: PortfolioContract,
    corr: pd.DataFrame | None,
    pair: pd.DataFrame | None,
    *,
    risk_weight: float = 1.0,
) -> list[dict[str, Any]]:
    if not pool:
        return []
    if _count_unis(pool) <= _SIMPLICITY_UNI_THRESHOLD:
        return pick_by_similarity(pool, contract.max_schools)
    return cached_recommended(pool, contract, corr, pair, risk_weight=risk_weight)


def admit_one_pct(
    unified: list[dict[str, Any]],
    contract: PortfolioContract,
    corr: pd.DataFrame | None,
    pair: pd.DataFrame | None,
) -> int:
    pool = filter_pool_by_contract(prediction_results_to_schools(unified), contract)
    p = _admit_one_prob(pool, contract, corr, pair, 1.0)
    return prob_to_pct(p)


def planning_uplift(
    unified: list[dict[str, Any]],
    contract: PortfolioContract,
    corr: pd.DataFrame | None,
    pair: pd.DataFrame | None,
) -> dict[str, int]:
    pool = filter_pool_by_contract(prediction_results_to_schools(unified), contract)
    p_opt = _admit_one_prob(pool, contract, corr, pair, 1.0)
    if p_opt is None:
        return {"opt_pct": 0, "base_pct": 0, "delta_pct": 0}
    p_base = _admit_one_prob(pool, contract, corr, pair, 0.0)
    opt = prob_to_pct(p_opt)
    base = opt if p_base is None else prob_to_pct(p_base)
    return {"opt_pct": opt, "base_pct": base, "delta_pct": max(0, opt - base)}


_PORTFOLIO_ADMIT_CACHE_KEY = "_hk_sales_portfolio_admit_cache"


def clear_portfolio_admit_cache() -> None:
    st.session_state.pop(_PORTFOLIO_ADMIT_CACHE_KEY, None)


def _unified_fingerprint(unified: list[dict[str, Any]]) -> str:
    key = [(r.get("university"), r.get("major"), prob_round(r.get("probability"))) for r in unified]
    return hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _selected_portfolio_key(selected: list[str]) -> tuple[str, str, str, str] | None:
    from src.pages.prediction.result_display.product_registry import (
        ProductKind,
        UpliftMode,
        registry_by_name,
    )

    by_name = registry_by_name()
    app = None
    for name in selected:
        p = by_name.get(name)
        if p and p.kind == ProductKind.APPLICATION:
            app = p
            break
    if not app:
        return None
    pipeline_key = "|".join(
        sorted(
            n for n in selected if (p := by_name.get(n)) and p.uplift_mode == UpliftMode.PIPELINE
        )
    )
    return f"{app.catalog_id}|{pipeline_key}", pipeline_key, app.name, app.catalog_id


def portfolio_admit_pending(
    unified_by_combo: dict[str, list[dict[str, Any]]],
    selected: list[str],
) -> bool:
    parsed = _selected_portfolio_key(selected)
    if not parsed:
        return False
    prefix, pipeline_key, _, _ = parsed
    unified = unified_by_combo.get(pipeline_key) or unified_by_combo.get("") or []
    if not unified:
        return False
    pk = f"{prefix}|{_unified_fingerprint(unified)}"
    cache = st.session_state.get(_PORTFOLIO_ADMIT_CACHE_KEY, {})
    return pk not in cache


def ensure_portfolio_admit_cache(
    unified_by_combo: dict[str, list[dict[str, Any]]],
    selected: list[str],
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = st.session_state.setdefault(_PORTFOLIO_ADMIT_CACHE_KEY, {})
    parsed = _selected_portfolio_key(selected)
    if not parsed:
        return _portfolio_cache_for_frontend(cache)

    prefix, pipeline_key, app_name, catalog_id = parsed
    unified = unified_by_combo.get(pipeline_key) or unified_by_combo.get("") or []
    if not unified:
        return _portfolio_cache_for_frontend(cache)

    pk = f"{prefix}|{_unified_fingerprint(unified)}"
    if pk not in cache:
        contract = PortfolioContract.from_tier(catalog_id)
        corr, pair = load_correlation()
        up = planning_uplift(unified, contract, corr, pair)
        cache[pk] = {
            "admit_one_pct": up["opt_pct"],
            "planning_base_pct": up["base_pct"],
            "planning_delta_pct": up["delta_pct"],
            "app_name": app_name,
        }
        if len(cache) > _MAX_CACHE_ENTRIES:
            oldest = next(iter(cache))
            del cache[oldest]
    return _portfolio_cache_for_frontend(cache)


def _portfolio_cache_for_frontend(
    cache: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for k, v in cache.items():
        front = k.rsplit("|", 1)[0] if "|" in k else k
        out[front] = v
    return out
