import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.adjustment.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.pages.prediction.core.utils import get_background_faculty
from src.pages.prediction.flow.pipeline import run_prediction_pipeline_with_progress
from src.pages.prediction.result_display.product_registry import (
    PRODUCTS_DEF,
    pipeline_product_names,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce, prob_to_pct

logger = setup_logger("page3", "prediction")

BAND_STABLE_MIN = 0.55
BAND_MID_MIN = 0.40


@dataclass
class ProductScenario:
    affects_probability: bool
    bottleneck: str
    apply: Callable[[dict], dict] | None = None
    narrative: str = ""


PRODUCT_SCENARIOS: dict[str, ProductScenario] = {
    p.name: ProductScenario(
        affects_probability=p.affects_probability,
        bottleneck=p.bottleneck,
        apply=p.apply,
        narrative=p.narrative,
    )
    for p in PRODUCTS_DEF
}


def band_of(prob: float) -> str:
    if prob >= BAND_STABLE_MIN:
        return "稳"
    if prob >= BAND_MID_MIN:
        return "偏稳"
    return "冲"


def summarize(unified: list[dict]) -> dict[str, Any]:
    probs = [clip_probability_coerce(r.get("probability")) for r in unified]
    bands = {"稳": 0, "偏稳": 0, "冲": 0}
    for p in probs:
        bands[band_of(p)] += 1
    return {"best_prob": max(probs, default=0.0), "bands": bands, "n": len(probs)}


def affecting_products(product_names: list[str]) -> list[str]:
    return [n for n in product_names if (sc := PRODUCT_SCENARIOS.get(n)) and sc.affects_probability]


def _log_lang_change(before: dict, after: dict) -> None:
    lt = before.get("language_type") or "雅思"
    old_raw = float(before.get("language_score_raw") or before.get("language_score") or 0)
    new_raw = float(after.get("language_score_raw") or after.get("language_score") or old_raw)
    if abs(new_raw - old_raw) > 0.001:
        logger.info(
            "销售语言调整 | %s: %.1f→%.1f (模拟培训前提)",
            lt,
            old_raw,
            new_raw,
        )
    else:
        logger.debug("销售语言调整 | 无需调整 (%.1f)", old_raw)


def apply_products(base_input: dict, product_names: list[str]) -> dict:
    out = dict(base_input)
    for name in product_names:
        sc = PRODUCT_SCENARIOS.get(name)
        if sc and sc.affects_probability and sc.apply:
            out = sc.apply(out)
    return out


def scenario_fingerprint(base_input: dict, product_names: list[str]) -> str:
    key = {
        "u": base_input.get("background_university"),
        "m": base_input.get("background_major"),
        "gpa": base_input.get("gpa"),
        "lang": base_input.get("language_score"),
        "intern": base_input.get("internship_count"),
        "research": base_input.get("research_count"),
        "products": sorted(affecting_products(product_names)),
    }
    return hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def sync_sales_sim_cache(
    base_input: dict,
    selected: list[str],
    precomputed: dict[str, Any],
) -> None:
    import streamlit as st

    from src.pages.prediction.result_display.product_meta import SIM_CACHE_KEY

    affecting = affecting_products(selected)
    if not affecting:
        return
    combo_key = "|".join(sorted(affecting))
    upgraded = precomputed.get("unified_by_combo", {}).get(combo_key)
    if not upgraded:
        return
    fp = scenario_fingerprint(base_input, selected)
    cache = st.session_state.setdefault(SIM_CACHE_KEY, {})
    cache[fp] = upgraded


def recompute_unified(
    page_state: Any, base_input: dict, unified_results: list[dict], product_names: list[str]
) -> list[dict]:
    affecting = affecting_products(product_names)
    if not affecting:
        return unified_results

    logger.info(
        "销售场景重算开始 | products=%s",
        affecting,
    )
    modified = apply_products(base_input, affecting)
    _log_lang_change(base_input, modified)
    return run_recompute_combos(page_state, base_input, unified_results, modified)


def run_recompute_combos(
    page_state: Any,
    base_input: dict,
    unified_results: list[dict],
    modified: dict,
) -> list[dict]:
    combos = [
        (str(r.get("university", "")), str(r.get("major", "")))
        for r in unified_results
        if r.get("university") and r.get("major")
    ]
    if not combos:
        logger.warning("场景重算 | 无院校-专业组合可供重算")
        return unified_results

    modified["_all_universities_target"] = sorted({u for u, _ in combos})
    modified["_all_majors_target"] = sorted({m for _, m in combos})
    modified["_cross_faculty_confirmed"] = True
    modified["_has_valid_experience"] = bool(base_input.get("_has_valid_experience", False))

    cases_df = page_state.cases_df
    bg_major = modified.get("background_major")
    model = run_prediction_pipeline_with_progress(
        modified,
        "xgboost",
        page_state.cases_df_fingerprint,
        page_state.loaded_feature_names,
        progress_cb=None,
        background_faculty=get_background_faculty(bg_major, cases_df),
        admitted_combinations=get_admitted_combinations_from_dataframe(cases_df, bg_major),
        page_state=page_state,
        cached_combinations=combos,
    )
    result = getattr(model, "unified_results", None) or unified_results
    logger.info("场景重算完成 | n_results=%d", len(result))
    return result


def school_deltas(before: list[dict], after: list[dict]) -> list[dict]:
    after_map = {
        (str(r.get("university", "")), str(r.get("major", ""))): clip_probability_coerce(
            r.get("probability")
        )
        for r in after
    }
    out = []
    for r in before:
        key = (str(r.get("university", "")), str(r.get("major", "")))
        old = clip_probability_coerce(r.get("probability"))
        new = after_map.get(key)
        if new is None:
            continue
        delta = new - old
        if delta > 0.002:
            out.append(
                {
                    "university": key[0],
                    "major": key[1],
                    "old": old,
                    "new": new,
                    "delta": delta,
                    "band_change": band_of(old) != band_of(new),
                }
            )
    out.sort(key=lambda x: x["delta"], reverse=True)
    return out


def uplift_range(deltas: list[dict]) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    vals = [d["delta"] for d in deltas]
    return min(vals), max(vals)


AFFECTING_PRODUCT_NAMES = pipeline_product_names()


_PRECOMPUTE_CACHE_KEY = "_hk_sales_precompute_cache"


def _precompute_cache_fp(base_input: dict, unified_results: list[dict]) -> str:
    school_combos = sorted(
        (str(r.get("university", "")), str(r.get("major", ""))) for r in unified_results
    )
    return hashlib.md5(
        json.dumps(
            [scenario_fingerprint(base_input, AFFECTING_PRODUCT_NAMES), school_combos],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _init_precompute_shell(unified_results: list[dict]) -> dict[str, Any]:
    base_sum = summarize(unified_results)
    return {
        "base_pct": prob_to_pct(base_sum["best_prob"]),
        "base_bands": base_sum["bands"],
        "combos": {
            "": {
                "products": [],
                "best_prob": base_sum["best_prob"],
                "bands": base_sum["bands"],
                "best_pct": prob_to_pct(base_sum["best_prob"]),
            }
        },
        "unified_by_combo": {"": unified_results},
    }


def required_pipeline_combo_keys(selected: list[str]) -> set[str]:
    from itertools import combinations

    affecting = affecting_products(selected)
    if not affecting:
        return {""}
    keys: set[str] = set()
    for r in range(len(affecting) + 1):
        for subset in combinations(affecting, r):
            keys.add("|".join(sorted(subset)))
    return keys


def pipeline_combos_pending(precomputed: dict[str, Any], selected: list[str]) -> bool:
    return not required_pipeline_combo_keys(selected).issubset(precomputed.get("combos", {}))


def ensure_pipeline_combos(
    precomputed: dict[str, Any],
    page_state: Any,
    base_input: dict,
    unified_results: list[dict],
    selected: list[str],
) -> None:
    combos = precomputed.setdefault("combos", {})
    unified_by_combo = precomputed.setdefault("unified_by_combo", {})
    missing = [k for k in sorted(required_pipeline_combo_keys(selected)) if k not in combos]
    if not missing:
        return

    logger.info(
        "销售预计算 | 懒算 | 选中=%s 缺失=%d 键=%s",
        selected,
        len(missing),
        missing,
    )
    for key in missing:
        names = key.split("|") if key else []
        after = recompute_unified(page_state, base_input, unified_results, names)
        summary = summarize(after)
        combos[key] = {
            "products": names,
            "best_prob": summary["best_prob"],
            "bands": summary["bands"],
            "best_pct": prob_to_pct(summary["best_prob"]),
        }
        unified_by_combo[key] = after


def precompute_product_combinations(
    page_state: Any,
    base_input: dict,
    unified_results: list[dict],
    *,
    contract_tier: str = "",
) -> dict[str, Any]:
    import streamlit as st

    from src.pages.prediction.result_display.product_registry import (
        build_personalized_fixed_pp_map,
    )

    fp = _precompute_cache_fp(base_input, unified_results) + "|" + (contract_tier or "")
    cache = st.session_state.setdefault(_PRECOMPUTE_CACHE_KEY, {})
    if fp in cache:
        logger.debug("销售预计算 | 命中缓存 | 指纹=%s", fp[:16])
        return cache[fp]

    result = _init_precompute_shell(unified_results)
    result["personalized_fixed_pp"] = build_personalized_fixed_pp_map(base_input, contract_tier)
    if len(cache) >= 8:
        oldest = next(iter(cache))
        del cache[oldest]
        logger.debug("销售预计算 | 移除最旧条目 | 缓存大小=%d", len(cache))
    cache[fp] = result
    logger.info(
        "销售预计算初始化 | 基准百分比=%d%% (pipeline 子集懒算)",
        result["base_pct"],
    )
    return result
