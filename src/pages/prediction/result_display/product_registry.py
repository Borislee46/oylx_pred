from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.adjustment.config import MIN_SIMILARITY_THRESHOLD
from src.pages.prediction.app_data import load_bg_target_similarity_cache
from src.pages.prediction.core.utils import get_cached_major_similarity, normalize_language_score
from src.utils.contract_config import get_applications, get_upgrades
from src.utils.logger import setup_logger

_logger = setup_logger("page3", "prediction")

_LANG_TARGET_IELTS = 6.5
_LANG_TARGET_TOEFL = 90.0


def _bump_language(d: dict) -> dict:
    out = dict(d)
    lang_type = out.get("language_type") or "雅思"
    target_raw = _LANG_TARGET_TOEFL if lang_type == "托福" else _LANG_TARGET_IELTS
    cur_raw = float(out.get("language_score_raw") or 0)
    if cur_raw >= target_raw:
        return out
    out["language_score_raw"] = target_raw
    out["language_score"] = normalize_language_score(target_raw, lang_type)
    return out


def _bump_internship(d: dict) -> dict:
    out = dict(d)
    out["internship_count"] = int(out.get("internship_count") or 0) + 1
    return out


def _bump_research(d: dict) -> dict:
    out = dict(d)
    out["research_count"] = int(out.get("research_count") or 0) + 1
    return out


def _diag_language(d: dict) -> str | None:
    lt = d.get("language_type") or "雅思"
    score = d.get("language_score_raw") or d.get("language_score")
    if score is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if lt == "雅思" and s < 6.5:
        return f"当前{s:.1f} -> 目标6.5"
    if lt == "托福" and s < 90:
        return f"当前{int(s)} -> 目标90"
    return None


def _diag_language_severity(d: dict) -> float:
    lt = d.get("language_type") or "雅思"
    score = d.get("language_score_raw") or d.get("language_score")
    if score is None:
        return 0.0
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.0
    if lt == "雅思":
        gap = max(0.0, 6.5 - s)
        return min(1.0, gap / 3.0)  # 3.5 → 6.5 = 1.0; 6.0 → 6.5 = 0.17
    if lt == "托福":
        gap = max(0.0, 90.0 - s)
        return min(1.0, gap / 60.0)  # 30 → 90 = 1.0; 80 → 90 = 0.17
    return 0.0


def _diag_internship(d: dict) -> str | None:
    n = int(d.get("internship_count") or 0)
    if n == 0:
        return "缺实习"
    if n == 1:
        return "实习偏少（仅1段）"
    return None


def _diag_internship_severity(d: dict) -> float:
    n = int(d.get("internship_count") or 0)
    if n == 0:
        return 0.8
    if n == 1:
        return 0.3
    return 0.0


def _diag_research(d: dict) -> str | None:
    n = int(d.get("research_count") or 0)
    if n == 0:
        return "缺科研"
    if n == 1:
        return "科研偏少（仅1段）"
    return None


def _diag_research_severity(d: dict) -> float:
    n = int(d.get("research_count") or 0)
    if n == 0:
        return 0.7
    if n == 1:
        return 0.2
    return 0.0


def _diag_tutoring(d: dict) -> str | None:
    bg_major = d.get("background_major") or ""
    targets = d.get("target_majors") or d.get("_all_majors_target") or []
    if not bg_major or not targets:
        return None
    cache = load_bg_target_similarity_cache()
    for m in targets:
        if get_cached_major_similarity(str(m), bg_major, cache) < MIN_SIMILARITY_THRESHOLD:
            return "跨专业"
    return None


def _diag_tutoring_severity(d: dict) -> float:
    bg_major = d.get("background_major") or ""
    targets = d.get("target_majors") or d.get("_all_majors_target") or []
    if not bg_major or not targets:
        return 0.0
    cache = load_bg_target_similarity_cache()
    min_sim = 1.0
    for m in targets:
        sim = get_cached_major_similarity(str(m), bg_major, cache)
        if sim < min_sim:
            min_sim = sim
    if min_sim >= MIN_SIMILARITY_THRESHOLD:
        return 0.0
    return min(1.0, (MIN_SIMILARITY_THRESHOLD - min_sim) / MIN_SIMILARITY_THRESHOLD)


def _diag_always(_d: dict) -> str | None:
    return ""


def _diag_always_severity(_d: dict) -> float:
    return 0.0


_DIAG_MAP: dict[str, Callable[[dict], str | None]] = {
    "language_below_target": _diag_language,
    "no_internship": _diag_internship,
    "no_research": _diag_research,
    "cross_major": _diag_tutoring,
    "always": _diag_always,
}

_DIAG_SEVERITY_MAP: dict[str, Callable[[dict], float]] = {
    "language_below_target": _diag_language_severity,
    "no_internship": _diag_internship_severity,
    "no_research": _diag_research_severity,
    "cross_major": _diag_tutoring_severity,
    "always": _diag_always_severity,
}

_APPLY_MAP: dict[str, Callable[[dict], dict]] = {
    "language_below_target": _bump_language,
    "no_internship": _bump_internship,
    "no_research": _bump_research,
}


_C9_UNIS: frozenset[str] = frozenset(
    {
        "北京大学",
        "清华大学",
        "浙江大学",
        "上海交通大学",
        "复旦大学",
        "南京大学",
        "中国科学技术大学",
        "哈尔滨工业大学",
        "西安交通大学",
    }
)


def _resolve_school_tier(university_name: str) -> str:
    name = str(university_name or "")
    if not name:
        return "其他"
    if name in _C9_UNIS:
        return "C9"
    if "985" in name or name.endswith("985"):
        return "985"
    if "211" in name or name.endswith("211"):
        return "211"
    if "大学" in name or "学院" in name:
        return "双非"
    return "其他"


def _language_gap(base_input: dict) -> float:
    lt = base_input.get("language_type") or "雅思"
    score = base_input.get("language_score_raw") or base_input.get("language_score")
    if score is None:
        return 0.0
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.0
    if lt and "托福" in str(lt):
        return max(0.0, (_LANG_TARGET_TOEFL - s) / 30.0)
    return max(0.0, _LANG_TARGET_IELTS - s)


_personalization_cache: dict[str, Any] | None = None


def _load_personalization_config() -> dict[str, Any]:
    global _personalization_cache
    if _personalization_cache is not None:
        return _personalization_cache
    from src.utils.contract_config import _load as _load_contract

    cfg = _load_contract()
    pers = cfg.get("personalization")
    if not isinstance(pers, dict):
        _personalization_cache = {
            "school_tier": {"其他": 1.0},
            "contract_tier": {"default": 1.0},
            "gap_amplification": {"language": {"per_point_gap": 0.0, "max": 1.0}},
            "synergy": {"2_products": 1.0, "3_plus_products": 1.0},
            "global_cap": 1.0,
        }
    else:
        _personalization_cache = pers
    return _personalization_cache


def personalize_fixed_pp(
    product_id: str,
    base_pp: int,
    base_input: dict,
    contract_tier: str = "",
    n_upgrades: int = 0,
) -> tuple[int, dict[str, object]]:
    if base_pp <= 0:
        return 0, {"base": base_pp, "final": 0}
    cfg = _load_personalization_config()
    m = 1.0

    tier = _resolve_school_tier(str(base_input.get("background_university", "")))
    school_coef = float(cfg.get("school_tier", {}).get(tier, 1.12))
    m *= school_coef

    ct_cfg = cfg.get("contract_tier", {})
    contract_coef = float(ct_cfg.get(contract_tier, ct_cfg.get("default", 1.0)))
    m *= contract_coef

    gap_coef = 1.0
    if product_id == "english":
        gap = _language_gap(base_input)
        if gap > 0:
            lang_cfg = cfg.get("gap_amplification", {}).get("language", {})
            step = float(lang_cfg.get("per_point_gap", 0.10))
            cap = float(lang_cfg.get("max", 1.25))
            gap_coef = min(cap, 1.0 + gap * step)
            m *= gap_coef

    synergy_coef = 1.0
    synergy_cfg = cfg.get("synergy", {})
    if n_upgrades >= 3:
        synergy_coef = float(synergy_cfg.get("3_plus_products", 1.08))
        m *= synergy_coef
    elif n_upgrades == 2:
        synergy_coef = float(synergy_cfg.get("2_products", 1.05))
        m *= synergy_coef

    global_cap = float(cfg.get("global_cap", 2.0))
    m = min(m, global_cap)

    personalized = max(1, round(base_pp * m))
    breakdown: dict[str, object] = {
        "base": base_pp,
        "school_tier": round(school_coef, 2),
        "contract_tier": round(contract_coef, 2),
        "gap": round(gap_coef, 2) if gap_coef != 1.0 else None,
        "synergy": round(synergy_coef, 2) if synergy_coef != 1.0 else None,
        "final": personalized,
    }
    return personalized, breakdown


class ProductKind(StrEnum):
    APPLICATION = "application"
    UPGRADE = "upgrade"


class UpliftMode(StrEnum):
    PIPELINE = "pipeline"
    FIXED = "fixed"
    NONE = "none"


@dataclass(frozen=True)
class SlotDef:
    id: str
    kind: ProductKind
    label: str
    hint: str
    max_select: int


SLOT_DEFS: tuple[SlotDef, ...] = (
    SlotDef("application", ProductKind.APPLICATION, "申请方案", "拖入一种申请服务（仅一项）", 1),
    SlotDef("upgrade", ProductKind.UPGRADE, "升级产品", "可拖入多项提升产品", 0),
)


@dataclass(frozen=True)
class ProductDef:
    catalog_id: str
    name: str
    kind: ProductKind
    uplift_mode: UpliftMode
    tooltip: str
    bottleneck: str = ""
    narrative: str = ""
    fixed_pp: int = 0
    priority: str = "secondary"
    bonus: str = ""
    sort_order: int = 100
    diag_rule: str = ""
    diagnose: Callable[[dict], str | None] | None = None
    severity: Callable[[dict], float] | None = None
    apply: Callable[[dict], dict] | None = None
    dot: str = "#94a3b8"

    @property
    def affects_probability(self) -> bool:
        return self.uplift_mode == UpliftMode.PIPELINE


def _build_application_def(app_id: str, cfg: dict) -> ProductDef:
    return ProductDef(
        catalog_id=app_id,
        name=cfg.get("name", app_id),
        kind=ProductKind.APPLICATION,
        uplift_mode=UpliftMode.NONE,
        tooltip=cfg.get("description", ""),
        bottleneck="方案",
        narrative="、".join(cfg.get("services", [])) if cfg.get("services") else "",
        priority=cfg.get("priority", "secondary"),
        bonus=cfg.get("bonus", ""),
        sort_order=cfg.get("sort", 100),
        dot=cfg.get("dot", "#94a3b8"),
    )


def _build_upgrade_def(up_id: str, cfg: dict) -> ProductDef:
    uplift_mode = UpliftMode(cfg.get("uplift_mode", "fixed"))
    diag_rule = cfg.get("diag_rule", "")
    apply_fn = _APPLY_MAP.get(diag_rule) if uplift_mode == UpliftMode.PIPELINE else None
    return ProductDef(
        catalog_id=up_id,
        name=cfg.get("name", up_id),
        kind=ProductKind.UPGRADE,
        uplift_mode=uplift_mode,
        tooltip=cfg.get("tooltip", ""),
        bottleneck=cfg.get("bottleneck", ""),
        narrative=cfg.get("narrative", ""),
        fixed_pp=cfg.get("fixed_pp", 0),
        priority=cfg.get("priority", "secondary"),
        sort_order=cfg.get("sort", 100),
        diag_rule=diag_rule,
        diagnose=_DIAG_MAP.get(diag_rule),
        severity=_DIAG_SEVERITY_MAP.get(diag_rule),
        apply=apply_fn,
        dot=cfg.get("dot", "#94a3b8"),
    )


def _build_product_defs() -> tuple[ProductDef, ...]:
    apps = get_applications()
    upgs = get_upgrades()
    defs: list[ProductDef] = []
    for app_id, cfg in apps.items():
        defs.append(_build_application_def(app_id, cfg))
    for up_id, cfg in upgs.items():
        defs.append(_build_upgrade_def(up_id, cfg))
    _logger.info(
        "_build_product_defs: %d apps + %d upgrades = %d total products",
        len(apps),
        len(upgs),
        len(defs),
    )
    return tuple(defs)


PRODUCTS_DEF: tuple[ProductDef, ...] = _build_product_defs()

_HIGHLIGHT_IDS: frozenset[str] = frozenset(
    {
        "premium_b",
        "premium_c",
        "bg_research",
        "research_r_plan",
        "academic_tutoring",
        "research_global",
        "bg_intern",
        "dual_excel",
        "butler_plan",
        "campus_recruit",
    }
)


def registry_by_name() -> dict[str, ProductDef]:
    return {p.name: p for p in PRODUCTS_DEF}


def pipeline_product_names() -> list[str]:
    return [p.name for p in PRODUCTS_DEF if p.uplift_mode == UpliftMode.PIPELINE]


def slot_defs_payload() -> list[dict[str, Any]]:
    from src.pages.prediction.result_display.sales_recommendation import load_sales_blocks_policy

    slot_labels = load_sales_blocks_policy().get("slot_labels", {})
    return [
        {
            "id": s.id,
            "kind": s.kind.value,
            "label": slot_labels.get(s.id, s.label),
            "hint": s.hint,
            "max_select": s.max_select,
        }
        for s in SLOT_DEFS
    ]


def synergy_config_payload() -> dict[str, float]:
    syn = _load_personalization_config().get("synergy", {})
    return {
        "two_products": float(syn.get("2_products", 1.0)),
        "three_plus": float(syn.get("3_plus_products", 1.0)),
    }


def synergy_multiplier(n_upgrades: int) -> float:
    if n_upgrades >= 3:
        return synergy_config_payload()["three_plus"]
    if n_upgrades == 2:
        return synergy_config_payload()["two_products"]
    return 1.0


def build_personalized_fixed_pp_map(base_input: dict, contract_tier: str = "") -> dict[str, int]:
    out: dict[str, int] = {}
    for p in PRODUCTS_DEF:
        if p.kind != ProductKind.UPGRADE or p.uplift_mode != UpliftMode.FIXED:
            continue
        pp, _ = personalize_fixed_pp(p.catalog_id, p.fixed_pp, base_input, contract_tier, 0)
        out[p.name] = pp
    return out


def contract_tier_from_selection(selected: list[str], fallback: str = "") -> str:
    by_name = registry_by_name()
    for name in normalize_selection(selected):
        p = by_name.get(name)
        if p and p.kind == ProductKind.APPLICATION:
            return p.catalog_id
    return fallback


def normalize_selection(names: list[str]) -> list[str]:
    by_name = registry_by_name()
    app: str | None = None
    upgrades: list[str] = []
    for name in names:
        p = by_name.get(name)
        if not p:
            upgrades.append(name)
            continue
        if p.kind == ProductKind.APPLICATION:
            app = name
        elif name not in upgrades:
            upgrades.append(name)
    out: list[str] = []
    if app:
        out.append(app)
    out.extend(upgrades)
    return out


def selectable_product_names(base_input: dict, ai_products: list[dict] | None = None) -> list[str]:
    by_name = registry_by_name()
    ai_names = [p.get("name") for p in (ai_products or []) if p.get("name") in by_name]
    ordered = list(dict.fromkeys(ai_names))

    def _sort_key(name: str) -> tuple[int, int]:
        p = by_name.get(name)
        if not p:
            return (2, 999)
        if p.priority == "primary":
            return (0, p.sort_order)
        return (1, p.sort_order)

    for p in sorted(PRODUCTS_DEF, key=lambda x: _sort_key(x.name)):
        if p.name in ordered:
            continue
        if p.priority == "primary":
            ordered.append(p.name)
        elif p.diagnose and p.diagnose(base_input):
            ordered.append(p.name)

    for p in sorted(PRODUCTS_DEF, key=lambda x: _sort_key(x.name)):
        if p.name not in ordered:
            ordered.append(p.name)
    return ordered


def resolve_attribution(
    selected: list[str],
    *,
    base_pct: int,
    combos: dict[str, Any],
    personalized_fixed_pp: dict[str, int] | None = None,
) -> dict[str, object]:
    from src.pages.prediction.result_display.uplift_attribution import attribute_selection

    by_name = registry_by_name()
    base_prob = float(combos.get("", {}).get("best_prob", base_pct / 100.0))
    pipeline_names = {p.name for p in PRODUCTS_DEF if p.uplift_mode == UpliftMode.PIPELINE}
    upgrades = [n for n in selected if (p := by_name.get(n)) and p.kind == ProductKind.UPGRADE]
    _override = personalized_fixed_pp or {}
    syn = synergy_multiplier(len(upgrades))
    fixed_pp: dict[str, float] = {}
    for n in upgrades:
        p = by_name.get(n)
        if not p or p.uplift_mode != UpliftMode.FIXED:
            continue
        base_pp = float(_override[n]) if n in _override else float(p.fixed_pp)
        fixed_pp[n] = float(max(1, round(base_pp * syn))) if syn != 1.0 else base_pp

    def pipeline_prob_of(subset: frozenset[str]) -> float:
        key = "|".join(sorted(n for n in subset if n in pipeline_names))
        return float(combos.get(key, {}).get("best_prob", base_prob))

    return attribute_selection(
        upgrades,
        base_prob=base_prob,
        pipeline_prob_of=pipeline_prob_of,
        fixed_pp=fixed_pp,
        pipeline_names=pipeline_names,
    )


def resolve_display_pct(
    selected: list[str],
    *,
    base_pct: int,
    combos: dict[str, Any],
    personalized_fixed_pp: dict[str, int] | None = None,
) -> int:
    return int(
        resolve_attribution(
            selected,
            base_pct=base_pct,
            combos=combos,
            personalized_fixed_pp=personalized_fixed_pp,
        )["final_pct"]
    )


def build_block_configs(base_input: dict, *, contract_tier: str = "") -> list[dict[str, Any]]:
    import hashlib
    import json

    import streamlit as st

    _cache_key_fields = [
        "background_university",
        "background_major",
        "gpa",
        "language_score",
        "language_score_raw",
        "language_type",
        "internship_count",
        "research_count",
    ]
    _key_data = {k: base_input.get(k) for k in _cache_key_fields}
    _key_data["_target_majors"] = base_input.get("_all_majors_target") or base_input.get(
        "target_majors", []
    )
    _key_data["_contract"] = contract_tier
    fp = hashlib.md5(json.dumps(_key_data, sort_keys=True, default=str).encode()).hexdigest()

    cache = st.session_state.setdefault("_hk_block_configs_cache", {})
    if fp in cache:
        return cache[fp]

    blocks = []

    for name in selectable_product_names(base_input):
        p = registry_by_name().get(name)
        if not p:
            continue
        personalized_pp: int = p.fixed_pp
        pp_breakdown: dict[str, object] | None = None
        if p.kind == ProductKind.UPGRADE and p.uplift_mode == UpliftMode.FIXED:
            personalized_pp, pp_breakdown = personalize_fixed_pp(
                p.catalog_id, p.fixed_pp, base_input, contract_tier, 0
            )
        blocks.append(
            {
                "id": p.catalog_id,
                "name": p.name,
                "kind": p.kind.value,
                "uplift_mode": p.uplift_mode.value,
                "fixed_pp": personalized_pp,
                "fixed_pp_base": p.fixed_pp,
                "dot": p.dot,
                "tooltip": p.tooltip,
                "diag": (p.diagnose(base_input) if p.diagnose else "") or "",
                "narrative": p.narrative if p.uplift_mode != UpliftMode.PIPELINE else "",
                "bonus": p.bonus,
                "priority": p.priority,
                "pp_breakdown": pp_breakdown,
                "highlight": p.catalog_id in _HIGHLIGHT_IDS,
            }
        )
    if len(cache) >= 4:
        oldest = next(iter(cache))
        del cache[oldest]
    cache[fp] = blocks
    return blocks
