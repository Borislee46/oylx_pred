from __future__ import annotations

import hashlib
import json
from typing import Any

from src.adjustment.sales_scenario import (
    band_of,
    run_recompute_combos,
    summarize,
)
from src.adjustment.config import BAYESIAN_SHRINKAGE_PRIOR_STRENGTH
from src.pages.prediction.data_facts import N_SAMPLES
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce, prob_to_pct

logger = setup_logger("page3", "prediction")

# what-if 推演：只做离散档位（延迟可控），不做连续滑杆。
GPA_TIERS: tuple[float, ...] = (3.2, 3.4, 3.6, 3.8)

GPA_CACHE_KEY = "_hk_gpa_scenario_cache"
GPA_CACHE_MAX = 8


def gpa_tier_label(gpa: float) -> str:
    return f"GPA {gpa:.1f}"


def gpa_tier_key(gpa: float) -> str:
    return f"gpa_{gpa:.1f}"


def gpa_scenario_fingerprint(
    base_input: dict,
    gpa: float,
    unified_results: list[dict],
) -> str:
    """GPA 档位场景指纹：背景画像 + 档位 + 目标组合（决定重算范围）。"""
    combos = sorted(
        (str(r.get("university", "")), str(r.get("major", "")))
        for r in unified_results
        if r.get("university") and r.get("major")
    )
    key = {
        "u": base_input.get("background_university"),
        "m": base_input.get("background_major"),
        "gpa": round(float(gpa), 2),
        "lang": base_input.get("language_score"),
        "lang_type": base_input.get("language_type"),
        "intern": base_input.get("internship_count"),
        "research": base_input.get("research_count"),
        "combos": combos,
    }
    return hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def recompute_gpa_tier(
    page_state: Any,
    base_input: dict,
    unified_results: list[dict],
    gpa: float,
) -> list[dict]:
    """把基准输入中的 GPA 替换为档位值后，按同一目标组合重跑 pipeline。"""
    gpa = float(gpa)
    if gpa <= 0:
        logger.warning("GPA 档位重算 | 非法档位 gpa=%s，跳过", gpa)
        return unified_results

    modified = dict(base_input)
    modified["gpa"] = gpa
    modified["gpa_model"] = gpa
    logger.info("GPA 档位重算开始 | gpa=%.2f", gpa)
    return run_recompute_combos(page_state, base_input, unified_results, modified)


def _entry_of(
    page_state: Any,
    base_input: dict,
    unified_results: list[dict],
    gpa: float,
) -> dict[str, Any]:
    fp = gpa_scenario_fingerprint(base_input, gpa, unified_results)
    import streamlit as st

    cache = st.session_state.setdefault(GPA_CACHE_KEY, {})
    entry = cache.get(fp)
    if entry is not None:
        logger.debug("GPA 档位缓存命中 | gpa=%.2f fp=%s", gpa, fp[:12])
        return entry

    after = recompute_gpa_tier(page_state, base_input, unified_results, gpa)
    summary = summarize(after)
    entry = {
        "gpa": gpa,
        "key": gpa_tier_key(gpa),
        "label": gpa_tier_label(gpa),
        "unified": after,
        "best_prob": summary["best_prob"],
        "bands": summary["bands"],
        "best_pct": prob_to_pct(summary["best_prob"]),
    }
    if len(cache) >= GPA_CACHE_MAX:
        oldest = next(iter(cache))
        del cache[oldest]
        logger.debug("GPA 档位缓存 | 移除最旧条目 | 大小=%d", len(cache))
    cache[fp] = entry
    logger.info(
        "GPA 档位重算完成 | gpa=%.2f best_pct=%d%% n_results=%d",
        gpa,
        entry["best_pct"],
        len(after),
    )
    return entry


def ensure_gpa_tier(
    page_state: Any,
    base_input: dict,
    unified_results: list[dict],
    gpa: float,
) -> dict[str, Any]:
    """确保单个档位结果就绪（懒重算 + 指纹缓存），供前端切换时调用。"""
    return _entry_of(page_state, base_input, unified_results, gpa)


def ensure_gpa_tiers(
    page_state: Any,
    base_input: dict,
    unified_results: list[dict],
    tiers: tuple[float, ...] = GPA_TIERS,
) -> dict[str, dict[str, Any]]:
    """批量确保档位结果（预取用）。返回 {gpa_3.2: entry, ...}。"""
    return {gpa_tier_key(g): _entry_of(page_state, base_input, unified_results, g) for g in tiers}


def build_gpa_grid(
    unified_by_tier: dict[str, list[dict]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """把各档位 unified 结果压成前端轴可消费的网格：档位 → 院校 → 最佳概率。"""
    grid: dict[str, dict[str, dict[str, Any]]] = {}
    for tier_key, results in unified_by_tier.items():
        schools: dict[str, dict[str, Any]] = {}
        for r in results:
            uni = str(r.get("university", "")).strip()
            if not uni:
                continue
            p = clip_probability_coerce(r.get("probability"))
            prev = schools.get(uni)
            if prev is None or p > prev["p"]:
                n = int(r.get("_baseline_sample_count", 0) or 0)
                trace = r.get("_adjustment_trace") or {}
                schools[uni] = {
                    "p": p,
                    "tier": band_of(p),
                    "n": n,
                    # 模型审计链：历史基线、KNN 基础概率、各调整项、收缩、最终值
                    "baseline_rate": r.get("_baseline_admit_rate"),
                    "trace": {
                        k: v
                        for k, v in trace.items()
                        if isinstance(v, (int, float))
                    }
                    or None,
                    "counterfactuals": r.get("_counterfactuals"),
                    # 样本量 ≥ 收缩先验强度 → 直接信任模型；否则已做贝叶斯收缩
                    "sample_ok": bool(n >= BAYESIAN_SHRINKAGE_PRIOR_STRENGTH),
                }
        grid[tier_key] = schools
    return grid


def build_gpa_payload(
    entries: dict[str, dict[str, Any]],
    *,
    base_input: dict | None = None,
    all_tiers: tuple[float, ...] = GPA_TIERS,
) -> dict[str, Any]:
    """给前端概率轴组件的完整 payload：档位元数据 + 网格 + 学校列表 + 基准。

    entries 只包含已就绪档位；all_tiers 保证 chips 始终展示全部档位
    （未就绪档位的 best_pct 为 None，前端据此显示"待重算"）。
    """
    grid = build_gpa_grid({k: e["unified"] for k, e in entries.items()})

    # 学校顺序：以最接近输入 GPA 的档位为基准，按概率降序；颜色按名称稳定分配
    if not entries:
        tiers_meta = [
            {
                "key": gpa_tier_key(g),
                "label": gpa_tier_label(g),
                "gpa": g,
                "best_pct": None,
                "best_prob": None,
                "bands": {},
                "ready": False,
            }
            for g in all_tiers
        ]
        return {
            "tiers": tiers_meta,
            "grid": {},
            "schools": [],
            "base_key": "",
            "corpus_n": N_SAMPLES,
        }

    base_key = _base_tier_key(entries, base_input)
    base_schools = grid.get(base_key, {})
    ordered = sorted(base_schools, key=lambda u: base_schools[u]["p"], reverse=True)
    palette = (
        "#38bdf8",
        "#a78bfa",
        "#f472b6",
        "#34d399",
        "#fbbf24",
        "#fb7185",
        "#94a3b8",
        "#60a5fa",
        "#f97316",
        "#2dd4bf",
    )

    def _color_of(name: str) -> str:
        idx = sum(ord(c) for c in name) % len(palette)
        return palette[idx]

    schools = [
        {
            "name": u,
            "color": _color_of(u),
            "n": int(base_schools[u].get("n", 0) or 0),
            "tier": base_schools[u]["tier"],
            "p": base_schools[u]["p"],
        }
        for u in ordered
    ]
    ready_keys = set(entries)
    tiers_meta = []
    for e in entries.values():
        tiers_meta.append(
            {
                "key": e["key"],
                "label": e["label"],
                "gpa": e["gpa"],
                "best_pct": e["best_pct"],
                "best_prob": e["best_prob"],
                "bands": e["bands"],
                "ready": e["key"] in ready_keys,
            }
        )
    missing = [g for g in all_tiers if gpa_tier_key(g) not in ready_keys]
    tiers_meta.extend(
        {
            "key": gpa_tier_key(g),
            "label": gpa_tier_label(g),
            "gpa": g,
            "best_pct": None,
            "best_prob": None,
            "bands": {},
            "ready": False,
        }
        for g in missing
    )
    tiers_meta.sort(key=lambda t: t["gpa"])

    return {
        "tiers": tiers_meta,
        "grid": grid,
        "schools": schools,
        "base_key": base_key,
        "corpus_n": N_SAMPLES,
    }


def _base_tier_key(
    entries: dict[str, dict[str, Any]],
    base_input: dict | None,
) -> str:
    """选出最接近学生输入 GPA 的档位作为基准（无输入时取首个档位）。"""
    gpa = None
    if base_input:
        try:
            gpa = float(base_input.get("gpa") or base_input.get("gpa_model"))
        except (TypeError, ValueError):
            gpa = None
    keys = list(entries)
    if gpa is None or not keys:
        return keys[0] if keys else ""
    return min(keys, key=lambda k: abs(_key_to_gpa(k) - gpa))


def _key_to_gpa(key: str) -> float:
    try:
        return float(key.split("_")[1])
    except (IndexError, ValueError):
        return 0.0
