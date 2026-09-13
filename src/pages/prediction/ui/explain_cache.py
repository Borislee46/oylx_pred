from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from src.agent.explain_profiles import ProfileType, classify_profile
from src.utils.logger import setup_logger
from src.utils.numeric import prob_round

_explain_logger = setup_logger("page3", "prediction")

_EXPLAIN_CACHE_DIR = Path(__file__).resolve().parents[4] / "cache" / "explain"
_EXPLAIN_CACHE_MAX = 50
_EXPLAIN_CACHE_TTL = 30 * 60


def _explain_cache_path() -> str:
    return os.path.join(_EXPLAIN_CACHE_DIR, "explain.json")


def load_explain_cache() -> dict[str, Any]:
    try:
        path = _explain_cache_path()
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        now = time.time()
        return {k: v for k, v in data.items() if now - v.get("_ts", 0) < _EXPLAIN_CACHE_TTL}
    except Exception:
        _explain_logger.exception("explain 缓存读取失败")
        return {}


def persist_explain_cache(cache: dict[str, Any]) -> None:
    try:
        os.makedirs(_EXPLAIN_CACHE_DIR, exist_ok=True)
        keys = sorted(cache.keys(), key=lambda k: cache[k].get("_ts", 0))
        while len(keys) > _EXPLAIN_CACHE_MAX:
            oldest = keys.pop(0)
            del cache[oldest]
        path = _explain_cache_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        _explain_logger.exception("explain 缓存写入失败")


_PORTFOLIO_FP_KEY = "_explain_portfolio_fp"
_PDF_PORTFOLIO_FP_KEY = "_hk_pdf_portfolio_fp"
_PATHFINDER_PDF_KEYS = ("_hk_pdf_bytes", "_hk_pdf_name", "_hk_pdf_error", _PDF_PORTFOLIO_FP_KEY)


def portfolio_combo_fingerprint(combo: list | None) -> str:
    if not combo:
        return ""
    payload = compact_results(combo)
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def invalidate_pathfinder_pdf() -> None:
    import streamlit as st

    for key in _PATHFINDER_PDF_KEYS:
        st.session_state.pop(key, None)


def compact_results(items: list[dict]) -> list[tuple[str, str, float]]:
    return [
        (
            str(item.get("university", "")),
            str(item.get("major", "")),
            prob_round(item.get("probability")),
        )
        for item in items
        if isinstance(item, dict)
    ]


def compact_experience(experience_details: dict | None) -> dict[str, int]:
    return {str(k): len(str(v or "")) for k, v in (experience_details or {}).items() if v}


def build_explain_cache_key(
    *,
    sim: list,
    cross: list,
    unified: list,
    background_major: str,
    background_major_2: str = "",
    is_dual_degree: bool = False,
    gpa: float,
    gpa_raw: float = 0.0,
    exam_type: str = "",
    exam_score: float = 0.0,
    language_score: float = 0.0,
    language_type: str = "",
    experience_details: dict | None = None,
    profile: ProfileType | None = None,
    portfolio_combo: list | None = None,
    blocks_selection: list[str] | None = None,
    display_pct: int | None = None,
    contract_tier: str = "",
) -> str:
    if profile is None:
        profile = classify_profile(
            {"similarity_results": sim, "cross_major_results": cross, "unified_results": unified}
        )
    key_data = {
        "v": 9,
        "profile": profile,
        "sim": compact_results(sim[:5]),
        "cross": compact_results(cross[:3]),
        "unified": compact_results(unified[:5]),
        "background_major": background_major,
        "background_major_2": background_major_2,
        "is_dual_degree": is_dual_degree,
        "gpa": gpa,
        "gpa_raw": gpa_raw,
        "exam": [exam_type, exam_score] if exam_type and exam_score > 0 else [],
        "language": [language_type, language_score],
        "experience": compact_experience(experience_details),
        "portfolio": compact_results((portfolio_combo or [])[:10]),
        "blocks": sorted(blocks_selection or []),
        "display_pct": display_pct,
        "contract_tier": contract_tier,
    }
    return hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()
