"""Compact writing profile shared with the prediction page."""

from __future__ import annotations

import hashlib
import time
from typing import Any


def build_writing_profile(
    text: str,
    result: dict[str, Any],
    new_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = float(result.get("score", 0) or 0)
    after_score = _score_or_none(new_result)
    top_features = _top_ai_features(result.get("features", {}))
    top_fixes = _top_fixes(result.get("local_fixes", []))

    return {
        "text_hash": hashlib.md5(text.encode()).hexdigest()[:10],
        "score": round(score, 1),
        "risk_level": _risk_level(score),
        "verdict": (result.get("verdict") or {}).get("label", ""),
        "estimated_new_score": result.get("estimated_new_score"),
        "after_rewrite_score": after_score,
        "improvement": round(score - after_score, 1) if after_score is not None else None,
        "top_features": top_features,
        "top_fixes": top_fixes,
        "text_stats": result.get("text_stats", {}),
        "updated_at": int(time.time()),
    }


def _score_or_none(result: dict[str, Any] | None) -> float | None:
    if not result or result.get("score") is None:
        return None
    return round(float(result["score"]), 1)


def _top_ai_features(features: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        features.items(),
        key=lambda item: float(item[1].get("contribution", 0) or 0),
        reverse=True,
    )
    return [
        {
            "name": name,
            "label": data.get("label", name),
            "value": data.get("value"),
            "contribution": data.get("contribution", 0),
        }
        for name, data in ranked[:3]
        if float(data.get("contribution", 0) or 0) > 0
    ]


def _top_fixes(fixes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "category": fix.get("category", ""),
            "label": fix.get("label", ""),
            "impact": fix.get("impact", 0),
            "difficulty": fix.get("difficulty", ""),
        }
        for fix in fixes[:3]
    ]


def _risk_level(score: float) -> str:
    if score < 30:
        return "low"
    if score < 60:
        return "medium"
    return "high"
