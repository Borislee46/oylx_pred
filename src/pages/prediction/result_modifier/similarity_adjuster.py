from __future__ import annotations

import json
from functools import lru_cache
from typing import Any


def _normalize_keywords(items: list[str]) -> list[str]:
    normalized: list[str] = []
    for it in items or []:
        if not isinstance(it, str):
            continue
        parts = []
        for seg in it.split("\n"):
            parts.extend(seg.split(","))
        for p in parts:
            s = p.strip().lower()
            if s:
                normalized.append(s)
    return normalized


@lru_cache(maxsize=1)
def _load_similarity_rules(
    config_path: str = "config/similarity_adjustment_rules.json",
) -> list[dict[str, Any]]:
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        rules = data.get("rules", [])
        enabled_rules: list[dict[str, Any]] = []
        for r in rules:
            if not isinstance(r, dict) or not r.get("enabled", False):
                continue
            enabled_rules.append(
                {
                    "background_keywords": _normalize_keywords(r.get("background_keywords", [])),
                    "target_keywords": _normalize_keywords(r.get("target_keywords", [])),
                    "adjustment": float(r.get("adjustment", 0.0)),
                }
            )
        return enabled_rules
    except Exception:
        return []


def adjust_similarity_score(background_major: str, target_major: str, similarity: float) -> float:
    if not background_major or not target_major:
        return similarity

    bg = background_major.lower()
    tgt = target_major.lower()
    adjusted = float(similarity)

    for rule in _load_similarity_rules():
        bks = rule.get("background_keywords", [])
        tks = rule.get("target_keywords", [])
        adj = float(rule.get("adjustment", 0.0))
        if any(k in bg for k in bks) and any(k in tgt for k in tks):
            adjusted += adj
            break

    return max(0.0, min(1.0, adjusted))
