from __future__ import annotations

from src.agent.schemas import compute_tiers
from src.utils.numeric import clip_probability_coerce

_TIER_DISPLAY = {"适中": "目标"}


def tier_display_label(label: str | None) -> str:
    if not label:
        return "冲刺"
    return _TIER_DISPLAY.get(label, label)


def attach_canonical_tier_labels(unified_results: list[dict]) -> list[dict]:
    if not unified_results:
        return []

    best: dict[str, float] = {}
    for r in unified_results:
        uni = str(r.get("university", "") or "").strip()
        if not uni:
            continue
        prob = clip_probability_coerce(r.get("probability"))
        if uni not in best or prob > best[uni]:
            best[uni] = prob

    if not best:
        return [dict(r) for r in unified_results]

    schools = sorted(best.keys(), key=lambda u: -best[u])
    tier_by_school = dict(zip(schools, compute_tiers([best[u] for u in schools]), strict=False))

    labeled: list[dict] = []
    for r in unified_results:
        row = dict(r)
        uni = str(r.get("university", "") or "").strip()
        if uni in tier_by_school:
            row["label"] = tier_by_school[uni]
        labeled.append(row)
    return labeled
