from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.portfolio.expected_value import (
    EVDecomposition,
    decompose,
)
from src.portfolio.portfolio_contract import PortfolioContract
from src.utils.numeric import clip_probability_coerce

_logger = logging.getLogger(__name__)


def _combo_key(combo: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{s.get('university', '')}|{s.get('major', '')}|"
            f"{s.get('probability')!r}|{s.get('similarity')!r}"
            for s in combo
        )
    )


def dedup_by_university(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from src.portfolio.config import max_programs_per_school

    by_uni: dict[str, list[dict[str, Any]]] = {}
    for s in pool:
        uni = s.get("university", "")
        by_uni.setdefault(uni, []).append(s)

    result: list[dict[str, Any]] = []
    for uni, entries in by_uni.items():
        max_n = max_programs_per_school(uni)
        entries.sort(
            key=lambda s: (
                clip_probability_coerce(s.get("probability")),
                float(s.get("similarity", 0.0)),
            ),
            reverse=True,
        )
        result.extend(entries[:max_n])
    return result


def ev_select(
    pool: list[dict[str, Any]],
    contract: PortfolioContract,
    *,
    risk_weight: float = 1.0,
    similarity_floor: float = 0.0,
    correlation_matrix: pd.DataFrame | None = None,
    pair_weight_matrix: pd.DataFrame | None = None,
    return_decomposition: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], EVDecomposition | None]:
    cands = [s for s in pool if s.get("similarity", 0.0) >= similarity_floor]
    if not cands:
        _logger.warning(
            "ev_select: no school meets similarity_floor=%.2f; falling back to full pool (%d schools)",
            similarity_floor,
            len(pool),
        )
        cands = list(pool)
    cands = dedup_by_university(cands)

    selected: list[dict[str, Any]] = []
    remaining = list(cands)
    decomp_memo: dict[tuple[str, ...], EVDecomposition] = {}

    def _uni_count(combo: list[dict[str, Any]]) -> int:
        return len({s.get("university") for s in combo})

    def decompose_cached(combo: list[dict[str, Any]]) -> EVDecomposition:
        key = _combo_key(combo)
        cached = decomp_memo.get(key)
        if cached is not None:
            return cached
        result = decompose(
            combo,
            contract,
            risk_weight=risk_weight,
            correlation_matrix=correlation_matrix,
            pair_weight_matrix=pair_weight_matrix,
        )
        decomp_memo[key] = result
        return result

    def value_of(combo: list[dict[str, Any]]) -> float:
        if not combo:
            return 0.0
        return decompose_cached(combo).value

    while _uni_count(selected) < contract.max_schools and remaining:
        base = value_of(selected)
        best, best_gain = None, None
        for s in remaining:
            trial = selected + [s]
            if _uni_count(trial) > contract.max_schools:
                continue
            gain = value_of(trial) - base
            if best_gain is None or gain > best_gain:
                best, best_gain = s, gain
        if best is None:
            break
        if (
            _uni_count(selected) >= contract.min_schools
            and best_gain is not None
            and best_gain <= 0
        ):
            break
        selected.append(best)
        remaining.remove(best)

    if return_decomposition:
        final_decomp = decompose_cached(selected) if selected else None
        return selected, final_decomp
    return selected
