from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from rapidfuzz import fuzz, process

_DEFAULT_SCORERS: tuple[Callable, ...] = (
    fuzz.partial_ratio,
    fuzz.token_sort_ratio,
    fuzz.token_set_ratio,
)


def _fuzz_search(
    term: str,
    candidates: list[str],
    *,
    scorer: Callable,
    limit: int,
) -> list[tuple[str, float, int]]:
    return process.extract(
        term,
        candidates,
        limit=limit,
        scorer=scorer,
    )


def _merge_rank(
    hits_batches: list[list[tuple[str, float, int]]],
    candidates: list[str],
    *,
    bonus_per_hit: float = 8.0,
) -> list[tuple[str, float, int]]:
    scores: dict[int, float] = {}
    for batch in hits_batches:
        for _candidate_text, score, idx in batch:
            if idx in scores:
                scores[idx] += score + bonus_per_hit
            else:
                scores[idx] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(candidates[idx], score, idx) for idx, score in ranked]


def smart_search(
    query: str,
    candidates: list[str],
    *,
    expander: Callable[[str], list[str]] | None = None,
    scorers: tuple[Callable, ...] | None = None,
    limit: int = 15,
    bonus_per_hit: float = 8.0,
    max_workers: int = 6,
) -> list[tuple[str, float, int]]:
    if expander is not None:
        search_terms = expander(query)
        if not search_terms:
            search_terms = [query]
    else:
        search_terms = [query]

    if scorers is None:
        scorers = _DEFAULT_SCORERS

    all_hits: list[list[tuple[str, float, int]]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for term in search_terms:
            for scorer in scorers:
                futures[
                    executor.submit(
                        _fuzz_search,
                        term,
                        candidates,
                        scorer=scorer,
                        limit=limit,
                    )
                ] = (term, scorer)

        for future in as_completed(futures):
            hits = future.result()
            if hits:
                all_hits.append(hits)

    if not all_hits:
        return []

    return _merge_rank(all_hits, candidates, bonus_per_hit=bonus_per_hit)
