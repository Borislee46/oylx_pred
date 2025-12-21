from typing import Any

from src.pages.prediction.result_modifier.engine import (
    AgentAdjustmentEngine,
    AgentAdjustmentSession,
)
from src.pages.prediction.result_modifier.strategies import (
    RankerStrategy,
    RelaxStrategy,
    TightenStrategy,
)
from src.pages.prediction.result_modifier.types import CaseKey, case_key
from src.pages.prediction.result_modifier.ui_handler import RankerUIHandler
from src.pages.prediction.result_modifier.utils import (
    clip_probability,
    deduplicate_results,
    get_probability,
)


# 当前场景heapq未必比sorted快, candidate多（十万级？）的话可换heapq
def _pick_supplement_cases_by_probability(
    candidates: list[dict[str, Any]],
    top_set: set[tuple[Any, Any]],
    p_min: float,
    k_high: int = 8,
    k_band: int = 8,
    band_delta: float = 0.03,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    with_prob = []
    for r in candidates:
        k = case_key(r)
        if k and k not in top_set:
            with_prob.append((r, get_probability(r), k))

    if not with_prob:
        return []

    with_prob.sort(key=lambda x: x[1], reverse=True)

    picked = []
    seen = set()

    for r, _p, k in with_prob:
        if k not in seen:
            seen.add(k)
            picked.append(r)
            if len(picked) >= k_high:
                break

    low = p_min - band_delta
    high = p_min + band_delta

    band_candidates = [(r, p, k) for r, p, k in with_prob if k not in seen and low <= p <= high]
    band_candidates.sort(key=lambda x: abs(x[1] - p_min))

    for r, _p, _k in band_candidates:
        if len(picked) >= (k_high + k_band):
            break
        picked.append(r)

    return picked


def adjust_similarity_results_with_agent(
    top_similarity_results: list[dict[str, Any]],
    results_with_similarity: list[dict[str, Any]],
    balance_diff: int,
    background_major: str,
    current_threshold: float,
    agent: Any,
    background_faculty: str | None = None,
    progress_reporter: Any | None = None,
) -> list[dict[str, Any]]:
    if not top_similarity_results or not agent or balance_diff == 0:
        return top_similarity_results

    mode = "relax" if balance_diff > 0 else "tighten"
    target_diff = abs(balance_diff)

    strategy: RankerStrategy
    if mode == "relax":
        strategy = RelaxStrategy(results_with_similarity, current_threshold, background_major)
    else:
        strategy = TightenStrategy(results_with_similarity, current_threshold, background_major)

    if background_faculty:
        from src.pages.prediction.result_modifier.faculty_filters import (
            filter_schools_by_faculty_rules,
        )

        results_for_agent = filter_schools_by_faculty_rules(
            results_with_similarity, background_faculty
        )
    else:
        results_for_agent = results_with_similarity

    top_set: set[CaseKey] = set()
    for r in top_similarity_results:
        k = case_key(r)
        if k:
            top_set.add(k)

    bg_faculties: list[str] = []
    if str(background_major or "").strip():
        from src.agent.background_faculty_agent import BackgroundFacultyAgent

        bg_agent = BackgroundFacultyAgent()
        bg_faculties = bg_agent.resolve_background_faculties(
            background_major_original=background_major,
            base_faculty=background_faculty,
            max_total=3,
            max_extra=2,
            use_persistent_cache=True,
        )

    base = background_faculty.strip() if isinstance(background_faculty, str) else ""
    extras = [f for f in bg_faculties if f and f != base]
    if extras:
        p_min = 0.0
        if top_similarity_results:
            try:
                p_min = min(float(r.get("probability", 0.0) or 0.0) for r in top_similarity_results)
            except (TypeError, ValueError):
                p_min = 0.0

        supplements: list[dict[str, Any]] = []

        faculty_map: dict[str, list[dict[str, Any]]] = {}
        for r in results_with_similarity:
            if not isinstance(r, dict):
                continue
            f = str(r.get("faculty", "")).strip()
            if f not in faculty_map:
                faculty_map[f] = []
            faculty_map[f].append(r)

        for extra in extras[:2]:
            extra_candidates = faculty_map.get(extra, [])
            if not extra_candidates:
                continue
            supplements.extend(
                _pick_supplement_cases_by_probability(
                    candidates=extra_candidates,
                    top_set=top_set,
                    p_min=p_min,
                    k_high=8,
                    k_band=8,
                    band_delta=0.03,
                )
            )

        if supplements:
            results_for_agent = deduplicate_results(results_for_agent + supplements)

    boundary_candidates, pool_for_exploration = strategy.get_initial_candidates(
        top_similarity_results, results_for_agent, top_set
    )

    session = AgentAdjustmentSession(strategy, target_diff, mode)

    final_results = top_similarity_results

    with RankerUIHandler(
        background_major=background_major,
        background_faculty=background_faculty,
        mode=mode,
        progress_reporter=progress_reporter,
    ) as ui_handler:
        engine = AgentAdjustmentEngine(agent, session, ui_handler)
        final_results = engine.run(
            boundary_candidates, pool_for_exploration, top_similarity_results
        )

    for c in final_results:
        if isinstance(c, dict) and "probability" in c:
            c["probability"] = clip_probability(c.get("probability", 0.0))

    final_results.sort(key=lambda x: get_probability(x), reverse=True)
    return final_results
