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
from src.pages.prediction.result_modifier.ui_handler import RankerUIHandler
from src.pages.prediction.result_modifier.utils import clip_probability


def adjust_similarity_results_with_agent(
    top_similarity_results: list[dict[str, Any]],
    results_with_similarity: list[dict[str, Any]],
    balance_diff: int,
    background_major: str,
    current_threshold: float,
    agent: Any,
    background_faculty: str | None = None,
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

    top_set = {(r.get("university"), r.get("major")) for r in top_similarity_results}

    boundary_candidates, pool_for_exploration = strategy.get_initial_candidates(
        top_similarity_results, results_for_agent, top_set
    )

    session = AgentAdjustmentSession(strategy, target_diff, mode)

    final_results = top_similarity_results

    with RankerUIHandler(
        background_major=background_major,
        background_faculty=background_faculty,
        mode=mode,
    ) as ui_handler:
        engine = AgentAdjustmentEngine(agent, session, ui_handler)
        final_results = engine.run(
            boundary_candidates, pool_for_exploration, top_similarity_results
        )

    for c in final_results:
        if isinstance(c, dict) and "probability" in c:
            c["probability"] = clip_probability(c.get("probability", 0.0))

    final_results.sort(key=lambda x: x.get("probability", 0.0), reverse=True)
    return final_results
