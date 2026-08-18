from typing import Any

from src.adjustment.engine import (
    AgentAdjustmentEngine,
    AgentAdjustmentSession,
    CaseKey,
    case_key,
)
from src.adjustment.faculty_filters import (
    filter_schools_by_faculty_rules,
)
from src.adjustment.strategies import (
    RankerStrategy,
    RelaxStrategy,
    TightenStrategy,
)
from src.adjustment.ui_handler import RankerUIHandler
from src.adjustment.utils import (
    deduplicate_results,
    get_probability,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability, clip_probability_coerce

logger = setup_logger("page3", "prediction")


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
        if k is not None and k not in top_set:
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


def _resolve_faculty_supplements(
    background_major: str,
    background_faculty: str | None,
    results_with_similarity: list[dict[str, Any]],
    top_set: set[CaseKey],
    top_similarity_results: list[dict[str, Any]],
    bg_faculty_agent: Any = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    bg_faculties: list[str] = []
    if str(background_major or "").strip():
        if bg_faculty_agent is None:
            from src.agent.background_faculty_agent import BackgroundFacultyAgent

            bg_faculty_agent = BackgroundFacultyAgent()
        bg_faculties = bg_faculty_agent.resolve_background_faculties(
            background_major_original=background_major,
            base_faculty=background_faculty,
            max_total=3,
            max_extra=2,
            use_persistent_cache=True,
        )

    base = background_faculty.strip() if isinstance(background_faculty, str) else ""
    extras = [f for f in bg_faculties if f and f != base]
    if not extras:
        return bg_faculties, []

    p_min = 0.0
    if top_similarity_results:
        try:
            p_min = min(
                clip_probability_coerce(r.get("probability")) for r in top_similarity_results
            )
        except (TypeError, ValueError):
            p_min = 0.0

    faculty_map: dict[str, list[dict[str, Any]]] = {}
    for r in results_with_similarity:
        if not isinstance(r, dict):
            continue
        f = str(r.get("faculty", "")).strip()
        if f not in faculty_map:
            faculty_map[f] = []
        faculty_map[f].append(r)

    supplements: list[dict[str, Any]] = []
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

    return bg_faculties, supplements


def adjust_similarity_results_with_agent(
    top_similarity_results: list[dict[str, Any]],
    results_with_similarity: list[dict[str, Any]],
    balance_diff: int,
    background_major: str,
    current_threshold: float,
    agent: Any,
    background_faculty: str | None = None,
    progress_reporter: Any | None = None,
    is_cross_faculty: bool = False,
    bg_faculty_agent: Any = None,
) -> list[dict[str, Any]]:
    if not top_similarity_results or not agent or balance_diff == 0:
        logger.debug(
            "Agent 调整跳过 | 结果=%s 代理=%s 平衡差=%d",
            bool(top_similarity_results),
            bool(agent),
            balance_diff,
        )
        return top_similarity_results

    mode = "relax" if balance_diff > 0 else "tighten"
    target_diff = abs(balance_diff)

    logger.info(
        "Agent 调整开始 | 模式=%s 目标差=%d 阈值=%.3f 背景专业=%s "
        "n_top=%d n_pool=%d is_cross_faculty=%s",
        mode,
        target_diff,
        current_threshold,
        background_major,
        len(top_similarity_results),
        len(results_with_similarity),
        is_cross_faculty,
    )

    strategy: RankerStrategy
    if mode == "relax":
        strategy = RelaxStrategy(results_with_similarity, current_threshold, background_major)
    else:
        strategy = TightenStrategy(results_with_similarity, current_threshold, background_major)

    if background_faculty and not is_cross_faculty:
        results_for_agent = filter_schools_by_faculty_rules(
            results_with_similarity, background_faculty
        )
        logger.debug(
            "Agent 调整 | 学部过滤 | before=%d after=%d",
            len(results_with_similarity),
            len(results_for_agent),
        )
    else:
        results_for_agent = results_with_similarity

    top_set: set[CaseKey] = {k for r in top_similarity_results if (k := case_key(r)) is not None}

    _, supplements = _resolve_faculty_supplements(
        background_major=background_major,
        background_faculty=background_faculty,
        results_with_similarity=results_with_similarity,
        top_set=top_set,
        top_similarity_results=top_similarity_results,
        bg_faculty_agent=bg_faculty_agent,
    )
    if supplements:
        logger.info(
            "Agent 调整 | 学部补充案例 | n_supplements=%d",
            len(supplements),
        )
        results_for_agent = deduplicate_results(results_for_agent + supplements)

    boundary_candidates, pool_for_exploration = strategy.get_initial_candidates(
        top_similarity_results, results_for_agent, top_set
    )

    session = AgentAdjustmentSession(strategy, target_diff, mode)

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
    logger.info(
        "Agent 调整完成 | 模式=%s 原始=%d 调整后=%d 调整=%d/%d",
        mode,
        len(top_similarity_results),
        len(final_results),
        session.adjusted_count,
        target_diff,
    )
    return final_results
