import heapq
from abc import ABC, abstractmethod
from typing import Any

from src.adjustment.config import (
    AGENT_BOUNDARY_SIMILARITY_RANGE,
    AGENT_MAX_BOUNDARY_CASES,
    AGENT_MIN_SAFE_RELAX_THRESHOLD,
    AGENT_TAIL_PERCENTAGE,
    CROSS_MAJOR_SIMILARITY_MIN,
    FUZZY_BIAS_THRESHOLD_HIGH,
    HIGHER_SIMILARITY_THRESHOLD,
)
from src.adjustment.engine import (
    AdjustmentDecision,
    CaseKey,
    case_key,
)
from src.adjustment.utils import get_similarity
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

_RELAX_AGENT_POOL_CAP = AGENT_MAX_BOUNDARY_CASES * 2


def _top_by_similarity_desc(items: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if not items:
        return []
    if len(items) <= k:
        return sorted(items, key=get_similarity, reverse=True)
    return heapq.nlargest(k, items, key=get_similarity)


class RankerStrategy(ABC):
    def __init__(self, results_with_similarity, current_threshold, background_major):
        self.results_with_similarity = results_with_similarity
        self.current_threshold = current_threshold
        self.background_major = background_major

    @abstractmethod
    def get_initial_candidates(
        self,
        top_similarity_results: list[dict[str, Any]],
        results_for_agent: list[dict[str, Any]],
        top_set: set[CaseKey],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        pass

    @abstractmethod
    def triage_decision(self, case: dict[str, Any]) -> AdjustmentDecision:
        pass

    def check_fuzzy_bypass(self, case: dict[str, Any]) -> bool:
        if not self.background_major:
            return False

        return case.get("_strong_match_score", 0) > FUZZY_BIAS_THRESHOLD_HIGH

    def update_results(
        self,
        adjusted_results: list[dict[str, Any]],
        cases_to_evaluate: list[dict[str, Any]],
        decisions: list[bool],
        max_adjust: int,
        reverse_sort: bool = True,
    ) -> tuple[int, list[dict[str, Any]]]:
        if max_adjust <= 0:
            return 0, adjusted_results

        selected = [
            (i, get_similarity(cases_to_evaluate[i]))
            for i, d in enumerate(decisions)
            if d and i < len(cases_to_evaluate)
        ]
        selected.sort(key=lambda x: x[1], reverse=reverse_sort)

        existing_keys = {k for r in adjusted_results if (k := case_key(r)) is not None}
        new_results = adjusted_results.copy()
        added_count = 0
        remove_keys = set()

        for i, _ in selected[:max_adjust]:
            case = cases_to_evaluate[i]
            if self.triage_decision(case) is AdjustmentDecision.NO_ADJUST:
                continue
            k = case_key(case)
            if not k:
                continue

            if reverse_sort:
                if k not in existing_keys:
                    new_results.append(case)
                    existing_keys.add(k)
                    added_count += 1
            else:
                remove_keys.add(k)
                added_count += 1

        if not reverse_sort and remove_keys:
            new_results = [
                r for r in adjusted_results if (k := case_key(r)) is None or k not in remove_keys
            ]

        logger.info(
            "策略 update_results | 模式=%s 最大调整=%d 决定=%d 选中=%d 添加=%d 移除=%d 最终=%d",
            "relax" if reverse_sort else "tighten",
            max_adjust,
            sum(decisions),
            len(selected),
            added_count if reverse_sort else 0,
            len(remove_keys) if not reverse_sort else 0,
            len(new_results),
        )
        return added_count, new_results

    def update_boundary_cases(self, boundary_candidates, evaluated_cases, adjusted_results):
        return [
            c
            for c in boundary_candidates
            if (k := case_key(c)) is not None and k not in evaluated_cases
        ]

    def get_exploration_candidates(self, adjusted_results, pool_for_exploration, evaluated_cases):
        return [
            c
            for c in pool_for_exploration
            if (k := case_key(c)) is not None and k not in evaluated_cases
        ]


class RelaxStrategy(RankerStrategy):
    def get_initial_candidates(self, top_similarity_results, results_for_agent, top_set):
        boundary_range = (
            AGENT_BOUNDARY_SIMILARITY_RANGE * 2.0
            if self.current_threshold >= HIGHER_SIMILARITY_THRESHOLD
            else AGENT_BOUNDARY_SIMILARITY_RANGE * 1.5
        )
        lower_bound = max(
            self.current_threshold - boundary_range,
            AGENT_MIN_SAFE_RELAX_THRESHOLD,
            CROSS_MAJOR_SIMILARITY_MIN,
        )

        candidates = []
        pool = []

        for r in results_for_agent:
            k = case_key(r)
            if not k or k in top_set:
                continue

            sim = get_similarity(r)
            if sim < CROSS_MAJOR_SIMILARITY_MIN or sim >= self.current_threshold:
                continue

            pool.append(r)
            if sim >= lower_bound:
                candidates.append(r)

        logger.info(
            "RelaxStrategy | 候选筛选 | 阈值=%.3f 边界范围=%.3f 下界=%.3f "
            "n_candidates=%d n_pool=%d n_top=%d",
            self.current_threshold,
            boundary_range,
            lower_bound,
            len(candidates),
            len(pool),
            len(top_set),
        )
        return (
            _top_by_similarity_desc(candidates, _RELAX_AGENT_POOL_CAP),
            _top_by_similarity_desc(pool, _RELAX_AGENT_POOL_CAP),
        )

    def triage_decision(self, case: dict[str, Any]) -> AdjustmentDecision:
        if self.check_fuzzy_bypass(case):
            return AdjustmentDecision.ADJUST

        sim = get_similarity(case)
        lower = max(
            self.current_threshold - (AGENT_BOUNDARY_SIMILARITY_RANGE * 1.5),
            CROSS_MAJOR_SIMILARITY_MIN,
        )
        if sim < lower:
            return AdjustmentDecision.NO_ADJUST
        return AdjustmentDecision.DEFER_TO_AGENT

    def update_results(self, adjusted_results, cases_to_evaluate, decisions, max_adjust: int):
        return super().update_results(
            adjusted_results, cases_to_evaluate, decisions, max_adjust, reverse_sort=True
        )


class TightenStrategy(RankerStrategy):
    def get_initial_candidates(self, top_similarity_results, results_for_agent, top_set):
        candidates_pool = sorted(
            [r for r in top_similarity_results if get_similarity(r) < HIGHER_SIMILARITY_THRESHOLD],
            key=get_similarity,
        )
        if not candidates_pool:
            logger.debug(
                "TightenStrategy | 无候选 | 全部高于阈值 %.3f", HIGHER_SIMILARITY_THRESHOLD
            )
            return [], []

        tail_count = max(1, int(len(top_similarity_results) * AGENT_TAIL_PERCENTAGE))
        logger.info(
            "TightenStrategy | 候选筛选 | n_top=%d n_below_threshold=%d tail_count=%d",
            len(top_similarity_results),
            len(candidates_pool),
            tail_count,
        )
        return candidates_pool[:tail_count], candidates_pool

    def triage_decision(self, case: dict[str, Any]) -> AdjustmentDecision:
        if self.check_fuzzy_bypass(case):
            return AdjustmentDecision.NO_ADJUST

        sim = get_similarity(case)
        if sim >= self.current_threshold:
            return AdjustmentDecision.NO_ADJUST
        borderline_low = self.current_threshold - (AGENT_BOUNDARY_SIMILARITY_RANGE * 1.5)
        if sim < borderline_low:
            return AdjustmentDecision.ADJUST
        return AdjustmentDecision.DEFER_TO_AGENT

    def update_results(self, adjusted_results, cases_to_evaluate, decisions, max_adjust: int):
        return super().update_results(
            adjusted_results, cases_to_evaluate, decisions, max_adjust, reverse_sort=False
        )

    def update_boundary_cases(self, boundary_candidates, evaluated_cases, adjusted_results):
        remaining = sorted(
            [
                r
                for r in adjusted_results
                if (k := case_key(r)) is not None and k not in evaluated_cases
            ],
            key=get_similarity,
        )
        tail_count = max(1, int(len(remaining) * AGENT_TAIL_PERCENTAGE))
        return remaining[:tail_count]
