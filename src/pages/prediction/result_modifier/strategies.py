from abc import ABC, abstractmethod
from typing import Any

from src.pages.prediction.result_modifier.config import (
    AGENT_BOUNDARY_SIMILARITY_RANGE,
    AGENT_MIN_SAFE_RELAX_THRESHOLD,
    AGENT_TAIL_PERCENTAGE,
    CROSS_MAJOR_SIMILARITY_MIN,
    HIGHER_SIMILARITY_THRESHOLD,
)
from src.pages.prediction.result_modifier.types import (
    AdjustmentDecision,
    CaseKey,
    case_key,
    get_similarity,
)


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

    @abstractmethod
    def update_results(
        self,
        adjusted_results: list[dict[str, Any]],
        cases_to_evaluate: list[dict[str, Any]],
        decisions: list[bool],
        max_adjust: int,
    ) -> tuple[int, list[dict[str, Any]]]:
        pass

    @abstractmethod
    def update_boundary_cases(
        self,
        boundary_candidates: list[dict[str, Any]],
        evaluated_cases: set[CaseKey],
        adjusted_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def get_exploration_candidates(
        self,
        adjusted_results: list[dict[str, Any]],
        pool_for_exploration: list[dict[str, Any]],
        evaluated_cases: set[CaseKey],
    ) -> list[dict[str, Any]]:
        pass


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

        boundary_candidates = [
            r
            for r in results_for_agent
            if (k := case_key(r)) is not None
            and k not in top_set
            and lower_bound <= get_similarity(r) < self.current_threshold
        ]
        boundary_candidates.sort(key=get_similarity, reverse=True)

        pool_for_exploration = [
            r
            for r in results_for_agent
            if (k := case_key(r)) is not None
            and k not in top_set
            and CROSS_MAJOR_SIMILARITY_MIN <= get_similarity(r) < self.current_threshold
        ]
        pool_for_exploration.sort(key=get_similarity, reverse=True)

        return boundary_candidates, pool_for_exploration

    def triage_decision(self, case: dict[str, Any]) -> AdjustmentDecision:
        similarity = get_similarity(case)
        if similarity < CROSS_MAJOR_SIMILARITY_MIN:
            return AdjustmentDecision.NO_ADJUST
        return AdjustmentDecision.DEFER_TO_AGENT

    def update_results(self, adjusted_results, cases_to_evaluate, decisions, max_adjust: int):
        if max_adjust <= 0:
            return 0, adjusted_results

        selected: list[tuple[int, float]] = [
            (i, get_similarity(cases_to_evaluate[i]))
            for i, d in enumerate(decisions)
            if d and i < len(cases_to_evaluate)
        ]
        selected.sort(key=lambda x: x[1], reverse=True)

        existing_keys: set[CaseKey] = set()
        for r in adjusted_results:
            k = case_key(r)
            if k:
                existing_keys.add(k)

        new_results = adjusted_results.copy()
        added_count = 0
        for i, _ in selected[:max_adjust]:
            case = cases_to_evaluate[i]
            if self.triage_decision(case) is AdjustmentDecision.NO_ADJUST:
                continue
            k = case_key(case)
            if not k or k in existing_keys:
                continue
            new_results.append(case)
            existing_keys.add(k)
            added_count += 1

        return added_count, new_results

    def update_boundary_cases(self, boundary_candidates, evaluated_cases, adjusted_results):
        filtered = [
            c
            for c in boundary_candidates
            if (k := case_key(c)) is not None and k not in evaluated_cases
        ]
        return filtered

    def get_exploration_candidates(self, adjusted_results, pool_for_exploration, evaluated_cases):
        next_candidates = [
            c
            for c in pool_for_exploration
            if (k := case_key(c)) is not None and k not in evaluated_cases
        ]
        return next_candidates


class TightenStrategy(RankerStrategy):
    def get_initial_candidates(self, top_similarity_results, results_for_agent, top_set):
        candidates_pool = [
            r for r in top_similarity_results if get_similarity(r) < HIGHER_SIMILARITY_THRESHOLD
        ]

        if not candidates_pool:
            return [], []

        candidates_pool.sort(key=get_similarity)

        tail_count = max(1, int(len(top_similarity_results) * AGENT_TAIL_PERCENTAGE))
        tail_candidates = candidates_pool[:tail_count]

        pool_for_exploration = candidates_pool

        return tail_candidates, pool_for_exploration

    def triage_decision(self, case: dict[str, Any]) -> AdjustmentDecision:
        similarity = get_similarity(case)
        if similarity >= self.current_threshold:
            return AdjustmentDecision.NO_ADJUST
        return AdjustmentDecision.DEFER_TO_AGENT

    def update_results(self, adjusted_results, cases_to_evaluate, decisions, max_adjust: int):
        if max_adjust <= 0:
            return 0, adjusted_results

        selected = [
            (i, get_similarity(cases_to_evaluate[i]))
            for i, d in enumerate(decisions)
            if d and i < len(cases_to_evaluate)
        ]
        selected.sort(key=lambda x: x[1])
        selected_indices = [i for i, _ in selected[:max_adjust]]
        cases_to_remove = [
            cases_to_evaluate[i] for i in selected_indices if i < len(cases_to_evaluate)
        ]

        filtered_to_remove = [
            c
            for c in cases_to_remove
            if self.triage_decision(c) is not AdjustmentDecision.NO_ADJUST
        ]
        remove_keys: set[CaseKey] = set()
        for c in filtered_to_remove:
            k = case_key(c)
            if k:
                remove_keys.add(k)
        new_results = [
            r for r in adjusted_results if (k := case_key(r)) is None or k not in remove_keys
        ]
        return len(filtered_to_remove), new_results

    def update_boundary_cases(self, boundary_candidates, evaluated_cases, adjusted_results):
        remaining_tail = [
            r
            for r in adjusted_results
            if (k := case_key(r)) is not None and k not in evaluated_cases
        ]
        remaining_tail.sort(key=get_similarity)
        tail_count = max(1, int(len(remaining_tail) * AGENT_TAIL_PERCENTAGE))
        return remaining_tail[:tail_count]

    def get_exploration_candidates(self, adjusted_results, pool_for_exploration, evaluated_cases):
        remaining_for_exploration = [
            r
            for r in pool_for_exploration
            if (k := case_key(r)) is not None and k not in evaluated_cases
        ]
        remaining_for_exploration.sort(key=get_similarity)
        return remaining_for_exploration
