from abc import ABC, abstractmethod

from src.pages.prediction.result_modifier.config import (
    AGENT_BOUNDARY_SIMILARITY_RANGE,
    AGENT_MIN_SAFE_RELAX_THRESHOLD,
    AGENT_TAIL_PERCENTAGE,
    CROSS_MAJOR_SIMILARITY_MIN,
    HIGHER_SIMILARITY_THRESHOLD,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class RankerStrategy(ABC):
    def __init__(self, results_with_similarity, current_threshold, background_major):
        self.results_with_similarity = results_with_similarity
        self.current_threshold = current_threshold
        self.background_major = background_major

    @abstractmethod
    def get_initial_candidates(
        self, top_similarity_results: list[dict], results_for_agent: list[dict], top_set: set[tuple]
    ) -> tuple[list[dict], list[dict]]:
        pass

    @abstractmethod
    def triage_decision(self, case: dict) -> bool | None:
        pass

    @abstractmethod
    def update_results(
        self,
        adjusted_results: list[dict],
        cases_to_evaluate: list[dict],
        decisions: list[bool],
        max_adjust: int,
    ) -> tuple[int, list[dict]]:
        pass

    @abstractmethod
    def update_boundary_cases(
        self,
        boundary_candidates: list[dict],
        evaluated_cases: set[tuple],
        adjusted_results: list[dict],
    ) -> list[dict]:
        pass

    @abstractmethod
    def get_exploration_candidates(
        self,
        adjusted_results: list[dict],
        pool_for_exploration: list[dict],
        evaluated_cases: set[tuple],
    ) -> list[dict]:
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
            if (r.get("university"), r.get("major")) not in top_set
            and lower_bound <= r.get("similarity", 0.0) < self.current_threshold
        ]
        boundary_candidates.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        pool_for_exploration = [
            r
            for r in results_for_agent
            if (r.get("university"), r.get("major")) not in top_set
            and CROSS_MAJOR_SIMILARITY_MIN <= r.get("similarity", 0.0) < self.current_threshold
        ]
        pool_for_exploration.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        return boundary_candidates, pool_for_exploration

    def triage_decision(self, case: dict) -> bool | None:
        similarity = case.get("similarity", 0.0)
        if similarity < CROSS_MAJOR_SIMILARITY_MIN:
            return False
        return None

    def update_results(self, adjusted_results, cases_to_evaluate, decisions, max_adjust: int):
        if max_adjust <= 0:
            return 0, adjusted_results

        selected = [
            (i, cases_to_evaluate[i].get("similarity", 0.0))
            for i, d in enumerate(decisions)
            if d and i < len(cases_to_evaluate)
        ]
        selected.sort(key=lambda x: x[1], reverse=True)
        selected_indices = {i for i, _ in selected[:max_adjust]}

        existing_keys = {
            (r.get("university"), r.get("major"))
            for r in adjusted_results
            if isinstance(r, dict)
        }

        added_count = 0
        for i in selected_indices:
            case = cases_to_evaluate[i]
            if self.triage_decision(case) is False:
                continue
            key = (case.get("university"), case.get("major"))
            if key in existing_keys:
                continue
            adjusted_results.append(case)
            existing_keys.add(key)
            added_count += 1

        return added_count, adjusted_results

    def update_boundary_cases(self, boundary_candidates, evaluated_cases, adjusted_results):
        filtered = [
            c
            for c in boundary_candidates
            if (c.get("university"), c.get("major")) not in evaluated_cases
        ]
        return filtered

    def get_exploration_candidates(self, adjusted_results, pool_for_exploration, evaluated_cases):
        next_candidates = [
            c
            for c in pool_for_exploration
            if (c.get("university"), c.get("major")) not in evaluated_cases
        ]
        return next_candidates


class TightenStrategy(RankerStrategy):
    def get_initial_candidates(self, top_similarity_results, results_for_agent, top_set):
        candidates_pool = [
            r
            for r in top_similarity_results
            if r.get("similarity", 0.0) < HIGHER_SIMILARITY_THRESHOLD
        ]

        if not candidates_pool:
            return [], []

        candidates_pool.sort(key=lambda x: x.get("similarity", 0.0))

        tail_count = max(1, int(len(top_similarity_results) * AGENT_TAIL_PERCENTAGE))
        tail_candidates = candidates_pool[:tail_count]

        pool_for_exploration = candidates_pool

        return tail_candidates, pool_for_exploration

    def triage_decision(self, case: dict) -> bool | None:
        similarity = case.get("similarity", 0.0)
        if similarity >= self.current_threshold:
            return False
        return None

    def update_results(self, adjusted_results, cases_to_evaluate, decisions, max_adjust: int):
        if max_adjust <= 0:
            return 0, adjusted_results

        selected = [
            (i, cases_to_evaluate[i].get("similarity", 0.0))
            for i, d in enumerate(decisions)
            if d and i < len(cases_to_evaluate)
        ]
        selected.sort(key=lambda x: x[1])
        selected_indices = [i for i, _ in selected[:max_adjust]]
        cases_to_remove = [cases_to_evaluate[i] for i in selected_indices if i < len(cases_to_evaluate)]

        filtered_to_remove = [c for c in cases_to_remove if self.triage_decision(c) is not False]
        remove_keys = {(c.get("university"), c.get("major")) for c in filtered_to_remove}
        new_results = [
            r for r in adjusted_results if (r.get("university"), r.get("major")) not in remove_keys
        ]
        return len(filtered_to_remove), new_results

    def update_boundary_cases(self, boundary_candidates, evaluated_cases, adjusted_results):
        remaining_tail = [
            r
            for r in adjusted_results
            if (r.get("university"), r.get("major")) not in evaluated_cases
        ]
        remaining_tail.sort(key=lambda x: x.get("similarity", 0.0))
        tail_count = max(1, int(len(remaining_tail) * AGENT_TAIL_PERCENTAGE))
        return remaining_tail[:tail_count]

    def get_exploration_candidates(self, adjusted_results, pool_for_exploration, evaluated_cases):
        remaining_for_exploration = [
            r
            for r in pool_for_exploration
            if (r.get("university"), r.get("major")) not in evaluated_cases
        ]
        remaining_for_exploration.sort(key=lambda x: x.get("similarity", 0.0))
        return remaining_for_exploration
