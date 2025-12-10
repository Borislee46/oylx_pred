from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple

from src.pages.prediction.result_modifier.config import (
    AGENT_BOUNDARY_SIMILARITY_RANGE,
    AGENT_MAX_BOUNDARY_CASES,
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
        self, top_similarity_results: List[Dict], results_for_agent: List[Dict], top_set: Set[Tuple]
    ) -> Tuple[List[Dict], List[Dict]]:
        pass

    @abstractmethod
    def update_results(
        self, adjusted_results: List[Dict], cases_to_evaluate: List[Dict], decisions: List[bool]
    ) -> Tuple[int, List[Dict]]:
        pass

    @abstractmethod
    def update_boundary_cases(
        self,
        boundary_candidates: List[Dict],
        evaluated_cases: Set[Tuple],
        adjusted_results: List[Dict],
    ) -> List[Dict]:
        pass

    @abstractmethod
    def get_exploration_candidates(
        self,
        adjusted_results: List[Dict],
        pool_for_exploration: List[Dict],
        evaluated_cases: Set[Tuple],
    ) -> List[Dict]:
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

        return boundary_candidates[:AGENT_MAX_BOUNDARY_CASES], pool_for_exploration

    def update_results(self, adjusted_results, cases_to_evaluate, decisions):
        added_count = 0
        new_results = adjusted_results

        for i, decision in enumerate(decisions):
            if i < len(cases_to_evaluate) and decision:
                case = cases_to_evaluate[i]
                similarity = case.get("similarity", 0.0)
                if similarity < CROSS_MAJOR_SIMILARITY_MIN:
                    logger.warning(
                        f"Agent 添加相似度过低的专业: {case.get('major')} "
                        f"(Sim: {similarity:.3f}, BG: {self.background_major})"
                    )
                adjusted_results.append(case)
                added_count += 1
        return added_count, new_results

    def update_boundary_cases(self, boundary_candidates, evaluated_cases, adjusted_results):
        filtered = [
            c
            for c in boundary_candidates
            if (c.get("university"), c.get("major")) not in evaluated_cases
        ]
        return filtered[:AGENT_MAX_BOUNDARY_CASES]

    def get_exploration_candidates(self, adjusted_results, pool_for_exploration, evaluated_cases):
        next_candidates = [
            c
            for c in pool_for_exploration
            if (c.get("university"), c.get("major")) not in evaluated_cases
        ]
        return next_candidates[:AGENT_MAX_BOUNDARY_CASES]


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

        return tail_candidates[:AGENT_MAX_BOUNDARY_CASES], pool_for_exploration

    def update_results(self, adjusted_results, cases_to_evaluate, decisions):
        cases_to_remove = [
            cases_to_evaluate[i]
            for i, decision in enumerate(decisions)
            if i < len(cases_to_evaluate) and decision
        ]

        for case in cases_to_remove:
            similarity = case.get("similarity", 0.0)
            if similarity >= self.current_threshold:
                logger.warning(
                    f"Agent 移除相似度较高的专业: {case.get('major')} "
                    f"(Sim: {similarity:.3f}, Threshold: {self.current_threshold:.3f})"
                )

        remove_keys = {(c.get("university"), c.get("major")) for c in cases_to_remove}
        new_results = [
            r for r in adjusted_results if (r.get("university"), r.get("major")) not in remove_keys
        ]
        return len(cases_to_remove), new_results

    def update_boundary_cases(self, boundary_candidates, evaluated_cases, adjusted_results):
        remaining_tail = [
            r
            for r in adjusted_results
            if (r.get("university"), r.get("major")) not in evaluated_cases
        ]
        remaining_tail.sort(key=lambda x: x.get("similarity", 0.0))
        tail_count = max(1, int(len(remaining_tail) * AGENT_TAIL_PERCENTAGE))
        return remaining_tail[: min(tail_count, AGENT_MAX_BOUNDARY_CASES)]

    def get_exploration_candidates(self, adjusted_results, pool_for_exploration, evaluated_cases):
        remaining_for_exploration = [
            r
            for r in pool_for_exploration
            if (r.get("university"), r.get("major")) not in evaluated_cases
        ]
        remaining_for_exploration.sort(key=lambda x: x.get("similarity", 0.0))
        return remaining_for_exploration[:AGENT_MAX_BOUNDARY_CASES]
