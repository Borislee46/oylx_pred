from abc import ABC, abstractmethod
from typing import Any

from rapidfuzz import fuzz

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

    def check_fuzzy_bypass(self, case: dict[str, Any]) -> bool:
        if not self.background_major:
            return False

        bg = str(self.background_major).lower()
        m_en = str(case.get("major", "")).lower()
        m_cn = str(case.get("major_cn", "")).lower()

        score_en = fuzz.token_sort_ratio(bg, m_en)
        score_cn = fuzz.token_sort_ratio(bg, m_cn) if m_cn else 0

        return max(score_en, score_cn) > 92

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

        existing_keys = {case_key(r) for r in adjusted_results if case_key(r)}
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
            new_results = [r for r in adjusted_results if case_key(r) not in remove_keys]

        return added_count, new_results

    def update_boundary_cases(self, boundary_candidates, evaluated_cases, adjusted_results):
        return [c for c in boundary_candidates if case_key(c) not in evaluated_cases]

    def get_exploration_candidates(self, adjusted_results, pool_for_exploration, evaluated_cases):
        return [c for c in pool_for_exploration if case_key(c) not in evaluated_cases]


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

        candidates = [
            r
            for r in results_for_agent
            if (k := case_key(r))
            and k not in top_set
            and lower_bound <= get_similarity(r) < self.current_threshold
        ]
        candidates.sort(key=get_similarity, reverse=True)

        pool = [
            r
            for r in results_for_agent
            if (k := case_key(r))
            and k not in top_set
            and CROSS_MAJOR_SIMILARITY_MIN <= get_similarity(r) < self.current_threshold
        ]
        pool.sort(key=get_similarity, reverse=True)
        return candidates, pool

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
            return [], []

        tail_count = max(1, int(len(top_similarity_results) * AGENT_TAIL_PERCENTAGE))
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
            [r for r in adjusted_results if case_key(r) not in evaluated_cases], key=get_similarity
        )
        tail_count = max(1, int(len(remaining) * AGENT_TAIL_PERCENTAGE))
        return remaining[:tail_count]
