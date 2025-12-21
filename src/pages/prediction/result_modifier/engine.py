import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.pages.prediction.result_modifier.config import (
    AGENT_MAX_BOUNDARY_CASES,
)
from src.pages.prediction.result_modifier.strategies import RankerStrategy
from src.pages.prediction.result_modifier.types import (
    AdjustmentDecision,
    CaseKey,
    case_key,
    is_case_with_key,
)
from src.pages.prediction.result_modifier.ui_handler import RankerUIHandler
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class AgentAdjustmentSession:
    def __init__(self, strategy: RankerStrategy, target_diff: int, mode: str):
        self.strategy = strategy
        self.target_diff = target_diff
        self.mode = mode
        self.adjusted_count = 0
        self.evaluated_cases: set[CaseKey] = set()

    def record_evaluation(self, cases: list[dict]):
        for case in cases:
            k = case_key(case)
            if k:
                self.evaluated_cases.add(k)

    def should_stop(self) -> bool:
        return self.adjusted_count >= self.target_diff


class AgentAdjustmentEngine:
    def __init__(self, agent: Any, session: AgentAdjustmentSession, ui_handler: RankerUIHandler):
        self.agent = agent
        self.session = session
        self.ui = ui_handler

    def run(
        self,
        initial_boundary_cases: list[dict],
        initial_pool: list[dict],
        initial_results: list[dict],
    ) -> list[dict]:
        """
        重构为单次大批次评估逻辑，彻底消除多轮迭代的 Token 损耗
        """
        self.ui.update_loop()
        remaining = self.session.target_diff - self.session.adjusted_count
        if remaining <= 0:
            return initial_results

        # 1. 收集候选案例（优先边界，其次池子）
        # 一次性取够足够的候选量，避免反复请求
        max_candidates = min(AGENT_MAX_BOUNDARY_CASES * 2, max(12, remaining * 3))
        candidates = []
        seen_keys = set()

        for pool in [initial_boundary_cases, initial_pool]:
            for c in pool:
                if not is_case_with_key(c):
                    continue
                key = case_key(c)
                if key not in seen_keys and key not in self.session.evaluated_cases:
                    candidates.append(c)
                    seen_keys.add(key)
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

        if not candidates:
            return initial_results

        # 2. UI 显示和单次批处理评估
        majors = [c.get("major", "") for c in candidates if c.get("major")]
        self.ui.show_candidates(majors)

        with ThreadPoolExecutor(max_workers=1) as executor:
            decisions = self._process_decisions(executor, candidates)

        self.session.record_evaluation(candidates)

        # 3. 应用结果
        if any(decisions):
            count, adjusted_results = self.session.strategy.update_results(
                initial_results, candidates, decisions, max_adjust=remaining
            )
            self.session.adjusted_count += count
            return adjusted_results

        return initial_results

    def _process_decisions(self, executor, cases: list[dict]) -> list[bool]:
        decisions = [False] * len(cases)
        agent_cases = []
        agent_indices = []

        for i, case in enumerate(cases):
            triage = self.session.strategy.triage_decision(case)
            if triage is AdjustmentDecision.DEFER_TO_AGENT:
                agent_cases.append(case)
                agent_indices.append(i)
            else:
                decisions[i] = triage is AdjustmentDecision.ADJUST

        if agent_cases:
            evaluation = self._evaluate_with_agent(executor, agent_cases)
            agent_decisions = evaluation.get("decisions", []) if evaluation else []
            for j, idx in enumerate(agent_indices):
                if j < len(agent_decisions):
                    decisions[idx] = bool(agent_decisions[j])

        return decisions

    def _evaluate_with_agent(self, executor: ThreadPoolExecutor, cases: list[dict]):
        future = executor.submit(
            self.agent.evaluate_boundary_cases,
            self.session.strategy.background_major,
            cases,
            self.session.mode,
        )
        while not future.done():
            self.ui.update_loop()
            time.sleep(0.2)
        return future.result()
