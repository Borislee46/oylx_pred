import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.pages.prediction.result_modifier.config import (
    AGENT_EARLY_STOP_THRESHOLD,
    AGENT_EXPLORATION_MAX_ROUNDS,
    AGENT_MAX_BOUNDARY_CASES,
    AGENT_NO_CHANGE_THRESHOLD,
)
from src.pages.prediction.result_modifier.strategies import RankerStrategy
from src.pages.prediction.result_modifier.ui_handler import RankerUIHandler
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class AgentAdjustmentSession:
    def __init__(self, strategy: RankerStrategy, target_diff: int, mode: str):
        self.strategy = strategy
        self.target_diff = target_diff
        self.mode = mode

        self.adjusted_count = 0
        self.no_change_count = 0
        self.exploration_rounds = 0
        self.in_exploration_mode = False
        self.exploration_no_change_count = 0

        self.evaluated_cases: set[tuple[str, str]] = set()
        self.stop_requested = False

    def record_evaluation(self, cases: list[dict]):
        for case in cases:
            university = case.get("university")
            major = case.get("major")
            if isinstance(university, str) and isinstance(major, str):
                self.evaluated_cases.add((university, major))

    def should_stop(self) -> bool:
        if self.stop_requested:
            return True
        if self.adjusted_count >= self.target_diff:
            logger.info(f"Agent 已调整 {self.adjusted_count} 个，达到目标 {self.target_diff}")
            return True
        if self.in_exploration_mode and self.exploration_rounds >= AGENT_EXPLORATION_MAX_ROUNDS:
            logger.info(f"达到最大探索轮数 {AGENT_EXPLORATION_MAX_ROUNDS}")
            return True
        return False

    def handle_no_adjustment(self) -> tuple[bool, bool]:
        should_stop = False
        should_explore = False

        if self.in_exploration_mode:
            self.exploration_no_change_count += 1
            if self.exploration_no_change_count >= AGENT_EARLY_STOP_THRESHOLD:
                logger.info(f"探索模式下连续 {self.exploration_no_change_count} 次无调整，停止")
                should_stop = True
        else:
            self.no_change_count += 1
            if self.no_change_count >= AGENT_EARLY_STOP_THRESHOLD:
                if self.adjusted_count > 0:
                    logger.info(
                        f"连续 {self.no_change_count} 次无调整，已调整 {self.adjusted_count}，停止"
                    )
                    should_stop = True
                elif self.target_diff > 0:
                    logger.info("连续无调整且未达标，进入探索模式")
                    should_explore = True
                else:
                    should_stop = True
            elif self.no_change_count >= AGENT_NO_CHANGE_THRESHOLD:
                logger.info("达到无调整阈值，尝试探索模式")
                should_explore = True

        if should_explore:
            self.in_exploration_mode = True
            self.exploration_rounds = 0
            self.exploration_no_change_count = 0

        return should_stop, should_explore

    def reset_no_change_counters(self):
        if self.in_exploration_mode:
            self.exploration_no_change_count = 0
        else:
            self.no_change_count = 0


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
        boundary_cases = initial_boundary_cases
        pool_for_exploration = initial_pool
        adjusted_results = initial_results.copy()

        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                self.ui.update_loop()

                if self.session.should_stop():
                    break

                remaining = self.session.target_diff - self.session.adjusted_count
                if remaining <= 0:
                    break

                batch_size = min(AGENT_MAX_BOUNDARY_CASES, remaining + 1)
                if self.session.in_exploration_mode:
                    exploration_candidates = self.session.strategy.get_exploration_candidates(
                        adjusted_results, pool_for_exploration, self.session.evaluated_cases
                    )
                    candidate_pool = exploration_candidates + boundary_cases
                else:
                    candidate_pool = boundary_cases

                cases_to_evaluate = self._prepare_next_batch(candidate_pool, batch_size)
                if not cases_to_evaluate:
                    break

                majors = [c.get("major", "") for c in cases_to_evaluate if c.get("major")]
                self.ui.show_candidates(majors)

                decisions = [False] * len(cases_to_evaluate)
                agent_cases: list[dict] = []
                agent_indices: list[int] = []
                for i, case in enumerate(cases_to_evaluate):
                    triage = self.session.strategy.triage_decision(case)
                    if triage is None:
                        agent_cases.append(case)
                        agent_indices.append(i)
                    else:
                        decisions[i] = triage

                if agent_cases:
                    evaluation_result = self._evaluate_batch_with_ui_animation(
                        executor, agent_cases, self.session.strategy.background_major
                    )
                    agent_decisions = (
                        evaluation_result.get("decisions", []) if evaluation_result else []
                    )
                    if len(agent_decisions) != len(agent_cases):
                        agent_decisions = [False] * len(agent_cases)
                    for j, idx in enumerate(agent_indices):
                        decisions[idx] = bool(agent_decisions[j])

                self.session.record_evaluation(cases_to_evaluate)

                needs_adjustment = any(decisions)
                if not needs_adjustment:
                    should_stop, should_explore = self.session.handle_no_adjustment()
                    if should_stop:
                        break
                    if should_explore:
                        continue
                else:
                    self.session.reset_no_change_counters()

                count, new_results = self.session.strategy.update_results(
                    adjusted_results, cases_to_evaluate, decisions, max_adjust=remaining
                )
                adjusted_results = new_results
                self.session.adjusted_count += count

                if self.session.in_exploration_mode:
                    self.session.exploration_rounds += 1
                else:
                    boundary_cases = self.session.strategy.update_boundary_cases(
                        boundary_cases, self.session.evaluated_cases, adjusted_results
                    )

        return adjusted_results

    def _prepare_next_batch(self, candidate_pool: list[dict], batch_size: int) -> list[dict]:
        result: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for c in candidate_pool:
            if not isinstance(c, dict):
                continue
            university = c.get("university")
            major = c.get("major")
            if not isinstance(university, str) or not isinstance(major, str):
                continue
            key = (university, major)
            if key in self.session.evaluated_cases or key in seen:
                continue
            seen.add(key)
            result.append(c)
            if len(result) >= batch_size:
                break
        return result

    def _evaluate_batch_with_ui_animation(
        self, executor: ThreadPoolExecutor, cases_to_evaluate: list[dict], background_major: str
    ):
        future = executor.submit(
            self.agent.evaluate_boundary_cases,
            background_major,
            cases_to_evaluate,
            self.session.mode,
        )
        while not future.done():
            self.ui.update_loop()
            time.sleep(0.3)
        return future.result(timeout=1.0)
