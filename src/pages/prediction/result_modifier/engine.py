import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Set, Tuple

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

        self.evaluated_cases: Set[Tuple[str, str]] = set()
        self.stop_requested = False

    def record_evaluation(self, cases: List[Dict]):
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
        if self.exploration_rounds > AGENT_EXPLORATION_MAX_ROUNDS:
            logger.info(f"达到最大探索轮数 {AGENT_EXPLORATION_MAX_ROUNDS}")
            return True
        return False

    def handle_no_adjustment(self) -> Tuple[bool, bool]:
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
            self.exploration_rounds = 1
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
        initial_boundary_cases: List[Dict],
        initial_pool: List[Dict],
        initial_results: List[Dict],
    ) -> List[Dict]:
        boundary_cases = initial_boundary_cases[
            : min(self.session.target_diff, len(initial_boundary_cases), AGENT_MAX_BOUNDARY_CASES)
        ]
        pool_for_exploration = initial_pool
        adjusted_results = initial_results.copy()

        while boundary_cases:
            self.ui.update_loop()

            if self.session.should_stop():
                break

            cases_to_evaluate = self._prepare_next_batch(boundary_cases)
            if not cases_to_evaluate:
                break

            majors = [c.get("major", "") for c in cases_to_evaluate if c.get("major")]
            self.ui.show_candidates(majors)

            evaluation_result = self._evaluate_batch_with_ui_animation(
                cases_to_evaluate, self.session.strategy.background_major
            )

            if not evaluation_result:
                continue

            decisions = evaluation_result.get("decisions", [])
            needs_adjustment = evaluation_result.get("needs_adjustment", False)

            self.session.record_evaluation(cases_to_evaluate)

            if not needs_adjustment:
                should_stop, should_explore = self.session.handle_no_adjustment()

                if should_stop:
                    break

                if should_explore:
                    next_candidates = self.session.strategy.get_exploration_candidates(
                        adjusted_results, pool_for_exploration, self.session.evaluated_cases
                    )
                    if not next_candidates:
                        break
                    boundary_cases = next_candidates
                    continue
            else:
                self.session.reset_no_change_counters()

            count, new_results = self.session.strategy.update_results(
                adjusted_results, cases_to_evaluate, decisions
            )
            adjusted_results = new_results
            self.session.adjusted_count += count

            if self.session.in_exploration_mode:
                self.session.exploration_rounds += 1
                boundary_cases = self.session.strategy.get_exploration_candidates(
                    adjusted_results, pool_for_exploration, self.session.evaluated_cases
                )
            else:
                boundary_cases = self.session.strategy.update_boundary_cases(
                    boundary_cases, self.session.evaluated_cases, adjusted_results
                )

        return adjusted_results

    def _prepare_next_batch(self, boundary_cases):
        cases = [
            c
            for c in boundary_cases
            if (
                isinstance(c.get("university"), str)
                and isinstance(c.get("major"), str)
                and (c.get("university"), c.get("major")) not in self.session.evaluated_cases
            )
        ]
        return cases[:AGENT_MAX_BOUNDARY_CASES]

    def _evaluate_batch_with_ui_animation(self, cases_to_evaluate, background_major):
        with ThreadPoolExecutor(max_workers=1) as executor:
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
