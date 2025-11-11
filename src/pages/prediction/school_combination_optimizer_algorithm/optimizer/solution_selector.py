from typing import Optional

import numpy as np
from pymoo.core.result import Result

from src.pages.prediction.school_combination_optimizer_algorithm.config import BALANCE_RATIOS
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.context import (
    OptimizationContext,
)
from src.pages.prediction.school_combination_optimizer_algorithm.problem import (
    SchoolSelectionProblem,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    clip_probability,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def get_feasible_mask(res: Result, n_solutions: int) -> Optional[np.ndarray]:
    if hasattr(res, "CV") and res.CV is not None:
        return (res.CV <= 0).flatten() if hasattr(res.CV, "flatten") else (res.CV <= 0)
    elif hasattr(res, "G") and res.G is not None:
        return np.all(res.G <= 0, axis=1)
    return None


def sort_and_select_candidates(
    candidate_indices: np.ndarray,
    balance_scores: list[float],
    F: np.ndarray,
    limit: int,
) -> list[int]:
    sortable_balance = np.array(balance_scores)[candidate_indices]

    if F.shape[0] == len(balance_scores) and F.shape[1] >= 1:
        f0 = F[candidate_indices, 0]
        f_sim = F[candidate_indices, 3] if F.shape[1] >= 4 else np.zeros_like(sortable_balance)
        local_order = np.lexsort((f0, -sortable_balance, f_sim))
        return candidate_indices[local_order[:limit]].tolist()
    else:
        sorted_indices = np.argsort(sortable_balance)[::-1]
        return candidate_indices[sorted_indices[:limit]].tolist()


def find_best_solution_indices(
    res: Result,
    problem: SchoolSelectionProblem,
    context: Optional[OptimizationContext],
    min_schools: int,
    limit: int = 1,
) -> list[int]:
    if not hasattr(res, "X") or res.X is None or not hasattr(res, "F") or res.F is None:
        return []

    X, F = res.X, res.F
    n_solutions, n_candidates = X.shape[0], X.shape[1]

    selected_counts = np.sum(X, axis=1)

    balance_scores = np.full(n_solutions, -np.inf, dtype=float)

    if context and context.adaptive_thresholds:
        probs_vec = np.array(
            [
                clip_probability(s.get("probability", 0.0))
                for s in problem.all_schools_data[:n_candidates]
            ],
            dtype=float,
        )
        safety_thresh = context.adaptive_thresholds.get("safety", 0.75)
        target_thresh = context.adaptive_thresholds.get("target_lower", 0.55)

        safety_mask = (probs_vec >= safety_thresh).astype(int)
        target_mask = ((probs_vec >= target_thresh) & (probs_vec < safety_thresh)).astype(int)
        reach_mask = (probs_vec < target_thresh).astype(int)

        safety_counts = X @ safety_mask
        target_counts = X @ target_mask
        reach_counts = X @ reach_mask

        ideal_safety = selected_counts * BALANCE_RATIOS["safety"]
        ideal_target = selected_counts * BALANCE_RATIOS["target"]
        ideal_reach = selected_counts * BALANCE_RATIOS["reach"]

        balance_scores = -(
            (safety_counts - ideal_safety) ** 2
            + (target_counts - ideal_target) ** 2
            + (reach_counts - ideal_reach) ** 2
        )

    balance_scores[selected_counts < min_schools] = -np.inf

    feasible_mask = get_feasible_mask(res, n_solutions)
    candidate_indices = (
        np.arange(n_solutions)[feasible_mask]
        if feasible_mask is not None
        else np.arange(n_solutions)
    )

    if candidate_indices.size == 0:
        return []

    return sort_and_select_candidates(candidate_indices, balance_scores.tolist(), F, limit)
