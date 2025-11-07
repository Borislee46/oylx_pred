from typing import Any, Callable, Optional

import pandas as pd

from src.pages.prediction.school_combination_optimizer_algorithm.config import BALANCE_RATIOS
from src.pages.prediction.school_combination_optimizer_algorithm.metrics_calculator import (
    calculate_metrics,
)
from src.pages.prediction.school_combination_optimizer_algorithm.monte_carlo import (
    run_monte_carlo_simulation,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.context import (
    OptimizationContext,
)
from src.pages.prediction.school_combination_optimizer_algorithm.problem import (
    SchoolSelectionProblem,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    build_school_set_key,
    build_selection_key,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def calculate_metrics_for_selection(
    selected_schools: list[dict[str, Any]],
    context: OptimizationContext,
    problem: SchoolSelectionProblem,
    correlation_matrix: Optional[pd.DataFrame],
    get_cached_data: Callable[[str, str, Callable[[], Any]], Any],
) -> dict[str, Any]:
    def calculate_all_metrics():
        metrics = get_cached_data(
            "metric",
            build_selection_key(context.background_major, selected_schools),
            lambda: calculate_metrics(
                selected_schools,
                context.background_major,
                context.adaptive_thresholds,
                bg_target_similarity_cache=problem.bg_target_similarity_cache_data,
                new_major_cache=problem.new_major_cache,
                background_faculty=problem.background_faculty,
                major_category_cache=problem.major_category_cache,
            ),
        )
        sim_rej_prob, sim_adm_prob = get_cached_data(
            "simulation",
            build_school_set_key(selected_schools),
            lambda: run_monte_carlo_simulation(selected_schools, correlation_matrix),
        )
        metrics.update(
            {
                "simulated_rejection_probability": sim_rej_prob,
                "simulated_admission_probability": sim_adm_prob,
            }
        )
        return metrics

    return calculate_all_metrics()


def calculate_balance_score(
    probabilities: list[float],
    total: int,
    adaptive_thresholds: Optional[dict[str, float]],
) -> float:
    if not adaptive_thresholds:
        return 0.0
    safety_thresh = adaptive_thresholds["safety"]
    target_thresh = adaptive_thresholds["target_lower"]

    safety = sum(1 for p in probabilities if p >= safety_thresh)
    target = sum(1 for p in probabilities if target_thresh <= p < safety_thresh)
    reach = sum(1 for p in probabilities if p < target_thresh)

    ideal_safety = total * BALANCE_RATIOS["safety"]
    ideal_target = total * BALANCE_RATIOS["target"]
    ideal_reach = total * BALANCE_RATIOS["reach"]

    return -(
        (safety - ideal_safety) ** 2 + (target - ideal_target) ** 2 + (reach - ideal_reach) ** 2
    )

