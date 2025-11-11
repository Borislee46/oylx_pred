from typing import Any, Callable, Optional

import numpy as np

from src.pages.prediction.school_combination_optimizer_algorithm.config import PlanConfig
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.context import (
    OptimizationContext,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.filters_handler import (
    apply_post_filters,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.metrics_calculator_wrapper import (
    calculate_metrics_for_selection,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.school_adjuster import (
    adjust_probability_by_university_difficulty,
    enforce_school_limits,
)
from src.pages.prediction.school_combination_optimizer_algorithm.problem import (
    SchoolSelectionProblem,
)
from src.pages.prediction.school_combination_optimizer_algorithm.school_selector import (
    generate_balanced_selection,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def default_objective_values(metrics: dict[str, Any]) -> list[float]:
    return [
        metrics.get("rejection_probability", 1.0),
        -metrics.get("diversity", 0),
        -metrics.get("balance_score", -1000),
        -metrics.get("major_similarity", 0),
        -metrics.get("new_major_ratio", 0),
    ]


def create_recommendation(
    schools: list[dict[str, Any]],
    metrics: dict[str, Any],
    objective_values: Optional[list[float]] = None,
    rec_type: Optional[str] = None,
) -> dict[str, Any]:
    recommendation: dict[str, Any] = {
        "schools": schools,
        "metrics": metrics,
        "objective_values": objective_values or default_objective_values(metrics),
    }
    if rec_type:
        recommendation["type"] = rec_type
    return recommendation


def build_final_recommendation(
    x: np.ndarray,
    f_values: np.ndarray,
    problem: SchoolSelectionProblem,
    context: OptimizationContext,
    plan_config: PlanConfig,
    correlation_matrix: Any,
    get_cached_data: Callable[[str, str, Callable[[], Any]], Any],
) -> Optional[dict[str, Any]]:
    selected_indices = np.where(x == 1)[0]
    selected_schools = [problem.all_schools_data[j] for j in selected_indices]

    selected_schools = apply_post_filters(selected_schools, context, problem)

    selected_schools = enforce_school_limits(
        selected_schools,
        plan_config.min_schools,
        plan_config.max_schools,
        context.adaptive_thresholds,
    )
    if not selected_schools:
        return None

    selected_schools = adjust_probability_by_university_difficulty(
        selected_schools, context.adaptive_thresholds
    )

    metrics = calculate_metrics_for_selection(
        selected_schools, context, problem, correlation_matrix, get_cached_data
    )

    return create_recommendation(selected_schools, metrics, f_values.tolist())


def _build_fallback_recommendation_from_schools(
    schools: list[dict[str, Any]],
    min_schools: int,
    max_schools: int,
    context: OptimizationContext,
    plan_config: PlanConfig,
    correlation_matrix: Any,
    get_cached_data: Callable[[str, str, Callable[[], Any]], Any],
) -> Optional[dict[str, Any]]:
    balanced_schools = generate_balanced_selection(
        schools,
        min_schools,
        max_schools,
        context.adaptive_thresholds,
    )

    if not balanced_schools or len(balanced_schools) < min_schools:
        return None

    if not context.problem:
        return None

    balanced_schools = adjust_probability_by_university_difficulty(
        balanced_schools, context.adaptive_thresholds
    )
    metrics = calculate_metrics_for_selection(
        balanced_schools, context, context.problem, correlation_matrix, get_cached_data
    )
    return create_recommendation(balanced_schools, metrics)


def get_fallback_recommendation(
    context: OptimizationContext,
    plan_config: PlanConfig,
    correlation_matrix: Any,
    get_cached_data: Callable[[str, str, Callable[[], Any]], Any],
) -> Optional[dict[str, Any]]:
    return _build_fallback_recommendation_from_schools(
        context.all_schools_data,
        plan_config.min_schools,
        plan_config.max_schools,
        context,
        plan_config,
        correlation_matrix,
        get_cached_data,
    )


def get_fallback_recommendation_with_filtered_schools(
    filtered_schools: list[dict[str, Any]],
    context: OptimizationContext,
    plan_config: PlanConfig,
    correlation_matrix: Any,
    get_cached_data: Callable[[str, str, Callable[[], Any]], Any],
) -> Optional[dict[str, Any]]:
    available_count = len(filtered_schools)
    adjusted_min_schools = min(plan_config.min_schools, available_count)
    adjusted_max_schools = min(plan_config.max_schools, available_count)

    return _build_fallback_recommendation_from_schools(
        filtered_schools,
        adjusted_min_schools,
        adjusted_max_schools,
        context,
        plan_config,
        correlation_matrix,
        get_cached_data,
    )
