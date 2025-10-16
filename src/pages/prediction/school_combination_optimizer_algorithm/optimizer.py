from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.config import Config
from pymoo.core.result import Result
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from src.pages.prediction.prediction_utils import get_cached_major_similarity
from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG,
    BALANCE_RATIOS,
    GLOBAL_MIN_SIMILARITY,
    MACAU_UNIVERSITIES,
    PlanConfig,
    get_plan_configs,
)
from src.pages.prediction.school_combination_optimizer_algorithm.filters import (
    deduplicate_majors,
    deduplicate_universities_by_similarity,
)
from src.pages.prediction.school_combination_optimizer_algorithm.metrics_calculator import (
    calculate_metrics,
)
from src.pages.prediction.school_combination_optimizer_algorithm.monte_carlo import (
    run_monte_carlo_simulation,
)
from src.pages.prediction.school_combination_optimizer_algorithm.problem import (
    SchoolSelectionProblem,
)
from src.pages.prediction.school_combination_optimizer_algorithm.school_selector import (
    generate_balanced_selection,
    reduce_schools_balanced,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    LRUCache,
    build_major_category_cache,
    build_school_set_key,
    build_selection_key,
    calculate_adaptive_thresholds,
    calibrate_cross_major_probabilities,
    get_cached_reference_directions,
)
from src.utils.app_data_loader import load_school_major_details_df
from src.utils.logger import setup_logger

from .visualizer import (
    visualize_recommendations as standalone_visualize_recommendations,
)

Config.warnings["not_compiled"] = False

logger = setup_logger("page3", "prediction")


@dataclass
class OptimizationContext:
    all_schools_data: list[dict[str, Any]]
    background_major: str
    background_faculty: Optional[str]
    school_level: Optional[str] = None
    gpa: Optional[float] = None
    adaptive_thresholds: Optional[dict[str, float]] = None
    problem: Optional[Any] = None
    major_category_cache: Optional[dict] = None


class SchoolSelectionOptimizer:
    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 50,
        correlation_matrix: pd.DataFrame = None,
        plan_configs: Iterable[PlanConfig] | None = None,
        cache_capacity: int = 256,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.correlation_matrix = correlation_matrix
        self.plan_configs = get_plan_configs(plan_configs)
        self.all_schools_data: list[dict[str, Any]] = []

        self._caches: dict[str, LRUCache] = {
            "result": LRUCache(capacity=cache_capacity),
            "metric": LRUCache(capacity=cache_capacity),
            "simulation": LRUCache(capacity=cache_capacity),
        }

    def _safe_execute(
        self,
        operation: Callable,
        fallback_operation: Callable = None,
        error_message: str = "Operation failed",
    ) -> Any:
        try:
            return operation()
        except Exception as e:
            logger.warning(f"{error_message}: {e}")
            return fallback_operation() if fallback_operation else None

    def _calculate_adaptive_thresholds(self, context: OptimizationContext) -> dict[str, float]:
        def get_probabilities():
            return [school.get("probability", 0.0) for school in context.all_schools_data]

        probabilities = self._safe_execute(
            get_probabilities,
            lambda: [school.get("probability", 0.0) for school in context.all_schools_data],
            "错误计算自适应阈值的概率",
        )

        is_high_bg_high_gpa = (
            context.school_level in {"985", "211", "1-50", "51-100"}
            and context.gpa is not None
            and context.gpa >= 3.2
        )

        if is_high_bg_high_gpa:
            return calculate_adaptive_thresholds(
                probabilities,
                reach_percentile_val=ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG["reach_percentile_val"],
                safety_percentile_val=ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG[
                    "safety_percentile_val"
                ],
            )
        return calculate_adaptive_thresholds(probabilities)

    def _apply_all_filters(
        self, schools_data: list[dict], context: OptimizationContext
    ) -> list[dict]:
        def similarity_filter(schools):
            return deduplicate_universities_by_similarity(
                schools, context.background_major, None, MACAU_UNIVERSITIES
            )

        def similarity_filter_only(schools):
            return [
                s
                for s in schools
                if get_cached_major_similarity(s.get("major", ""), context.background_major, {})
                >= GLOBAL_MIN_SIMILARITY
            ]

        def probability_calibration(schools):
            return calibrate_cross_major_probabilities(schools, context.background_faculty)

        filters = [
            deduplicate_majors,
            similarity_filter,
            similarity_filter_only,
            probability_calibration,
        ]

        filtered_data = schools_data
        for filter_func in filters:
            original_len = len(filtered_data)
            filtered_data = self._safe_execute(
                lambda: filter_func(filtered_data),
                lambda: filtered_data,
                f"过滤器 {filter_func.__name__} 出现错误",
            )
            if len(filtered_data) != original_len:
                break

        return filtered_data

    def _create_problem(
        self,
        schools_data: list[dict],
        plan_config: PlanConfig,
        context: OptimizationContext,
    ) -> SchoolSelectionProblem:
        return SchoolSelectionProblem(
            all_schools_data=schools_data,
            background_major=context.background_major,
            background_faculty=context.background_faculty,
            max_schools=plan_config.max_schools,
            adaptive_thresholds=context.adaptive_thresholds,
            school_level=context.school_level,
            gpa=context.gpa,
            min_schools=plan_config.min_schools,
        )

    def _compute_algo_params(self, problem_size: int) -> tuple[int, int, Any]:
        if problem_size < 10:
            n_ref, pop, n_gen = 10, 20, 20
        elif problem_size < 20:
            n_ref, pop, n_gen = 15, max(15, min(problem_size * 2, 30)), 25
        elif problem_size < 50:
            n_ref, pop, n_gen = 28, max(28, min(problem_size, 40)), 35
        else:
            n_ref, pop, n_gen = (
                42,
                max(42, min(50, self.population_size)),
                min(40, self.n_generations),
            )
        ref = get_cached_reference_directions("energy", n_dim=5, n_points=n_ref)
        return pop, n_gen, ref

    def _run_optimization(self, problem: SchoolSelectionProblem) -> Optional[Result]:
        dynamic_pop_size, dynamic_n_gen, ref_dirs = self._compute_algo_params(
            len(problem.all_schools_data)
        )

        algorithm = NSGA3(
            pop_size=dynamic_pop_size,
            ref_dirs=ref_dirs,
            sampling=BinaryRandomSampling(),
            crossover=HUX(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True,
        )

        return self._safe_execute(
            lambda: minimize(problem, algorithm, ("n_gen", dynamic_n_gen), seed=1, verbose=False),
            error_message="优化过程中出现错误",
        )

    def _get_cached_data(self, cache_type: str, key: str, calculation_func: Callable) -> Any:
        cache = self._caches[cache_type]
        if cached := cache.get(key):
            return cached.copy() if hasattr(cached, "copy") else cached

        result = calculation_func()
        cache.put(key, result.copy() if hasattr(result, "copy") else result)
        return result

    def _calculate_metrics(
        self,
        selected_schools: list[dict],
        context: OptimizationContext,
        problem: SchoolSelectionProblem,
    ) -> dict[str, Any]:
        def calculate_all_metrics():
            metrics = self._get_cached_data(
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

            sim_rej_prob, sim_adm_prob = self._get_cached_data(
                "simulation",
                build_school_set_key(selected_schools),
                lambda: run_monte_carlo_simulation(selected_schools, self.correlation_matrix),
            )

            metrics.update(
                {
                    "simulated_rejection_probability": sim_rej_prob,
                    "simulated_admission_probability": sim_adm_prob,
                }
            )
            return metrics

        return self._safe_execute(calculate_all_metrics, lambda: {}, "错误计算指标")

    def _create_recommendation(
        self,
        schools: list[dict],
        metrics: dict,
        objective_values: list = None,
        rec_type: str = None,
    ) -> dict[str, Any]:
        recommendation: dict[str, Any] = {
            "schools": schools,
            "metrics": metrics,
            "objective_values": objective_values or self._default_objective_values(metrics),
        }
        if rec_type:
            recommendation["type"] = rec_type
        return recommendation

    def _default_objective_values(self, metrics: dict) -> list[float]:
        return [
            metrics.get("rejection_probability", 1.0) * 0.5,
            -metrics.get("diversity", 0),
            -metrics.get("balance_score", -1000),
            -metrics.get("major_similarity", 0),
            -metrics.get("new_major_ratio", 0),
        ]

    def _has_sufficient_schools(self, schools_data: list[dict], min_schools: int) -> bool:
        return len(schools_data) >= min_schools if schools_data else False

    def _find_best_solution_indices(
        self,
        res: Result,
        problem: SchoolSelectionProblem,
        min_schools: int,
        limit: int = 1,
    ) -> list[int]:
        if not hasattr(res, "X") or res.X is None or not hasattr(res, "F") or res.F is None:
            return []

        X, F, CV = res.X, res.F, getattr(res, "CV", None)
        balance_scores = []

        for i in range(len(X)):
            if np.sum(X[i]) < min_schools:
                balance_scores.append(-float("inf"))
                continue

            selected_indices = np.where(X[i] == 1)[0]
            probabilities = [problem.all_schools_data[j]["probability"] for j in selected_indices]
            balance_scores.append(self._calculate_balance_score(probabilities, len(probabilities)))

        feasible_mask = self._get_feasible_mask(res, len(X))
        candidate_indices = (
            np.arange(len(X))[feasible_mask] if feasible_mask is not None else np.arange(len(X))
        )

        if not len(candidate_indices):
            return []

        return self._sort_and_select_candidates(candidate_indices, balance_scores, F, limit)

    def _calculate_balance_score(self, probabilities: list[float], total: int) -> float:
        if not self.context.adaptive_thresholds:
            return 0.0
        safety_thresh = self.context.adaptive_thresholds["safety"]
        target_thresh = self.context.adaptive_thresholds["target_lower"]

        safety = sum(1 for p in probabilities if p >= safety_thresh)
        target = sum(1 for p in probabilities if target_thresh <= p < safety_thresh)
        reach = sum(1 for p in probabilities if p < target_thresh)

        ideal_safety = total * BALANCE_RATIOS["safety"]
        ideal_target = total * BALANCE_RATIOS["target"]
        ideal_reach = total * BALANCE_RATIOS["reach"]

        return -(
            (safety - ideal_safety) ** 2 + (target - ideal_target) ** 2 + (reach - ideal_reach) ** 2
        )

    def _get_feasible_mask(self, res: Result, n_solutions: int) -> Optional[np.ndarray]:
        if hasattr(res, "CV") and res.CV is not None:
            return (res.CV <= 0).flatten() if hasattr(res.CV, "flatten") else (res.CV <= 0)
        elif hasattr(res, "G") and res.G is not None:
            return np.all(res.G <= 0, axis=1)
        return None

    def _sort_and_select_candidates(
        self,
        candidate_indices: np.ndarray,
        balance_scores: list,
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

    def _build_final_recommendation(
        self,
        x: np.ndarray,
        f_values: np.ndarray,
        problem: SchoolSelectionProblem,
        context: OptimizationContext,
        plan_config: PlanConfig,
    ) -> Optional[dict[str, Any]]:
        selected_indices = np.where(x == 1)[0]
        selected_schools = [problem.all_schools_data[j] for j in selected_indices]

        selected_schools = self._apply_post_filters(selected_schools, context, problem)

        selected_schools = self._enforce_school_limits(
            selected_schools, plan_config.min_schools, plan_config.max_schools
        )
        if not selected_schools:
            return None

        metrics = self._calculate_metrics(selected_schools, context, problem)

        return self._create_recommendation(selected_schools, metrics, f_values.tolist())

    def _apply_post_filters(
        self,
        schools: list[dict],
        context: OptimizationContext,
        problem: SchoolSelectionProblem,
    ) -> list[dict]:
        if not context.background_faculty or not context.adaptive_thresholds:
            return schools

        reach_threshold = context.adaptive_thresholds.get("target_lower", 0.0)
        filtered_selection = []

        for school in schools:
            is_reach = school.get("probability", 1.0) < reach_threshold
            if not is_reach:
                filtered_selection.append(school)
                continue

            uni = school.get("university", "")
            major = school.get("major", "")
            cache_key = f"{uni}|{major}"
            target_faculty = problem.major_category_cache.get(cache_key)

            if target_faculty == context.background_faculty:
                filtered_selection.append(school)

        return filtered_selection

    def _enforce_school_limits(
        self, schools: list[dict], min_schools: int, max_schools: int
    ) -> list[dict]:
        num_selected = len(schools)

        if num_selected < min_schools:
            return []
        elif num_selected > max_schools:
            return reduce_schools_balanced(schools, max_schools, self.context.adaptive_thresholds)

        return schools

    def _get_fallback_recommendation(
        self, context: OptimizationContext, plan_config: PlanConfig
    ) -> Optional[dict[str, Any]]:
        balanced_schools = generate_balanced_selection(
            context.all_schools_data,
            plan_config.min_schools,
            plan_config.max_schools,
            context.adaptive_thresholds,
        )

        if not balanced_schools or len(balanced_schools) < plan_config.min_schools:
            return None

        if not context.problem:
            return None

        metrics = self._calculate_metrics(balanced_schools, context, context.problem)
        return self._create_recommendation(balanced_schools, metrics)

    def _optimize_single_plan(
        self, plan_config: PlanConfig, context: OptimizationContext
    ) -> Optional[dict[str, Any]]:
        filtered_schools = self._apply_all_filters(context.all_schools_data, context)

        if not self._has_sufficient_schools(filtered_schools, plan_config.min_schools):
            return None

        problem = self._create_problem(filtered_schools, plan_config, context)
        context.problem = problem

        result = self._run_optimization(problem)

        if result and hasattr(result, "X"):
            best_indices = self._find_best_solution_indices(
                result, problem, plan_config.min_schools
            )

            for idx in best_indices:
                recommendation = self._build_final_recommendation(
                    result.X[idx], result.F[idx], problem, context, plan_config
                )
                if recommendation:
                    return recommendation

        return self._get_fallback_recommendation(context, plan_config)

    def clear_cache(self):
        for cache in self._caches.values():
            cache.clear()

    def optimize(
        self,
        all_schools_data: list[dict[str, Any]],
        background_major: str,
        background_faculty: Optional[str] = None,
        school_level: str = None,
        gpa: float = None,
        major_category_cache: Optional[dict] = None,
        bg_target_similarity_cache: Optional[dict] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if not all_schools_data:
            return [], {}

        self.all_schools_data = all_schools_data

        if major_category_cache is None:
            details_df = load_school_major_details_df()
            major_category_cache = build_major_category_cache(details_df)

        self.context = OptimizationContext(
            all_schools_data=all_schools_data,
            background_major=background_major,
            background_faculty=background_faculty,
            school_level=school_level,
            gpa=gpa,
            major_category_cache=major_category_cache,
        )

        self.bg_target_similarity_cache = bg_target_similarity_cache or {}

        self.context.adaptive_thresholds = self._calculate_adaptive_thresholds(self.context)

        final_recommendations = []
        for plan_config in get_plan_configs(self.plan_configs):
            recommendation = self._optimize_single_plan(plan_config, self.context)
            if recommendation:
                recommendation["type"] = plan_config.name
                final_recommendations.append(recommendation)

        return final_recommendations, self.context.adaptive_thresholds

    def visualize_recommendations(
        self,
        recommendations: list[dict[str, Any]],
        adaptive_thresholds: dict[str, float],
    ) -> None:
        standalone_visualize_recommendations(recommendations, adaptive_thresholds)
