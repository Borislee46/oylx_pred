from typing import Any, Iterable

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.config import Config
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from src.pages.prediction.prediction_utils import get_cached_major_similarity
from src.pages.prediction.school_combination_optimizer_algorithm.adaptive_admission_probability_threshold import (
    calculate_adaptive_thresholds,
)
from src.pages.prediction.school_combination_optimizer_algorithm.cache_utils import (
    LRUCache,
    build_school_set_key,
    build_selection_key,
)
from src.pages.prediction.school_combination_optimizer_algorithm.faculty_based_filter import (
    filter_schools_by_faculty_rules,
)
from src.pages.prediction.school_combination_optimizer_algorithm.monte_carlo import (
    run_monte_carlo_simulation,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer_config import (
    BALANCE_RATIOS,
    GLOBAL_MIN_SIMILARITY,
)
from src.pages.prediction.school_combination_optimizer_algorithm.plan_config import (
    PlanConfig,
    get_plan_configs,
)
from src.pages.prediction.school_combination_optimizer_algorithm.pre_filter import (
    deduplicate_majors,
)
from src.pages.prediction.school_combination_optimizer_algorithm.probability_utils import (
    calibrate_cross_major_probabilities,
)
from src.pages.prediction.school_combination_optimizer_algorithm.problem import (
    SchoolSelectionProblem,
)
from src.pages.prediction.school_combination_optimizer_algorithm.reference_direction_cache import (
    get_cached_reference_directions,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    calculate_metrics,
    generate_balanced_selection,
    reduce_schools_balanced,
)
from src.utils.logger import setup_logger

Config.warnings["not_compiled"] = False

logger = setup_logger("page3", "prediction")


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
        self._result_cache: LRUCache[str, Any] = LRUCache(capacity=cache_capacity)
        self._metric_cache: LRUCache[str, dict[str, Any]] = LRUCache(capacity=cache_capacity)
        self._simulation_cache: LRUCache[str, tuple[float, float]] = LRUCache(
            capacity=cache_capacity
        )

    def clear_cache(self):
        self._result_cache.clear()
        self._metric_cache.clear()
        self._simulation_cache.clear()

    def optimize(
        self,
        all_schools_data: list[dict[str, Any]],
        background_major: str,
        background_faculty: str | None,
        school_level: str = None,
        gpa: float = None,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if not all_schools_data:
            return [], {}

        self.all_schools_data = all_schools_data

        if background_faculty:
            from src.utils.app_data_loader import load_school_major_details_df

            from .faculty_based_filter import filter_schools_by_faculty_rules
            from .problem_initializer import build_major_category_cache

            details_df = load_school_major_details_df()
            major_category_cache = build_major_category_cache(details_df)

            filtered_schools = filter_schools_by_faculty_rules(
                all_schools_data,
                background_faculty,
                major_category_cache,
            )

            probabilities_for_threshold = [
                school.get("probability", 0.0) for school in filtered_schools
            ]

        else:
            probabilities_for_threshold = [
                school.get("probability", 0.0) for school in all_schools_data
            ]

        is_high_bg_high_gpa = (
            school_level in {"985", "211", "1-50", "51-100"} and gpa is not None and gpa >= 3.2
        )

        if is_high_bg_high_gpa:
            from .optimizer_config import ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG

            adaptive_thresholds = calculate_adaptive_thresholds(
                probabilities_for_threshold,
                reach_percentile_val=ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG["reach_percentile_val"],
                safety_percentile_val=ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG[
                    "safety_percentile_val"
                ],
            )
        else:
            adaptive_thresholds = calculate_adaptive_thresholds(probabilities_for_threshold)

        plan_types = get_plan_configs(self.plan_configs)

        final_recommendations = []
        for plan_type in plan_types:
            recommendation = self._optimize_for_plan(
                plan_type,
                all_schools_data,
                background_major,
                background_faculty,
                school_level,
                gpa,
                adaptive_thresholds,
            )
            if recommendation:
                recommendation["type"] = plan_type.name
                final_recommendations.append(recommendation)

        return final_recommendations, adaptive_thresholds

    def _optimize_for_plan(
        self,
        plan_config: PlanConfig,
        all_schools_data: list[dict[str, Any]],
        background_major: str,
        background_faculty: str | None,
        school_level: str | None,
        gpa: float | None,
        adaptive_thresholds: dict[str, float],
    ) -> dict[str, Any] | None:
        min_schools = plan_config.min_schools
        if not all_schools_data or len(all_schools_data) < min_schools:
            return None

        all_schools_data = deduplicate_majors(all_schools_data)

        problem = SchoolSelectionProblem(
            all_schools_data,
            background_major,
            background_faculty,
            plan_config.max_schools,
            adaptive_thresholds,
            school_level=school_level,
            gpa=gpa,
            min_schools=min_schools,
        )

        all_schools_data = problem.all_schools_data

        try:
            from .optimizer_config import MACAU_UNIVERSITIES
            from .pre_filter import deduplicate_universities_by_similarity

            filtered_once = deduplicate_universities_by_similarity(
                schools=all_schools_data,
                background_major=background_major,
                similarity_cache=problem.bg_target_similarity_cache_data,
                target_universities=MACAU_UNIVERSITIES,
            )
            if len(filtered_once) != len(all_schools_data):
                all_schools_data = filtered_once
                try:
                    problem.close()
                except Exception:
                    pass
                problem = SchoolSelectionProblem(
                    all_schools_data,
                    background_major,
                    background_faculty,
                    plan_config.max_schools,
                    adaptive_thresholds,
                    school_level=school_level,
                    gpa=gpa,
                    min_schools=min_schools,
                )
        except Exception:
            pass

        def _compute_algo_params(problem_size: int) -> tuple[int, int, Any]:
            n_ref = 42
            ref = get_cached_reference_directions("energy", n_dim=6, n_points=n_ref)
            if problem_size < 10:
                n_ref = 10
                ref = get_cached_reference_directions("energy", n_dim=6, n_points=n_ref)
                pop = max(n_ref, 20)
                n_gen = 20
            elif problem_size < 20:
                n_ref = 15
                ref = get_cached_reference_directions("energy", n_dim=6, n_points=n_ref)
                pop = max(n_ref, min(problem_size * 2, 30))
                n_gen = 25
            elif problem_size < 50:
                n_ref = 28
                ref = get_cached_reference_directions("energy", n_dim=6, n_points=n_ref)
                pop = max(n_ref, min(problem_size, 40))
                n_gen = 35
            else:
                pop = max(n_ref, min(50, self.population_size))
                n_gen = min(40, self.n_generations)
            return pop, n_gen, ref

        dynamic_pop_size, dynamic_n_gen, ref_dirs = _compute_algo_params(len(all_schools_data))

        algorithm = NSGA3(
            pop_size=dynamic_pop_size,
            ref_dirs=ref_dirs,
            sampling=BinaryRandomSampling(),
            crossover=HUX(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True,
        )

        try:
            filtered = filter_schools_by_faculty_rules(
                all_schools_data,
                problem.background_faculty,
                problem.major_category_cache,
            )

            similarity_filtered = []
            cache = problem.bg_target_similarity_cache_data or {}
            for s in filtered:
                target_major = s.get("major", "")
                if not target_major:
                    similarity_filtered.append(s)
                    continue

                sim = get_cached_major_similarity(
                    target_major=target_major,
                    background_major=background_major,
                    cache=cache,
                )

                cache_key = f"{s.get('university', '')}|{target_major}"
                target_faculty = problem.major_category_cache.get(cache_key, "未知学院")
                logger.debug(f"专业: {target_major} (学院: {target_faculty}) - 相似度: {sim:.4f}")

                if sim >= GLOBAL_MIN_SIMILARITY:
                    similarity_filtered.append(s)

            calibrated = calibrate_cross_major_probabilities(
                similarity_filtered,
                problem.background_faculty,
                problem.major_category_cache,
            )

            if calibrated:
                all_schools_data = calibrated

                try:
                    problem.close()
                except Exception:
                    pass
                problem = SchoolSelectionProblem(
                    all_schools_data,
                    background_major,
                    background_faculty,
                    plan_config.max_schools,
                    adaptive_thresholds,
                    school_level=school_level,
                    gpa=gpa,
                    min_schools=min_schools,
                )
                dynamic_pop_size, dynamic_n_gen, ref_dirs = _compute_algo_params(
                    len(all_schools_data)
                )

            res = minimize(problem, algorithm, ("n_gen", dynamic_n_gen), seed=1, verbose=False)
            recommendations = self._process_results(
                res,
                all_schools_data,
                background_major,
                background_faculty,
                min_schools,
                plan_config.max_schools,
                self.correlation_matrix,
                adaptive_thresholds,
                problem,
                limit=1,
            )
            if recommendations:
                return recommendations[0]
        except Exception as e:
            logger.error(f"处理优化结果时发生错误: {e}")
        finally:
            try:
                problem.close()
            except Exception:
                pass

        return self._get_fallback_recommendation(
            all_schools_data,
            background_major,
            background_faculty,
            min_schools,
            plan_config.max_schools,
            adaptive_thresholds,
            problem,
        )

    def _get_fallback_recommendation(
        self,
        all_schools_data: list[dict[str, Any]],
        background_major: str,
        background_faculty: str | None,
        min_schools: int,
        max_schools: int,
        adaptive_thresholds: dict[str, float],
        problem: SchoolSelectionProblem,
    ) -> dict[str, Any] | None:
        balanced_schools = generate_balanced_selection(
            all_schools_data, min_schools, max_schools, adaptive_thresholds
        )
        if not balanced_schools or len(balanced_schools) < min_schools:
            return None

        cache_key = build_selection_key(background_major, balanced_schools)
        cached_metrics = self._metric_cache.get(cache_key)
        if cached_metrics is not None:
            metrics = cached_metrics.copy()
        else:
            metrics = calculate_metrics(
                balanced_schools,
                background_major,
                adaptive_thresholds,
                bg_target_similarity_cache=problem.bg_target_similarity_cache_data,
                new_major_cache=problem.new_major_cache,
                background_faculty=problem.background_faculty,
                major_category_cache=problem.major_category_cache,
            )
            self._metric_cache.put(cache_key, metrics.copy())

        sim_cache_key = build_school_set_key(balanced_schools)
        cached_sim = self._simulation_cache.get(sim_cache_key)
        if cached_sim is not None:
            sim_rej_prob, sim_adm_prob = cached_sim
        else:
            sim_rej_prob, sim_adm_prob = run_monte_carlo_simulation(
                balanced_schools, self.correlation_matrix
            )
            self._simulation_cache.put(sim_cache_key, (sim_rej_prob, sim_adm_prob))

        metrics["simulated_rejection_probability"] = sim_rej_prob
        metrics["simulated_admission_probability"] = sim_adm_prob

        return {
            "schools": balanced_schools,
            "metrics": metrics,
            "objective_values": [
                metrics.get("rejection_probability", 1.0) * 0.5,
                -metrics.get("diversity", 0),
                -metrics.get("balance_score", -1000),
                -metrics.get("major_similarity", 0),
                -metrics.get("new_major_ratio", 0),
                -metrics.get("major_category_score", 0),
            ],
        }

    def _process_results(
        self,
        res,
        all_schools_data: list[dict[str, Any]],
        background_major: str,
        background_faculty: str | None,
        min_schools: int,
        max_schools: int,
        correlation_matrix: pd.DataFrame,
        adaptive_thresholds: dict[str, float],
        problem: SchoolSelectionProblem,
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        if res is None or getattr(res, "X", None) is None or getattr(res, "F", None) is None:
            fallback = self._get_fallback_recommendation(
                all_schools_data,
                background_major,
                background_faculty,
                min_schools,
                max_schools,
                adaptive_thresholds,
                problem,
            )
            return [fallback] if fallback else []

        top_indices = self._find_best_solution_indices(
            res, all_schools_data, min_schools, adaptive_thresholds, limit
        )
        recommendations = []
        for idx in top_indices:
            recommendation = self._finalize_recommendation(
                res.X[idx],
                res.F[idx],
                all_schools_data,
                background_major,
                background_faculty,
                min_schools,
                max_schools,
                correlation_matrix,
                adaptive_thresholds,
                problem,
            )
            if recommendation:
                recommendations.append(recommendation)

        if not recommendations:
            fallback = self._get_fallback_recommendation(
                all_schools_data,
                background_major,
                background_faculty,
                min_schools,
                max_schools,
                adaptive_thresholds,
                problem,
            )
            if fallback:
                recommendations.append(fallback)

        feasible = [r for r in recommendations if r.get("objective_values") is not None]
        if feasible:
            return sorted(
                feasible,
                key=lambda x: x["metrics"].get("simulated_rejection_probability", 1.0),
            )
        else:
            fallback = self._get_fallback_recommendation(
                all_schools_data,
                background_major,
                background_faculty,
                min_schools,
                max_schools,
                adaptive_thresholds,
                problem,
            )
            return [fallback] if fallback else []

    def _find_best_solution_indices(
        self,
        res,
        all_schools_data: list[dict[str, Any]],
        min_schools: int,
        adaptive_thresholds: dict[str, float],
        limit: int,
    ) -> list[int]:
        X, F, CV = res.X, res.F, getattr(res, "CV", None)
        balance_scores = []

        for i in range(len(X)):
            if np.sum(X[i]) < min_schools:
                balance_scores.append(-float("inf"))
                continue
            selected_indices = np.where(X[i] == 1)[0]
            probabilities = [all_schools_data[j]["probability"] for j in selected_indices]

            safety_thresh = adaptive_thresholds["safety"]
            target_thresh = adaptive_thresholds["target_lower"]
            safety = sum(1 for p in probabilities if p >= safety_thresh)
            target = sum(1 for p in probabilities if target_thresh <= p < safety_thresh)
            reach = sum(1 for p in probabilities if p < target_thresh)
            total = len(probabilities)

            ideal_safety = total * BALANCE_RATIOS["safety"]
            ideal_target = total * BALANCE_RATIOS["target"]
            ideal_reach = total * BALANCE_RATIOS["reach"]

            score = -(
                (safety - ideal_safety) ** 2
                + (target - ideal_target) ** 2
                + (reach - ideal_reach) ** 2
            )
            balance_scores.append(score)

        if not balance_scores or all(bs == -float("inf") for bs in balance_scores):
            return []

        feasible_mask = None
        if CV is not None:
            feasible_mask = (CV <= 0).flatten() if hasattr(CV, "flatten") else (CV <= 0)
        elif getattr(res, "G", None) is not None:
            feasible_mask = np.all(res.G <= 0, axis=1)

        indices_all = np.arange(len(X))
        candidate_indices = (
            indices_all[feasible_mask]
            if feasible_mask is not None and np.any(feasible_mask)
            else indices_all
        )

        if F is not None and len(F) > 0 and len(candidate_indices) > 0:
            sortable_balance = np.array(balance_scores)[candidate_indices]
            f0 = (
                F[candidate_indices, 0]
                if F.shape[0] == len(X) and F.shape[1] >= 1
                else np.zeros_like(sortable_balance)
            )
            if F.shape[1] >= 4:
                f_sim = F[candidate_indices, 3]
            else:
                f_sim = np.zeros_like(sortable_balance)
            local_order = np.lexsort((f0, -sortable_balance, f_sim))
            return candidate_indices[local_order[:limit]].tolist()
        else:
            sorted_indices = np.argsort(balance_scores)[::-1]
            candidate_set = set(candidate_indices)
            sorted_candidates = [idx for idx in sorted_indices if idx in candidate_set]
            return sorted_candidates[:limit]

    def _finalize_recommendation(
        self,
        x: np.ndarray,
        f_values: np.ndarray,
        all_schools_data: list[dict[str, Any]],
        background_major: str,
        background_faculty: str | None,
        min_schools: int,
        max_schools: int,
        correlation_matrix: pd.DataFrame,
        adaptive_thresholds: dict[str, float],
        problem: SchoolSelectionProblem,
    ) -> dict[str, Any] | None:
        selected_indices = np.where(x == 1)[0]
        selected_schools = [all_schools_data[j] for j in selected_indices]

        if background_faculty and adaptive_thresholds:
            reach_threshold = adaptive_thresholds.get("target_lower", 0.0)

            filtered_selection = []
            for school in selected_schools:
                is_reach = school.get("probability", 1.0) < reach_threshold
                if not is_reach:
                    filtered_selection.append(school)
                    continue

                uni = school.get("university", "")
                major = school.get("major", "")
                cache_key = f"{uni}|{major}"
                target_faculty = problem.major_category_cache.get(cache_key)

                if target_faculty == background_faculty:
                    filtered_selection.append(school)

            selected_schools = filtered_selection

        num_selected = len(selected_schools)

        if num_selected < min_schools:
            return None

        if num_selected > max_schools:
            selected_schools = reduce_schools_balanced(
                selected_schools, max_schools, adaptive_thresholds
            )
            if len(selected_schools) < min_schools:
                return None

        cache_key = build_selection_key(background_major, selected_schools)
        cached_metrics = self._metric_cache.get(cache_key)
        if cached_metrics is not None:
            metrics = cached_metrics.copy()
        else:
            metrics = calculate_metrics(
                selected_schools,
                background_major,
                adaptive_thresholds,
                bg_target_similarity_cache=problem.bg_target_similarity_cache_data,
                new_major_cache=problem.new_major_cache,
                background_faculty=problem.background_faculty,
                major_category_cache=problem.major_category_cache,
            )
            self._metric_cache.put(cache_key, metrics.copy())

        sim_cache_key = build_school_set_key(selected_schools)
        cached_sim = self._simulation_cache.get(sim_cache_key)
        if cached_sim is not None:
            sim_rej_prob, sim_adm_prob = cached_sim
        else:
            sim_rej_prob, sim_adm_prob = run_monte_carlo_simulation(
                selected_schools, correlation_matrix
            )
            self._simulation_cache.put(sim_cache_key, (sim_rej_prob, sim_adm_prob))

        metrics["simulated_rejection_probability"] = sim_rej_prob
        metrics["simulated_admission_probability"] = sim_adm_prob

        return {
            "schools": selected_schools,
            "metrics": metrics,
            "objective_values": f_values.tolist(),
        }

    def visualize_recommendations(
        self, recommendations: list[dict[str, Any]], adaptive_thresholds: dict[str, float]
    ) -> None:
        from .visualizer import visualize_recommendations as standalone_visualize_recommendations

        standalone_visualize_recommendations(recommendations, adaptive_thresholds)
