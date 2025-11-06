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
from src.pages.prediction.result_modifier.config import UNIVERSITY_DIFFICULTY_ORDER
from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG,
    BALANCE_RATIOS,
    DEFAULT_REFERENCE_DIRECTIONS_COUNT,
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
    clip_probability,
    get_cached_reference_directions,
    normalize_major_name,
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
    problem: Optional[SchoolSelectionProblem] = None
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
        self.context: Optional[OptimizationContext] = None
        self.bg_target_similarity_cache: dict[str, Any] = {}

    def _safe_execute(
        self,
        operation: Callable[[], Any],
        fallback_operation: Optional[Callable[[], Any]] = None,
        error_message: str = "Operation failed",
    ) -> Any:
        try:
            return operation()
        except Exception as e:
            logger.warning(f"{error_message}: {e}")
            return fallback_operation() if fallback_operation else None

    def _calculate_adaptive_thresholds(self, context: OptimizationContext) -> dict[str, float]:
        def get_probabilities():
            return [
                clip_probability(school.get("probability", 0.0))
                for school in context.all_schools_data
            ]

        probabilities = self._safe_execute(
            get_probabilities,
            lambda: [
                clip_probability(school.get("probability", 0.0))
                for school in context.all_schools_data
            ],
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
        self, schools_data: list[dict[str, Any]], context: OptimizationContext
    ) -> list[dict[str, Any]]:
        logger.info(
            f"_apply_all_filters开始: 输入学校数量={len(schools_data)}, "
            f"background_major={context.background_major}"
        )

        schools_data = [
            ({**s, "major_norm": normalize_major_name(s.get("major", ""))} if s else s)
            for s in schools_data
        ]
        logger.info(f"添加major_norm后，学校数量={len(schools_data)}")

        def similarity_filter(schools):
            result = deduplicate_universities_by_similarity(
                schools, context.background_major, None, MACAU_UNIVERSITIES
            )
            logger.info(f"similarity_filter: {len(schools)} -> {len(result)}")
            return result

        def similarity_filter_only(schools):
            similarities = []
            filtered = []
            cache = self.bg_target_similarity_cache
            logger.info(f"similarity_filter_only使用缓存大小: {len(cache)}")

            sample_majors = []
            for s in schools:
                major = s.get("major", "")
                similarity = get_cached_major_similarity(major, context.background_major, cache)
                similarities.append(similarity)

                if len(sample_majors) < 5:
                    sample_majors.append((major, similarity))

                if similarity >= GLOBAL_MIN_SIMILARITY:
                    filtered.append(s)

            if similarities:
                logger.info(
                    f"similarity_filter_only: {len(schools)} -> {len(filtered)}, "
                    f"相似度范围=[{min(similarities):.3f}, {max(similarities):.3f}], "
                    f"平均值={sum(similarities) / len(similarities):.3f}, "
                    f"阈值={GLOBAL_MIN_SIMILARITY}"
                )
                if sample_majors:
                    sample_info = ", ".join([f"{m[:30]}:{sim:.3f}" for m, sim in sample_majors])
                    logger.info(f"前5个专业相似度示例: {sample_info}")
                if len(filtered) == 0 and len(schools) > 0:
                    logger.warning(
                        f"所有学校都被过滤掉！最高相似度={max(similarities):.3f} < 阈值{GLOBAL_MIN_SIMILARITY}, "
                        f"background_major={context.background_major}"
                    )
            return filtered

        def probability_calibration(schools):
            result = calibrate_cross_major_probabilities(schools, context.background_faculty)
            logger.info(f"probability_calibration: {len(schools)} -> {len(result)}")
            return result

        filters = [
            ("deduplicate_majors", deduplicate_majors, False),
            ("similarity_filter", similarity_filter, True),
            ("similarity_filter_only", similarity_filter_only, True),
            ("probability_calibration", probability_calibration, True),
        ]

        filtered_data = schools_data
        for filter_name, filter_func, continue_on_empty in filters:
            original_len = len(filtered_data)
            logger.info(f"应用过滤器 {filter_name}: 输入数量={original_len}")

            try:
                filtered_data = filter_func(filtered_data)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"过滤器 {filter_name} 出现错误: {type(e).__name__}: {e}")
                if not continue_on_empty:
                    logger.error(f"过滤器 {filter_name} 出现关键错误，停止后续过滤器")
                    break
                filtered_data = filtered_data
            except Exception as e:
                logger.error(
                    f"过滤器 {filter_name} 出现未知错误: {type(e).__name__}: {e}", exc_info=True
                )
                if not continue_on_empty:
                    logger.error(f"过滤器 {filter_name} 出现未知错误，停止后续过滤器")
                    break
                filtered_data = filtered_data

            new_len = len(filtered_data)
            logger.info(f"过滤器 {filter_name} 完成: {original_len} -> {new_len}")

            if new_len == 0 and not continue_on_empty:
                logger.warning(f"过滤器 {filter_name} 将所有数据过滤为空，停止后续过滤器")
                break

        logger.info(f"_apply_all_filters完成: 最终学校数量={len(filtered_data)}")
        return filtered_data

    def _create_problem(
        self,
        schools_data: list[dict[str, Any]],
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
        n_ref = DEFAULT_REFERENCE_DIRECTIONS_COUNT
        base_pop = self.population_size
        base_gen = self.n_generations

        if problem_size < 30:
            pop = max(base_pop, problem_size * 2)
            n_gen = base_gen
        elif problem_size < 50:
            pop = max(base_pop, int(base_pop * 1.2))
            n_gen = int(base_gen * 1.2)
        else:
            pop = base_pop
            n_gen = base_gen

        ref = get_cached_reference_directions("energy", n_dim=5, n_points=n_ref)
        logger.info(
            f"优化参数: problem_size={problem_size}, population_size={pop}, n_generations={n_gen}"
        )
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

    def _get_cached_data(
        self, cache_type: str, key: str, calculation_func: Callable[[], Any]
    ) -> Any:
        cache = self._caches[cache_type]
        if cached := cache.get(key):
            return cached.copy() if hasattr(cached, "copy") else cached

        result = calculation_func()
        cache.put(key, result.copy() if hasattr(result, "copy") else result)
        return result

    def _calculate_metrics(
        self,
        selected_schools: list[dict[str, Any]],
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
        schools: list[dict[str, Any]],
        metrics: dict[str, Any],
        objective_values: Optional[list[float]] = None,
        rec_type: Optional[str] = None,
    ) -> dict[str, Any]:
        recommendation: dict[str, Any] = {
            "schools": schools,
            "metrics": metrics,
            "objective_values": objective_values or self._default_objective_values(metrics),
        }
        if rec_type:
            recommendation["type"] = rec_type
        return recommendation

    def _default_objective_values(self, metrics: dict[str, Any]) -> list[float]:
        return [
            metrics.get("rejection_probability", 1.0),
            -metrics.get("diversity", 0),
            -metrics.get("balance_score", -1000),
            -metrics.get("major_similarity", 0),
            -metrics.get("new_major_ratio", 0),
        ]

    def _has_sufficient_schools(self, schools_data: list[dict[str, Any]], min_schools: int) -> bool:
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

        X, F = res.X, res.F
        n_solutions, n_candidates = X.shape[0], X.shape[1]

        selected_counts = np.sum(X, axis=1)

        balance_scores = np.full(n_solutions, -np.inf, dtype=float)

        if self.context and self.context.adaptive_thresholds:
            probs_vec = np.array(
                [
                    clip_probability(problem.all_schools_data[j].get("probability", 0.0))
                    for j in range(n_candidates)
                ],
                dtype=float,
            )
            safety_thresh = self.context.adaptive_thresholds.get("safety", 0.75)
            target_thresh = self.context.adaptive_thresholds.get("target_lower", 0.55)

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

        feasible_mask = self._get_feasible_mask(res, n_solutions)
        candidate_indices = (
            np.arange(n_solutions)[feasible_mask]
            if feasible_mask is not None
            else np.arange(n_solutions)
        )

        if candidate_indices.size == 0:
            return []

        return self._sort_and_select_candidates(
            candidate_indices, balance_scores.tolist(), F, limit
        )

    def _calculate_balance_score(self, probabilities: list[float], total: int) -> float:
        if not self.context or not self.context.adaptive_thresholds:
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

        selected_schools = self._adjust_probability_by_university_difficulty(
            selected_schools, context.adaptive_thresholds
        )

        metrics = self._calculate_metrics(selected_schools, context, problem)

        return self._create_recommendation(selected_schools, metrics, f_values.tolist())

    def _apply_post_filters(
        self,
        schools: list[dict[str, Any]],
        context: OptimizationContext,
        problem: SchoolSelectionProblem,
    ) -> list[dict[str, Any]]:
        if not context.background_faculty or not context.adaptive_thresholds:
            return schools

        reach_threshold = context.adaptive_thresholds.get("target_lower", 0.0)
        filtered_selection = []

        for school in schools:
            is_reach = clip_probability(school.get("probability", 1.0)) < reach_threshold
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
        self, schools: list[dict[str, Any]], min_schools: int, max_schools: int
    ) -> list[dict[str, Any]]:
        num_selected = len(schools)

        if num_selected < min_schools:
            return []
        elif num_selected > max_schools:
            adaptive_thresholds = self.context.adaptive_thresholds if self.context else None
            return reduce_schools_balanced(schools, max_schools, adaptive_thresholds)

        return schools

    def _adjust_probability_by_university_difficulty(
        self,
        schools: list[dict[str, Any]],
        adaptive_thresholds: Optional[dict[str, float]] = None,
    ) -> list[dict[str, Any]]:
        if not schools:
            return schools

        difficulty_map = {uni: idx for idx, uni in enumerate(UNIVERSITY_DIFFICULTY_ORDER)}
        target_thresh = (
            adaptive_thresholds.get("target_lower", 0.55) if adaptive_thresholds else 0.55
        )
        total_universities = len(UNIVERSITY_DIFFICULTY_ORDER)

        adjusted_schools = []
        for school in schools:
            university = school.get("university", "")
            current_prob = clip_probability(school.get("probability", 0.0))

            difficulty_rank = difficulty_map.get(university, total_universities)
            normalized_rank = (
                difficulty_rank / total_universities if total_universities > 0 else 0.5
            )

            if normalized_rank >= 0.3:
                if current_prob < target_thresh:
                    adjustment_factor = (
                        (normalized_rank - 0.3) / 0.7 if normalized_rank > 0.3 else 0.1
                    )
                    boost_amount = max(0.08, 0.15 * adjustment_factor)
                    adjusted_prob = min(1.0, current_prob + boost_amount)

                    if adjusted_prob < target_thresh:
                        adjusted_prob = target_thresh + 0.02

                    school = {**school, "probability": min(1.0, adjusted_prob)}
                elif current_prob < target_thresh + 0.15:
                    adjustment_factor = (
                        (normalized_rank - 0.3) / 0.7 if normalized_rank > 0.3 else 0.1
                    )
                    boost_amount = 0.06 * adjustment_factor
                    adjusted_prob = min(1.0, current_prob + boost_amount)
                    school = {**school, "probability": adjusted_prob}

            adjusted_schools.append(school)

        return adjusted_schools

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

        balanced_schools = self._adjust_probability_by_university_difficulty(
            balanced_schools, context.adaptive_thresholds
        )
        metrics = self._calculate_metrics(balanced_schools, context, context.problem)
        return self._create_recommendation(balanced_schools, metrics)

    def _get_fallback_recommendation_with_filtered_schools(
        self,
        filtered_schools: list[dict[str, Any]],
        context: OptimizationContext,
        plan_config: PlanConfig,
    ) -> Optional[dict[str, Any]]:
        available_count = len(filtered_schools)
        adjusted_min_schools = min(plan_config.min_schools, available_count)
        adjusted_max_schools = min(plan_config.max_schools, available_count)

        balanced_schools = generate_balanced_selection(
            filtered_schools,
            adjusted_min_schools,
            adjusted_max_schools,
            context.adaptive_thresholds,
        )

        if not balanced_schools:
            return None

        if not context.problem:
            return None

        balanced_schools = self._adjust_probability_by_university_difficulty(
            balanced_schools, context.adaptive_thresholds
        )
        metrics = self._calculate_metrics(balanced_schools, context, context.problem)
        return self._create_recommendation(balanced_schools, metrics)

    def _optimize_single_plan(
        self, plan_config: PlanConfig, context: OptimizationContext
    ) -> Optional[dict[str, Any]]:
        logger.info(
            f"_optimize_single_plan开始: plan_config={plan_config.name}, "
            f"输入学校数量={len(context.all_schools_data)}"
        )

        filtered_schools = self._apply_all_filters(context.all_schools_data, context)
        logger.info(f"过滤后学校数量: {len(filtered_schools)}, 最小要求: {plan_config.min_schools}")

        if not self._has_sufficient_schools(filtered_schools, plan_config.min_schools):
            logger.warning(
                f"学校数量不足，过滤后={len(filtered_schools)}, 最小要求={plan_config.min_schools}，"
                f"将尝试使用fallback机制"
            )
            if len(filtered_schools) == 0:
                return None
            problem = self._create_problem(filtered_schools, plan_config, context)
            context.problem = problem
            logger.info(f"问题创建完成（学校数量不足），变量数量: {problem.n_var}")
            logger.info("直接尝试获取fallback推荐结果")
            fallback = self._get_fallback_recommendation_with_filtered_schools(
                filtered_schools, context, plan_config
            )
            if fallback:
                logger.info(f"获得fallback推荐结果，学校数量: {len(fallback.get('schools', []))}")
            else:
                logger.warning("fallback推荐结果也为空")
            return fallback

        problem = self._create_problem(filtered_schools, plan_config, context)
        context.problem = problem
        logger.info(f"问题创建完成，变量数量: {problem.n_var}")

        result = self._run_optimization(problem)
        logger.info(f"优化运行完成，结果是否存在: {result is not None}")

        if result and hasattr(result, "X"):
            n_solutions = len(result.X) if result.X is not None else 0
            logger.info(f"优化结果有效，解数量: {n_solutions}")

            if n_solutions == 0:
                logger.warning(
                    f"优化器未找到任何解，可能原因: "
                    f"变量数量={problem.n_var}, min_schools={plan_config.min_schools}, "
                    f"max_schools={plan_config.max_schools}, "
                    f"约束条件可能过于严格"
                )
                if hasattr(result, "pop") and result.pop is not None:
                    logger.info(
                        f"种群大小: {len(result.pop) if hasattr(result.pop, '__len__') else 'N/A'}"
                    )
            else:
                feasible_count = 0
                if hasattr(result, "CV") and result.CV is not None:
                    feasible_count = np.sum(result.CV <= 0)
                elif hasattr(result, "G") and result.G is not None:
                    feasible_count = np.sum(np.all(result.G <= 0, axis=1))
                logger.info(f"可行解数量: {feasible_count}/{n_solutions}")

                if feasible_count == 0 and hasattr(result, "G") and result.G is not None:
                    max_violations = np.max(result.G, axis=0)
                    constraint_names = [
                        "max_schools",
                        "min_reach",
                        "min_target",
                        "min_safety",
                        "min_schools",
                        "hk_violation",
                        "min_top3",
                        "min_top5",
                    ]
                    violations_info = ", ".join(
                        [
                            f"{name}={v:.2f}"
                            for name, v in zip(constraint_names, max_violations)
                            if v > 0
                        ]
                    )
                    logger.warning(f"所有解都违反约束，最大违反值: {violations_info}")

            best_indices = self._find_best_solution_indices(
                result, problem, plan_config.min_schools
            )
            logger.info(f"找到最佳解索引数量: {len(best_indices)}")

            for idx in best_indices:
                recommendation = self._build_final_recommendation(
                    result.X[idx], result.F[idx], problem, context, plan_config
                )
                if recommendation:
                    logger.info(
                        f"成功构建推荐结果，学校数量: {len(recommendation.get('schools', []))}"
                    )
                    return recommendation
                else:
                    logger.warning(f"索引{idx}未能构建推荐结果")
        else:
            logger.warning("优化结果无效或缺少X属性")

        logger.info("尝试获取fallback推荐结果")
        fallback = self._get_fallback_recommendation(context, plan_config)
        if fallback:
            logger.info(f"获得fallback推荐结果，学校数量: {len(fallback.get('schools', []))}")
        else:
            logger.warning("fallback推荐结果也为空")
        return fallback

    def clear_cache(self) -> None:
        for cache in self._caches.values():
            cache.clear()

    def optimize(
        self,
        all_schools_data: list[dict[str, Any]],
        background_major: str,
        background_faculty: Optional[str] = None,
        school_level: Optional[str] = None,
        gpa: Optional[float] = None,
        major_category_cache: Optional[dict[str, str]] = None,
        bg_target_similarity_cache: Optional[dict[str, Any]] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        logger.info(
            f"optimizer.optimize开始: all_schools_data数量={len(all_schools_data)}, "
            f"background_major={background_major}, background_faculty={background_faculty}, "
            f"school_level={school_level}, gpa={gpa}"
        )

        if not all_schools_data:
            logger.warning("all_schools_data为空，返回空结果")
            return [], {}

        if major_category_cache is None:
            logger.info("major_category_cache为空，正在加载")
            details_df = load_school_major_details_df()
            major_category_cache = build_major_category_cache(details_df)
            logger.info(f"major_category_cache加载完成，大小: {len(major_category_cache)}")

        self.context = OptimizationContext(
            all_schools_data=all_schools_data,
            background_major=background_major,
            background_faculty=background_faculty,
            school_level=school_level,
            gpa=gpa,
            major_category_cache=major_category_cache,
        )

        self.bg_target_similarity_cache = bg_target_similarity_cache or {}
        logger.info(f"bg_target_similarity_cache大小: {len(self.bg_target_similarity_cache)}")

        input_hash_key = self._build_optimization_input_hash(
            all_schools_data, background_major, background_faculty, school_level, gpa
        )
        cached_result = self._caches["result"].get(input_hash_key)
        if cached_result:
            logger.info(f"找到缓存结果，直接返回，输入hash: {input_hash_key[:50]}...")
            return cached_result["recommendations"], cached_result["adaptive_thresholds"]

        self.context.adaptive_thresholds = self._calculate_adaptive_thresholds(self.context)
        logger.info(f"自适应阈值计算完成: {self.context.adaptive_thresholds}")

        all_schools_data = self._adjust_probability_by_university_difficulty(
            all_schools_data, self.context.adaptive_thresholds
        )
        self.all_schools_data = all_schools_data
        self.context.all_schools_data = all_schools_data
        logger.info(f"已对选校池进行概率纠偏，学校数量: {len(all_schools_data)}")

        plan_configs = list(get_plan_configs(self.plan_configs))
        logger.info(f"开始优化，计划配置数量: {len(plan_configs)}")

        final_recommendations = []
        for plan_config in plan_configs:
            logger.info(f"开始优化计划配置: {plan_config.name}")
            recommendation = self._optimize_single_plan(plan_config, self.context)
            if recommendation:
                recommendation["type"] = plan_config.name
                final_recommendations.append(recommendation)
                logger.info(f"计划配置 {plan_config.name} 优化完成，获得推荐结果")
            else:
                logger.warning(f"计划配置 {plan_config.name} 未获得推荐结果")

        for recommendation in final_recommendations:
            if recommendation and "schools" in recommendation:
                recommendation["schools"] = self._adjust_probability_by_university_difficulty(
                    recommendation["schools"], self.context.adaptive_thresholds
                )
                if self.context.problem:
                    recommendation["metrics"] = self._calculate_metrics(
                        recommendation["schools"], self.context, self.context.problem
                    )

        logger.info(f"优化完成，最终推荐数量: {len(final_recommendations)}")

        result_to_cache = {
            "recommendations": final_recommendations,
            "adaptive_thresholds": self.context.adaptive_thresholds,
        }
        self._caches["result"].put(input_hash_key, result_to_cache)
        logger.info(f"结果已缓存，输入hash: {input_hash_key[:50]}...")

        return final_recommendations, self.context.adaptive_thresholds

    def _build_optimization_input_hash(
        self,
        all_schools_data: list[dict[str, Any]],
        background_major: str,
        background_faculty: Optional[str],
        school_level: Optional[str],
        gpa: Optional[float],
    ) -> str:
        import hashlib
        import json

        input_data = {
            "background_major": background_major,
            "background_faculty": background_faculty,
            "school_level": school_level,
            "gpa": gpa,
            "schools": sorted(
                [
                    {
                        "university": s.get("university", ""),
                        "major": s.get("major", ""),
                        "probability": s.get("probability", 0.0),
                    }
                    for s in all_schools_data
                ],
                key=lambda x: (x["university"], x["major"]),
            ),
        }

        input_str = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(input_str.encode("utf-8")).hexdigest()

    def visualize_recommendations(
        self,
        recommendations: list[dict[str, Any]],
        adaptive_thresholds: dict[str, float],
    ) -> None:
        standalone_visualize_recommendations(recommendations, adaptive_thresholds)
