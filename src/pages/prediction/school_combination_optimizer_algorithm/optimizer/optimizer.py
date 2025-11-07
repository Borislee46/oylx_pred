from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
from pymoo.config import Config

from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    PlanConfig,
    get_plan_configs,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.cache_manager import (
    build_optimization_input_hash,
    clear_all_caches,
    get_cached_data,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.context import (
    OptimizationContext,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.filters_handler import (
    apply_all_filters,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.metrics_calculator_wrapper import (
    calculate_metrics_for_selection,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.optimization_runner import (
    create_problem,
    run_optimization,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.recommendation_builder import (
    build_final_recommendation,
    get_fallback_recommendation,
    get_fallback_recommendation_with_filtered_schools,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.school_adjuster import (
    adjust_probability_by_university_difficulty,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.solution_selector import (
    find_best_solution_indices,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.threshold_calculator import (
    calculate_adaptive_thresholds_for_context,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    LRUCache,
    build_major_category_cache,
)
from src.pages.prediction.school_combination_optimizer_algorithm.visualizer import (
    visualize_recommendations as standalone_visualize_recommendations,
)
from src.utils.app_data_loader import load_school_major_details_df
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

    def _get_cached_data_wrapper(
        self, cache_type: str, key: str, calculation_func: Callable[[], Any]
    ) -> Any:
        return get_cached_data(self._caches, cache_type, key, calculation_func)

    def _calculate_adaptive_thresholds(self, context: OptimizationContext) -> dict[str, float]:
        return calculate_adaptive_thresholds_for_context(context, self._safe_execute)

    def _apply_all_filters(
        self, schools_data: list[dict[str, Any]], context: OptimizationContext
    ) -> list[dict[str, Any]]:
        return apply_all_filters(schools_data, context, self.bg_target_similarity_cache)

    def _create_problem(
        self,
        schools_data: list[dict[str, Any]],
        plan_config: PlanConfig,
        context: OptimizationContext,
    ) -> Any:
        return create_problem(schools_data, plan_config, context)

    def _run_optimization(self, problem: Any) -> Optional[Any]:
        return run_optimization(
            problem, self.population_size, self.n_generations, self._safe_execute
        )

    def _calculate_metrics(
        self,
        selected_schools: list[dict[str, Any]],
        context: OptimizationContext,
        problem: Any,
    ) -> dict[str, Any]:
        return calculate_metrics_for_selection(
            selected_schools,
            context,
            problem,
            self.correlation_matrix,
            self._get_cached_data_wrapper,
        )

    def _has_sufficient_schools(self, schools_data: list[dict[str, Any]], min_schools: int) -> bool:
        return len(schools_data) >= min_schools if schools_data else False

    def _find_best_solution_indices(
        self,
        res: Any,
        problem: Any,
        min_schools: int,
        limit: int = 1,
    ) -> list[int]:
        return find_best_solution_indices(res, problem, self.context, min_schools, limit)

    def _build_final_recommendation(
        self,
        x: np.ndarray,
        f_values: np.ndarray,
        problem: Any,
        context: OptimizationContext,
        plan_config: PlanConfig,
    ) -> Optional[dict[str, Any]]:
        return build_final_recommendation(
            x,
            f_values,
            problem,
            context,
            plan_config,
            self.correlation_matrix,
            self._get_cached_data_wrapper,
        )

    def _adjust_probability_by_university_difficulty(
        self,
        schools: list[dict[str, Any]],
        adaptive_thresholds: Optional[dict[str, float]] = None,
    ) -> list[dict[str, Any]]:
        return adjust_probability_by_university_difficulty(schools, adaptive_thresholds)

    def _get_fallback_recommendation(
        self, context: OptimizationContext, plan_config: PlanConfig
    ) -> Optional[dict[str, Any]]:
        return get_fallback_recommendation(
            context, plan_config, self.correlation_matrix, self._get_cached_data_wrapper
        )

    def _get_fallback_recommendation_with_filtered_schools(
        self,
        filtered_schools: list[dict[str, Any]],
        context: OptimizationContext,
        plan_config: PlanConfig,
    ) -> Optional[dict[str, Any]]:
        return get_fallback_recommendation_with_filtered_schools(
            filtered_schools,
            context,
            plan_config,
            self.correlation_matrix,
            self._get_cached_data_wrapper,
        )

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

        fallback = self._get_fallback_recommendation(context, plan_config)
        if fallback:
            logger.info(f"获得fallback推荐结果，学校数量: {len(fallback.get('schools', []))}")
        else:
            logger.warning("fallback推荐结果也为空")
        return fallback

    def clear_cache(self) -> None:
        clear_all_caches(self._caches)

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

        input_hash_key = build_optimization_input_hash(
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

    def visualize_recommendations(
        self,
        recommendations: list[dict[str, Any]],
        adaptive_thresholds: dict[str, float],
    ) -> None:
        standalone_visualize_recommendations(recommendations, adaptive_thresholds)
