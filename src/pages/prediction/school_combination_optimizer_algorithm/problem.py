from typing import Any

import numpy as np
from pymoo.config import Config
from pymoo.core.problem import Problem

from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    BALANCE_RATIOS,
    BALANCE_RATIOS_HIGH_BG,
    CONSTRAINT_FLEXIBILITY,
    MIN_TOP3_COUNT_FOR_HIGH_BG,
    MIN_TOP5_COUNT_FOR_HIGH_BG,
    OBJECTIVE_WEIGHTS,
    SCHOOL_CATEGORY_THRESHOLDS,
    TOP3_SCHOOLS,
    TOP5_SCHOOLS,
    TOP8_SCHOOLS,
)
from src.pages.prediction.school_combination_optimizer_algorithm.filters import (
    filter_candidates_by_background,
)
from src.pages.prediction.school_combination_optimizer_algorithm.metrics_calculator import (
    calculate_metrics,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    build_major_category_cache,
    build_new_major_cache,
    normalize_school_name,
)
from src.utils.app_data_loader import (
    load_bg_target_similarity_cache,
    load_school_major_details_df,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

Config.warnings["not_compiled"] = False


class SchoolSelectionProblem(Problem):
    def __init__(
        self,
        all_schools_data: list[dict[str, Any]],
        background_major: str,
        background_faculty: str | None,
        max_schools: int = 10,
        adaptive_thresholds: dict[str, float] = None,
        school_level: str = None,
        gpa: float = None,
        min_schools: int = 1,
    ):
        self._norm_top3 = {normalize_school_name(u) for u in TOP3_SCHOOLS}
        self._norm_top5 = {normalize_school_name(u) for u in TOP5_SCHOOLS}
        self._norm_top8 = {normalize_school_name(u) for u in TOP8_SCHOOLS}

        input_count = len(all_schools_data)
        all_schools_data = filter_candidates_by_background(
            all_schools_data,
            school_level,
            gpa,
            min_schools,
            background_faculty,
            adaptive_thresholds,
        )
        output_count = len(all_schools_data)
        if input_count != output_count:
            logger.info(
                f"filter_candidates_by_background过滤: {input_count} -> {output_count}, "
                f"school_level={school_level}, gpa={gpa}, min_schools={min_schools}"
            )

        self.all_schools_data = all_schools_data
        self.background_major = background_major
        self.background_faculty = background_faculty
        self.max_schools = max_schools
        self.min_schools = max(0, int(min_schools))
        self.adaptive_thresholds = adaptive_thresholds or SCHOOL_CATEGORY_THRESHOLDS
        self.school_level = school_level
        self.gpa = gpa

        self.is_high_bg_high_gpa = (
            school_level in {"985", "211", "1-50", "51-100"} and gpa is not None and gpa >= 3.2
        )

        n_var = len(self.all_schools_data)
        super().__init__(n_var=n_var, n_obj=5, n_constr=8, xl=0, xu=1, type_var=np.bool_)

        self.bg_target_similarity_cache_data = load_bg_target_similarity_cache()
        details_df = load_school_major_details_df()
        self.major_category_cache = build_major_category_cache(details_df)
        self.new_major_cache = build_new_major_cache(all_schools_data)

        self._precompute_constraint_helpers()

    def _precompute_constraint_helpers(self):
        self._setup_hk_constraint_flags()
        self._precompute_potential_options_exist()

    def _setup_hk_constraint_flags(self):
        is_elite = self.school_level in {"985", "211", "1-50", "51-100"}
        is_ordinary = self.school_level in {"普通本科", "101-200", "201-300", "301-500", "500之后"}

        self.applies_hk_reach_constraint = is_elite or is_ordinary
        self.required_hk_list_for_reach = []
        self.also_requires_one_top3 = False

        if self.applies_hk_reach_constraint:
            if is_elite or (is_ordinary and self.gpa and self.gpa >= 3.0):
                self.required_hk_list_for_reach = TOP3_SCHOOLS
            else:
                self.required_hk_list_for_reach = TOP5_SCHOOLS
                self.also_requires_one_top3 = True

        self.applies_hk_safety_constraint = (is_elite and self.gpa and self.gpa >= 2.5) or (
            is_ordinary and self.gpa and self.gpa > 3.0
        )
        self.required_hk_list_for_safety = TOP8_SCHOOLS if self.applies_hk_safety_constraint else []

    def _precompute_potential_options_exist(self):
        reach_threshold = self.adaptive_thresholds.get("target_lower", 0.6)
        safety_threshold = self.adaptive_thresholds.get("safety", 0.8)

        norm_required_reach = {normalize_school_name(u) for u in self.required_hk_list_for_reach}
        norm_required_safety = {normalize_school_name(u) for u in self.required_hk_list_for_safety}

        self.potential_hk_reach_options_existed = any(
            normalize_school_name(s.get("university")) in norm_required_reach
            and s.get("probability", 1.0) < reach_threshold
            for s in self.all_schools_data
        )

        self.potential_top3_options_existed = any(
            normalize_school_name(s.get("university")) in self._norm_top3
            and s.get("probability", 1.0) < reach_threshold
            for s in self.all_schools_data
        )

        self.potential_hk_safety_options_exist = any(
            normalize_school_name(s.get("university")) in norm_required_safety
            and s.get("probability", 0.0) >= safety_threshold
            for s in self.all_schools_data
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n_pop = len(x)

        objectives = np.zeros((n_pop, 5))
        constraints = np.zeros((n_pop, 8))
        num_selected_vec = np.sum(x, axis=1)

        empty_mask = num_selected_vec == 0
        if np.any(empty_mask):
            self._handle_empty_solutions(empty_mask, constraints, objectives)

        non_empty_indices = np.where(~empty_mask)[0]
        if non_empty_indices.size > 0:
            self._evaluate_non_empty_solutions(x, non_empty_indices, objectives, constraints)

        out["F"] = self._apply_objective_weights(objectives)
        out["G"] = constraints

    def _handle_empty_solutions(self, empty_mask, constraints, objectives):
        constraints[empty_mask, 0] = -self.max_schools
        constraints[empty_mask, 1] = 1
        constraints[empty_mask, 2] = 1
        constraints[empty_mask, 3] = 1
        constraints[empty_mask, 4] = self.min_schools
        objectives[empty_mask, 2] = -1000

    def _evaluate_non_empty_solutions(self, x, indices, objectives, constraints):
        for i in indices:
            selected_indices = np.where(x[i] == 1)[0]
            selected_schools = [self.all_schools_data[j] for j in selected_indices]

            i, obj_vals, constr_vals = self._evaluate_single(
                i, selected_schools, len(selected_indices)
            )
            objectives[i] = obj_vals
            constraints[i] = constr_vals

    def _apply_objective_weights(self, objectives):
        weights = np.array(
            [
                OBJECTIVE_WEIGHTS.get("rejection_probability"),
                OBJECTIVE_WEIGHTS.get("diversity"),
                OBJECTIVE_WEIGHTS.get("balance_score"),
                OBJECTIVE_WEIGHTS.get("major_similarity"),
                OBJECTIVE_WEIGHTS.get("new_major_ratio"),
            ]
        )
        return objectives * weights

    def _evaluate_single(self, index, selected_schools, num_selected):
        metrics = calculate_metrics(
            schools=selected_schools,
            background_major=self.background_major,
            adaptive_thresholds=self.adaptive_thresholds,
            bg_target_similarity_cache=self.bg_target_similarity_cache_data,
            new_major_cache=self.new_major_cache,
            background_faculty=self.background_faculty,
            major_category_cache=self.major_category_cache,
        )

        objectives = self._calculate_objectives(metrics)

        constraints = self._calculate_constraints(num_selected, metrics, selected_schools)

        return index, objectives, constraints

    def _calculate_objectives(self, metrics):
        return (
            metrics.get("rejection_probability", 1.0),
            -metrics.get("diversity", 0),
            -metrics.get("balance_score", -1000),
            -metrics.get("major_similarity", 0),
            metrics.get("new_major_ratio", 0),
        )

    def _calculate_constraints(self, num_selected, metrics, selected_schools):
        ratios = BALANCE_RATIOS_HIGH_BG if self.is_high_bg_high_gpa else BALANCE_RATIOS

        reach_count = metrics.get("reach_count", 0)
        target_count = metrics.get("target_count", 0)
        safety_count = metrics.get("safety_count", 0)

        min_reach_required = max(1, round(num_selected * ratios["reach"]))
        min_target_required = max(1, round(num_selected * ratios["target"]))
        min_safety_required = max(1, round(num_selected * ratios["safety"]))

        min_top3_violation, min_top5_violation = self._calculate_top_school_violations(
            selected_schools
        )

        return (
            num_selected - self.max_schools,
            max(0, min_reach_required - reach_count),
            max(0, min_target_required - target_count),
            max(0, min_safety_required - safety_count),
            max(0, self.min_schools - num_selected),
            self._calculate_hk_violation(selected_schools),
            min_top3_violation,
            min_top5_violation,
        )

    def _calculate_top_school_violations(self, selected_schools):
        if not self.is_high_bg_high_gpa:
            return 0, 0

        try:
            schools_set = {normalize_school_name(s.get("university")) for s in selected_schools}
            top3_count = len(schools_set & self._norm_top3)
            top5_count = len(schools_set & self._norm_top5)

            return (
                max(0, MIN_TOP3_COUNT_FOR_HIGH_BG - top3_count),
                max(0, MIN_TOP5_COUNT_FOR_HIGH_BG - top5_count),
            )
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"计算top学校违规数时出现错误: {type(e).__name__}: {e}")
            return 0, 0
        except Exception as e:
            logger.error(f"计算top学校违规数时出现未知错误: {type(e).__name__}: {e}", exc_info=True)
            return 0, 0

    def _calculate_hk_violation(self, selected_schools):
        if not CONSTRAINT_FLEXIBILITY.get("enable_strict_hk_constraint", True):
            return 0

        violations = 0
        reach_threshold = self.adaptive_thresholds.get("target_lower", 0.6)
        safety_threshold = self.adaptive_thresholds.get("safety", 0.8)

        if self.applies_hk_reach_constraint:
            violations += self._check_hk_reach_violation(selected_schools, reach_threshold)

        if self.applies_hk_safety_constraint:
            violations += self._check_hk_safety_violation(selected_schools, safety_threshold)

        return violations

    def _check_hk_reach_violation(self, selected_schools, reach_threshold):
        reach_schools = [s for s in selected_schools if s.get("probability", 1.0) < reach_threshold]
        if not reach_schools or not self.potential_hk_reach_options_existed:
            return 0

        norm_required_reach = {normalize_school_name(u) for u in self.required_hk_list_for_reach}
        violating_reach = [
            s
            for s in reach_schools
            if normalize_school_name(s.get("university")) not in norm_required_reach
        ]

        violations = len(violating_reach) if violating_reach else 0

        if self.also_requires_one_top3 and self.potential_top3_options_existed:
            has_top3_reach = any(
                normalize_school_name(s.get("university")) in self._norm_top3 for s in reach_schools
            )
            if not has_top3_reach:
                violations += 1

        return violations

    def _check_hk_safety_violation(self, selected_schools, safety_threshold):
        safety_schools = [
            s for s in selected_schools if s.get("probability", 0.0) >= safety_threshold
        ]
        if not safety_schools or not self.potential_hk_safety_options_exist:
            return 0

        norm_required_safety = {normalize_school_name(u) for u in self.required_hk_list_for_safety}
        violating_safety = [
            s
            for s in safety_schools
            if normalize_school_name(s.get("university")) not in norm_required_safety
        ]

        return len(violating_safety) if violating_safety else 0
