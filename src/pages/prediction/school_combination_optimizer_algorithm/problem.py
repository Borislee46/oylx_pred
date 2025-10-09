from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from pymoo.config import Config
from pymoo.core.problem import Problem

from src.pages.prediction.school_combination_optimizer_algorithm.candidate_filter import (
    filter_candidates_by_background,
)
from src.pages.prediction.school_combination_optimizer_algorithm.common_utils import (
    normalize_school_name,
)
from src.pages.prediction.school_combination_optimizer_algorithm.major_category_config import (
    get_cross_major_limit,
)
from src.pages.prediction.school_combination_optimizer_algorithm.metrics_calculator import (
    calculate_metrics,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer_config import (
    TOP3_SCHOOLS,
    TOP5_SCHOOLS,
    TOP8_SCHOOLS,
)
from src.pages.prediction.school_combination_optimizer_algorithm.problem_initializer import (
    build_major_category_cache,
    build_new_major_cache,
)
from src.utils.app_data_loader import (
    load_bg_target_similarity_cache,
    load_school_major_details_df,
)

Config.warnings["not_compiled"] = False


class SchoolSelectionProblem(Problem):
    def __init__(
        self,
        all_schools_data: list[dict[str, Any]],
        background_major: str,
        max_schools: int = 10,
        adaptive_thresholds: dict[str, float] = None,
        school_level: str = None,
        gpa: float = None,
        min_schools: int = 1,
    ):
        self._norm_top3 = {normalize_school_name(u) for u in TOP3_SCHOOLS}
        self._norm_top5 = {normalize_school_name(u) for u in TOP5_SCHOOLS}
        self._norm_top8 = {normalize_school_name(u) for u in TOP8_SCHOOLS}

        all_schools_data = filter_candidates_by_background(
            all_schools_data, school_level, gpa, min_schools
        )

        self.all_schools_data = all_schools_data
        self.background_major = background_major
        self.max_schools = max_schools
        self.min_schools = max(0, int(min_schools))

        if adaptive_thresholds is None:
            from .optimizer_config import SCHOOL_CATEGORY_THRESHOLDS

            self.adaptive_thresholds = SCHOOL_CATEGORY_THRESHOLDS
        else:
            self.adaptive_thresholds = adaptive_thresholds

        self.school_level = school_level
        self.gpa = gpa

        self.is_high_bg_high_gpa = (
            school_level in {"985", "211", "1-50", "51-100"} and gpa is not None and gpa >= 3.2
        )

        n_var = len(self.all_schools_data)
        super().__init__(n_var=n_var, n_obj=6, n_constr=10, xl=0, xu=1, type_var=np.bool_)

        self.bg_target_similarity_cache_data = load_bg_target_similarity_cache()

        details_df = load_school_major_details_df()
        self.background_major_category, self.major_category_cache = build_major_category_cache(
            background_major, details_df
        )
        self.new_major_cache = build_new_major_cache(all_schools_data)

        self._precompute_constraint_helpers()

        try:
            max_workers = min(8, max(2, int(len(self.all_schools_data) ** 0.5) + 1))
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        except Exception:
            self._executor = ThreadPoolExecutor(max_workers=2)

    def _precompute_constraint_helpers(self):
        self.applies_hk_reach_constraint = False
        self.required_hk_list_for_reach = []
        self.also_requires_one_top3 = False
        self.applies_hk_safety_constraint = False
        self.required_hk_list_for_safety = []

        if self.school_level in {"985", "211", "1-50", "51-100"}:
            self.applies_hk_reach_constraint = True
            self.required_hk_list_for_reach = TOP3_SCHOOLS
        elif self.school_level == "普通本科":
            self.applies_hk_reach_constraint = True
            if self.gpa is not None and self.gpa >= 3.0:
                self.required_hk_list_for_reach = TOP3_SCHOOLS
            else:
                self.required_hk_list_for_reach = TOP5_SCHOOLS
                self.also_requires_one_top3 = True

        if (
            (self.school_level in {"985", "211", "1-50", "51-100"})
            and self.gpa is not None
            and self.gpa >= 2.5
        ):
            self.applies_hk_safety_constraint = True
            self.required_hk_list_for_safety = TOP8_SCHOOLS
        elif self.school_level == "普通本科" and self.gpa is not None and self.gpa > 3.0:
            self.applies_hk_safety_constraint = True
            self.required_hk_list_for_safety = TOP8_SCHOOLS

        reach_threshold = self.adaptive_thresholds.get("target_lower", 0.6)
        self.potential_hk_reach_options_existed = any(
            normalize_school_name(s.get("university"))
            in {normalize_school_name(u) for u in self.required_hk_list_for_reach}
            and s.get("probability", 1.0) < reach_threshold
            for s in self.all_schools_data
        )
        self.potential_top3_options_existed = any(
            normalize_school_name(s.get("university")) in self._norm_top3
            and s.get("probability", 1.0) < reach_threshold
            for s in self.all_schools_data
        )

        safety_threshold = self.adaptive_thresholds.get("safety", 0.8)
        self.potential_hk_safety_options_exist = any(
            normalize_school_name(s.get("university"))
            in {normalize_school_name(u) for u in self.required_hk_list_for_safety}
            and s.get("probability", 0.0) >= safety_threshold
            for s in self.all_schools_data
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n_pop = len(x)
        f1_rejection = np.ones(n_pop)
        f2_diversity = np.zeros(n_pop)
        f3_balance = np.zeros(n_pop)
        f4_similarity = np.zeros(n_pop)
        f5_new_major = np.zeros(n_pop)
        f6_major_category = np.zeros(n_pop)

        g1_max_schools = np.zeros(n_pop)
        g2_min_reach = np.zeros(n_pop)
        g3_min_target = np.zeros(n_pop)
        g4_min_safety = np.zeros(n_pop)
        g5_min_count = np.zeros(n_pop)
        g6_hk_violation = np.zeros(n_pop)
        g7_cross_major_violation = np.zeros(n_pop)
        g8_same_group_min_violation = np.zeros(n_pop)
        g9_min_top3 = np.zeros(n_pop)
        g10_min_top5 = np.zeros(n_pop)

        num_selected_vec = np.sum(x, axis=1)

        cross_major_limit = (
            get_cross_major_limit(self.background_major_category)
            if self.background_major_category
            else 0.5
        )
        cross_major_limit = max(0.0, min(1.0, cross_major_limit - 0.1))

        empty_mask = num_selected_vec == 0
        if np.any(empty_mask):
            g1_max_schools[empty_mask] = -self.max_schools
            g2_min_reach[empty_mask] = 1
            g3_min_target[empty_mask] = 1
            g4_min_safety[empty_mask] = 1
            g5_min_count[empty_mask] = self.min_schools
            f3_balance[empty_mask] = -1000

        non_empty_indices = np.where(~empty_mask)[0]
        if len(non_empty_indices) > 0:
            futures = []
            for i in non_empty_indices:
                selected_indices = np.where(x[i] == 1)[0]
                selected_schools = [self.all_schools_data[j] for j in selected_indices]

                future = self._executor.submit(
                    self._evaluate_single,
                    i,
                    selected_schools,
                    len(selected_indices),
                    cross_major_limit,
                )
                futures.append(future)

            for future in futures:
                try:
                    i, objectives, constraints = future.result()
                    (
                        f1_rejection[i],
                        f2_diversity[i],
                        f3_balance[i],
                        f4_similarity[i],
                        f5_new_major[i],
                        f6_major_category[i],
                    ) = objectives
                    (
                        g1_max_schools[i],
                        g2_min_reach[i],
                        g3_min_target[i],
                        g4_min_safety[i],
                        g5_min_count[i],
                        g6_hk_violation[i],
                        g7_cross_major_violation[i],
                        g8_same_group_min_violation[i],
                        g9_min_top3[i],
                        g10_min_top5[i],
                    ) = constraints
                except Exception:
                    f1_rejection[i] = 1.0
                    f3_balance[i] = -1000
                    g5_min_count[i] = max(1, self.min_schools)

        try:
            from .optimizer_config import MAJOR_SIMILARITY_WEIGHT, OBJECTIVE_WEIGHTS
        except Exception:
            MAJOR_SIMILARITY_WEIGHT = 1.0
            OBJECTIVE_WEIGHTS = {
                "rejection_probability": 1.0,
                "diversity": 2.5,
                "balance_score": 2.0,
                "major_similarity": 1.0,
                "new_major_ratio": 0.5,
                "major_category_score": 1.0,
            }

        w_rej = OBJECTIVE_WEIGHTS.get("rejection_probability", 1.0)
        w_div = OBJECTIVE_WEIGHTS.get("diversity", 2.5)
        w_bal = OBJECTIVE_WEIGHTS.get("balance_score", 2.0)
        w_sim = OBJECTIVE_WEIGHTS.get("major_similarity", 1.0)
        w_new = OBJECTIVE_WEIGHTS.get("new_major_ratio", 0.5)
        w_cat = OBJECTIVE_WEIGHTS.get("major_category_score", 1.0)

        out["F"] = np.column_stack(
            [
                f1_rejection * w_rej,
                f2_diversity * w_div,
                f3_balance * w_bal,
                -(-f4_similarity * MAJOR_SIMILARITY_WEIGHT * w_sim),
                f5_new_major * w_new,
                f6_major_category * w_cat,
            ]
        )
        out["G"] = np.column_stack(
            [
                g1_max_schools,
                g2_min_reach,
                g3_min_target,
                g4_min_safety,
                g5_min_count,
                g6_hk_violation,
                g7_cross_major_violation,
                g8_same_group_min_violation,
                g9_min_top3,
                g10_min_top5,
            ]
        )

    def _evaluate_single(self, index, selected_schools, num_selected, cross_major_limit):
        metrics = calculate_metrics(
            schools=selected_schools,
            background_major=self.background_major,
            adaptive_thresholds=self.adaptive_thresholds,
            bg_target_similarity_cache=self.bg_target_similarity_cache_data,
            new_major_cache=self.new_major_cache,
            background_major_category=self.background_major_category,
            major_category_cache=self.major_category_cache,
        )

        objectives = (
            metrics.get("rejection_probability", 1.0),
            -metrics.get("diversity", 0),
            -metrics.get("balance_score", -1000),
            -metrics.get("major_similarity", 0),
            -metrics.get("new_major_ratio", 0),
            -metrics.get("major_category_score", 0),
        )

        reach_count = metrics.get("reach_count", 0)
        target_count = metrics.get("target_count", 0)
        safety_count = metrics.get("safety_count", 0)

        if self.is_high_bg_high_gpa:
            from .optimizer_config import BALANCE_RATIOS_HIGH_BG

            ratios = BALANCE_RATIOS_HIGH_BG
        else:
            from .optimizer_config import BALANCE_RATIOS

            ratios = BALANCE_RATIOS

        min_reach_required = max(1, round(num_selected * ratios["reach"]))
        min_target_required = max(1, round(num_selected * ratios["target"]))
        min_safety_required = max(1, round(num_selected * ratios["safety"]))

        try:
            from .optimizer_config import CONSTRAINT_FLEXIBILITY, SAME_GROUP_MIN_RATIO
        except Exception:
            SAME_GROUP_MIN_RATIO = 0.7
            CONSTRAINT_FLEXIBILITY = {}

        flexible_same_group_ratio = CONSTRAINT_FLEXIBILITY.get(
            "min_same_group_ratio", SAME_GROUP_MIN_RATIO
        )

        same_group_min_violation = 0
        if self.background_major_category and metrics.get("major_category_diversity", 0) >= 1:
            cross_ratio = float(metrics.get("cross_major_ratio", 0.0))
            same_ratio = max(0.0, 1.0 - cross_ratio)

            if CONSTRAINT_FLEXIBILITY.get("adaptive_balance", True) and num_selected <= 5:
                flexible_same_group_ratio = max(0.4, flexible_same_group_ratio - 0.1)

            if same_ratio + 1e-9 < flexible_same_group_ratio:
                gap = flexible_same_group_ratio - same_ratio
                same_group_min_violation = max(1, int(round(gap * num_selected)))

        min_top3_violation = 0
        min_top5_violation = 0
        if self.school_level in {"985", "211", "1-50", "51-100"} and (
            self.gpa is not None and self.gpa >= 3.2
        ):
            try:
                from .optimizer_config import MIN_TOP3_COUNT_FOR_HIGH_BG, MIN_TOP5_COUNT_FOR_HIGH_BG

                top3_count = sum(
                    1
                    for s in selected_schools
                    if normalize_school_name(s.get("university")) in self._norm_top3
                )
                top5_count = sum(
                    1
                    for s in selected_schools
                    if normalize_school_name(s.get("university")) in self._norm_top5
                )

                if top3_count < MIN_TOP3_COUNT_FOR_HIGH_BG:
                    min_top3_violation = MIN_TOP3_COUNT_FOR_HIGH_BG - top3_count
                if top5_count < MIN_TOP5_COUNT_FOR_HIGH_BG:
                    min_top5_violation = MIN_TOP5_COUNT_FOR_HIGH_BG - top5_count
            except Exception:
                pass

        constraints = (
            num_selected - self.max_schools,
            max(0, min_reach_required - reach_count),
            max(0, min_target_required - target_count),
            max(0, min_safety_required - safety_count),
            max(0, self.min_schools - num_selected),
            self._calculate_hk_violation(selected_schools),
            max(0.0, metrics.get("cross_major_ratio", 0.0) - cross_major_limit),
            same_group_min_violation,
            min_top3_violation,
            min_top5_violation,
        )

        return index, objectives, constraints

    def close(self):
        try:
            if hasattr(self, "_executor") and self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None
        except Exception:
            pass

    def _calculate_hk_violation(self, selected_schools: list[dict[str, Any]]) -> int:
        try:
            from .optimizer_config import CONSTRAINT_FLEXIBILITY
        except Exception:
            CONSTRAINT_FLEXIBILITY = {"enable_strict_hk_constraint": True}

        if not CONSTRAINT_FLEXIBILITY.get("enable_strict_hk_constraint", True):
            return 0

        num_violations = 0
        reach_threshold = self.adaptive_thresholds.get("target_lower", 0.6)
        safety_threshold = self.adaptive_thresholds.get("safety", 0.8)

        if self.applies_hk_reach_constraint:
            reach_schools = [
                s for s in selected_schools if s.get("probability", 1.0) < reach_threshold
            ]
            if reach_schools:
                violating_reach = [
                    s
                    for s in reach_schools
                    if normalize_school_name(s.get("university"))
                    not in {normalize_school_name(u) for u in self.required_hk_list_for_reach}
                ]
                if violating_reach and self.potential_hk_reach_options_existed:
                    num_violations += len(violating_reach)

                if self.also_requires_one_top3:
                    has_top3_reach = any(
                        normalize_school_name(s.get("university")) in self._norm_top3
                        for s in reach_schools
                    )
                    if not has_top3_reach and self.potential_top3_options_existed:
                        num_violations += 1

        if self.applies_hk_safety_constraint:
            safety_schools = [
                s for s in selected_schools if s.get("probability", 0.0) >= safety_threshold
            ]
            if safety_schools:
                violating_safety = [
                    s
                    for s in safety_schools
                    if normalize_school_name(s.get("university"))
                    not in {normalize_school_name(u) for u in self.required_hk_list_for_safety}
                ]
                if violating_safety and self.potential_hk_safety_options_exist:
                    num_violations += len(violating_safety)

        return num_violations
