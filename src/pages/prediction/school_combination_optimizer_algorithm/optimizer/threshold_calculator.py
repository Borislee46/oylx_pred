# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
from collections.abc import Callable
from typing import Any

from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.context import (
    OptimizationContext,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    calculate_adaptive_thresholds,
    clip_probability,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def calculate_adaptive_thresholds_for_context(
    context: OptimizationContext,
    safe_execute: Callable[[Callable[[], Any], Callable[[], Any] | None, str], Any],
) -> dict[str, float]:
    def get_probabilities():
        return [
            clip_probability(school.get("probability", 0.0)) for school in context.all_schools_data
        ]

    probabilities = safe_execute(
        get_probabilities,
        lambda: [
            clip_probability(school.get("probability", 0.0)) for school in context.all_schools_data
        ],
        "计算自适应阈值的概率时出错",
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
            safety_percentile_val=ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG["safety_percentile_val"],
        )
    return calculate_adaptive_thresholds(probabilities)
