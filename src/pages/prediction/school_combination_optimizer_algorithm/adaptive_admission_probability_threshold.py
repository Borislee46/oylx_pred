import numpy as np

from src.pages.prediction.school_combination_optimizer_algorithm.optimizer_config import (
    ADAPTIVE_THRESHOLD_PERCENTILES,
    SCHOOL_CATEGORY_THRESHOLDS,
)


def calculate_adaptive_thresholds(
    all_school_probabilities: list[float],
    reach_percentile_val: int = None,
    safety_percentile_val: int = None,
) -> dict[str, float]:
    if reach_percentile_val is None:
        reach_percentile_val = ADAPTIVE_THRESHOLD_PERCENTILES["reach_percentile_val"]
    if safety_percentile_val is None:
        safety_percentile_val = ADAPTIVE_THRESHOLD_PERCENTILES["safety_percentile_val"]
    if not all_school_probabilities or len(all_school_probabilities) < 3:
        return {
            "safety": SCHOOL_CATEGORY_THRESHOLDS["safety"],
            "target_lower": SCHOOL_CATEGORY_THRESHOLDS["target_lower"],
        }

    target_lower_threshold = float(np.percentile(all_school_probabilities, reach_percentile_val))
    safety_threshold = float(np.percentile(all_school_probabilities, safety_percentile_val))

    if target_lower_threshold > safety_threshold:
        if reach_percentile_val > safety_percentile_val:
            target_lower_threshold, safety_threshold = safety_threshold, target_lower_threshold
        else:
            return {
                "safety": SCHOOL_CATEGORY_THRESHOLDS["safety"],
                "target_lower": SCHOOL_CATEGORY_THRESHOLDS["target_lower"],
            }

    if safety_threshold == target_lower_threshold:
        if safety_threshold < 1.0:
            safety_threshold = min(safety_threshold + 0.005, 1.0)
        if target_lower_threshold > 0.0 and safety_threshold == target_lower_threshold:
            target_lower_threshold = max(target_lower_threshold - 0.005, 0.0)

        if target_lower_threshold >= safety_threshold:
            return {
                "safety": SCHOOL_CATEGORY_THRESHOLDS["safety"],
                "target_lower": SCHOOL_CATEGORY_THRESHOLDS["target_lower"],
            }

    return {"safety": safety_threshold, "target_lower": target_lower_threshold}
