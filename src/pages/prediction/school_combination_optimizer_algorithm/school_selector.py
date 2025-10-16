import math
from typing import Any

from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    BALANCE_RATIOS,
    SCHOOL_CATEGORY_THRESHOLDS,
)


def _categorize_schools(
    schools: list[dict[str, Any]], safety_thresh: float, target_thresh: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    safety, target, reach = [], [], []

    for s in schools:
        prob = s["probability"]
        if prob >= safety_thresh:
            safety.append(s)
        elif prob >= target_thresh:
            target.append(s)
        else:
            reach.append(s)

    return (
        sorted(safety, key=lambda x: x["probability"], reverse=True),
        sorted(target, key=lambda x: x["probability"], reverse=True),
        sorted(reach, key=lambda x: x["probability"], reverse=True),
    )


def _calculate_removal_distribution(
    total_to_remove: int,
    over_safety: int,
    over_target: int,
    over_reach: int,
    current_counts: tuple[int, int, int],
) -> tuple[int, int, int]:
    total_over = over_safety + over_target + over_reach

    if total_over > 0:
        safety_to_remove = int(total_to_remove * over_safety / total_over)
        target_to_remove = int(total_to_remove * over_target / total_over)
        reach_to_remove = total_to_remove - safety_to_remove - target_to_remove
    else:
        current_safety, current_target, current_reach = current_counts
        total_schools = sum(current_counts)
        safety_to_remove = (
            int(total_to_remove * current_safety / total_schools) if total_schools else 0
        )
        target_to_remove = (
            int(total_to_remove * current_target / total_schools) if total_schools else 0
        )
        reach_to_remove = total_to_remove - safety_to_remove - target_to_remove

    return safety_to_remove, target_to_remove, reach_to_remove


def _adjust_result_to_count(
    result: list[dict[str, Any]], all_schools: list[dict[str, Any]], target_count: int
) -> list[dict[str, Any]]:
    if len(result) > target_count:
        return sorted(result, key=lambda x: x["probability"])[:target_count]
    elif len(result) < target_count:
        remaining = [s for s in all_schools if s not in result]
        remaining.sort(key=lambda x: x["probability"], reverse=True)
        result.extend(remaining[: target_count - len(result)])

    return result


def reduce_schools_balanced(
    schools: list[dict[str, Any]],
    max_count: int,
    adaptive_thresholds: dict[str, float] = None,
) -> list[dict[str, Any]]:
    if len(schools) <= max_count:
        return schools

    current_thresholds = adaptive_thresholds or SCHOOL_CATEGORY_THRESHOLDS
    safety_schools, target_schools, reach_schools = _categorize_schools(
        schools, current_thresholds["safety"], current_thresholds["target_lower"]
    )

    current_safety, current_target, current_reach = (
        len(safety_schools),
        len(target_schools),
        len(reach_schools),
    )

    ideal_safety = max_count * BALANCE_RATIOS["safety"]
    ideal_target = max_count * BALANCE_RATIOS["target"]
    ideal_reach = max_count * BALANCE_RATIOS["reach"]

    over_safety = int(max(0, current_safety - ideal_safety))
    over_target = int(max(0, current_target - ideal_target))
    over_reach = int(max(0, current_reach - ideal_reach))

    total_to_remove = len(schools) - max_count
    safety_to_remove, target_to_remove, reach_to_remove = _calculate_removal_distribution(
        total_to_remove,
        over_safety,
        over_target,
        over_reach,
        (current_safety, current_target, current_reach),
    )

    safety_schools = safety_schools[: max(0, current_safety - safety_to_remove)]
    target_schools = target_schools[: max(0, current_target - target_to_remove)]
    reach_schools = reach_schools[: max(0, current_reach - reach_to_remove)]

    result = safety_schools + target_schools + reach_schools

    return _adjust_result_to_count(result, schools, max_count)


def _get_minimum_selections(
    safety_schools: list[dict[str, Any]],
    target_schools: list[dict[str, Any]],
    reach_schools: list[dict[str, Any]],
) -> tuple[int, int, int]:
    return (
        min(1, len(reach_schools)),
        min(1, len(target_schools)),
        min(1, len(safety_schools)),
    )


def _ensure_reach_school_present(
    result: list[dict[str, Any]],
    reach_schools: list[dict[str, Any]],
    target_threshold: float,
    safety_threshold: float,
) -> list[dict[str, Any]]:
    if not any(s["probability"] < target_threshold for s in result) and reach_schools:
        safety_in_result = [s for s in result if s["probability"] >= safety_threshold]
        if len(safety_in_result) > 1:
            safety_to_remove = min(safety_in_result, key=lambda x: x["probability"])
            result.remove(safety_to_remove)
            result.append(reach_schools[0])
    return result


def generate_balanced_selection(
    schools: list[dict[str, Any]],
    min_count: int,
    max_count: int,
    adaptive_thresholds: dict[str, float] = None,
) -> list[dict[str, Any]]:
    if not schools:
        return []

    count = min(max(min_count, max_count // 2), len(schools))
    current_thresholds = adaptive_thresholds or SCHOOL_CATEGORY_THRESHOLDS

    safety_schools, target_schools, reach_schools = _categorize_schools(
        schools, current_thresholds["safety"], current_thresholds["target_lower"]
    )

    min_reach, min_target, min_safety = _get_minimum_selections(
        safety_schools, target_schools, reach_schools
    )
    min_selections_total = min_reach + min_target + min_safety

    if count < min_selections_total:
        return (
            reach_schools[:min_reach] + target_schools[:min_target] + safety_schools[:min_safety]
        )[:count]

    remaining = count - min_selections_total
    ideal_safety = round(remaining * BALANCE_RATIOS["safety"]) + min_safety
    ideal_target = round(remaining * BALANCE_RATIOS["target"]) + min_target
    ideal_reach = count - ideal_safety - ideal_target

    selected_safety = safety_schools[: min(ideal_safety, len(safety_schools))]
    selected_target = target_schools[: min(ideal_target, len(target_schools))]
    selected_reach = reach_schools[: min(ideal_reach, len(reach_schools))]

    result = selected_safety + selected_target + selected_reach

    while len(result) < count:
        all_available_schools = [s for s in schools if s not in result]
        if not all_available_schools:
            break
        next_school = max(all_available_schools, key=lambda x: x["probability"])
        result.append(next_school)

    if len(result) > count:
        result = _trim_to_balanced_count(
            result,
            count,
            current_thresholds["safety"],
            current_thresholds["target_lower"],
        )

    result = _ensure_reach_school_present(
        result,
        reach_schools,
        current_thresholds["target_lower"],
        current_thresholds["safety"],
    )

    return result


def _calculate_ideal_counts(
    count: int,
    reach_len: int,
    target_len: int,
    safety_len: int,
) -> tuple[int, int, int]:
    target_to_keep = min(target_len, math.ceil(count * BALANCE_RATIOS["target"]))
    safety_to_keep = min(safety_len, math.ceil(count * BALANCE_RATIOS["safety"]))
    reach_to_keep = count - target_to_keep - safety_to_keep

    if reach_to_keep < 0:
        min_reach_ideal = math.ceil(count * BALANCE_RATIOS["reach"])
        reach_to_keep = min(
            max(1 if min_reach_ideal > 0 else 0, min_reach_ideal if reach_len > 0 else 0),
            reach_len,
        )
        remaining = count - reach_to_keep
        if remaining > 0:
            target_share = BALANCE_RATIOS["target"] / (
                BALANCE_RATIOS["target"] + BALANCE_RATIOS["safety"]
            )
            target_to_keep = min(target_len, math.ceil(remaining * target_share))
            safety_to_keep = min(safety_len, max(0, count - reach_to_keep - target_to_keep))
            if safety_to_keep < (count - reach_to_keep - target_to_keep):
                target_to_keep = min(target_len, count - reach_to_keep - safety_to_keep)
        else:
            target_to_keep, safety_to_keep = 0, 0

    return max(0, reach_to_keep), max(0, target_to_keep), max(0, safety_to_keep)


def _adjust_counts_for_deficit(count: int, counts: list[int], lengths: list[int]) -> list[int]:
    deficit = count - sum(counts)
    if deficit > 0:
        for i in range(len(counts)):
            can_add = lengths[i] - counts[i]
            add = min(deficit, can_add)
            counts[i] += add
            deficit -= add
            if deficit <= 0:
                break
    return counts


def _trim_to_balanced_count(
    result: list[dict[str, Any]],
    count: int,
    safety_thresh_val: float,
    target_thresh_val: float,
) -> list[dict[str, Any]]:
    safety_schools, target_schools, reach_schools = _categorize_schools(
        result, safety_thresh_val, target_thresh_val
    )

    reach_to_keep, target_to_keep, safety_to_keep = _calculate_ideal_counts(
        count, len(reach_schools), len(target_schools), len(safety_schools)
    )

    counts = [reach_to_keep, target_to_keep, safety_to_keep]
    lengths = [len(reach_schools), len(target_schools), len(safety_schools)]
    reach_to_keep, target_to_keep, safety_to_keep = _adjust_counts_for_deficit(
        count, counts, lengths
    )

    return (
        reach_schools[:reach_to_keep]
        + target_schools[:target_to_keep]
        + safety_schools[:safety_to_keep]
    )
