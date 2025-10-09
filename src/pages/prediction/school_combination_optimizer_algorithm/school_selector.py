import math
from typing import Any

from src.pages.prediction.school_combination_optimizer_algorithm.optimizer_config import (
    BALANCE_RATIOS,
    SCHOOL_CATEGORY_THRESHOLDS,
)


def reduce_schools_balanced(
    schools: list[dict[str, Any]], max_count: int, adaptive_thresholds: dict[str, float] = None
) -> list[dict[str, Any]]:
    if len(schools) <= max_count:
        return schools

    current_thresholds = adaptive_thresholds or SCHOOL_CATEGORY_THRESHOLDS
    safety_thresh_val = current_thresholds["safety"]
    target_thresh_val = current_thresholds["target_lower"]

    safety_schools = [s for s in schools if s["probability"] >= safety_thresh_val]
    target_schools = [
        s for s in schools if target_thresh_val <= s["probability"] < safety_thresh_val
    ]
    reach_schools = [s for s in schools if s["probability"] < target_thresh_val]

    ideal_safety = max_count * BALANCE_RATIOS["safety"]
    ideal_target = max_count * BALANCE_RATIOS["target"]
    ideal_reach = max_count * BALANCE_RATIOS["reach"]

    current_safety = len(safety_schools)
    current_target = len(target_schools)
    current_reach = len(reach_schools)

    safety_schools = sorted(safety_schools, key=lambda x: x["probability"], reverse=True)
    target_schools = sorted(target_schools, key=lambda x: x["probability"], reverse=True)
    reach_schools = sorted(reach_schools, key=lambda x: x["probability"], reverse=True)

    total_to_remove = len(schools) - max_count
    over_safety = max(0, current_safety - ideal_safety)
    over_target = max(0, current_target - ideal_target)
    over_reach = max(0, current_reach - ideal_reach)
    total_over = over_safety + over_target + over_reach

    if total_over > 0:
        safety_to_remove = int(total_to_remove * over_safety / total_over)
        target_to_remove = int(total_to_remove * over_target / total_over)
        reach_to_remove = total_to_remove - safety_to_remove - target_to_remove
    else:
        safety_to_remove = int(total_to_remove * current_safety / len(schools))
        target_to_remove = int(total_to_remove * current_target / len(schools))
        reach_to_remove = total_to_remove - safety_to_remove - target_to_remove

    safety_to_remove = min(safety_to_remove, current_safety)
    target_to_remove = min(target_to_remove, current_target)
    reach_to_remove = min(reach_to_remove, current_reach)

    safety_schools = safety_schools[: current_safety - safety_to_remove]
    target_schools = target_schools[: current_target - target_to_remove]
    reach_schools = reach_schools[: current_reach - reach_to_remove]
    result = safety_schools + target_schools + reach_schools

    if len(result) > max_count:
        result = sorted(result, key=lambda x: x["probability"])[:max_count]

    while len(result) < max_count and (
        len(safety_schools) + len(target_schools) + len(reach_schools) < len(schools)
    ):
        remaining_schools = [s for s in schools if s not in result]
        if not remaining_schools:
            break
        next_school = max(remaining_schools, key=lambda x: x["probability"])
        result.append(next_school)

    return result


def generate_balanced_selection(
    schools: list[dict[str, Any]],
    min_count: int,
    max_count: int,
    adaptive_thresholds: dict[str, float] = None,
) -> list[dict[str, Any]]:
    count = min(max(min_count, max_count // 2), len(schools))

    current_thresholds = adaptive_thresholds or SCHOOL_CATEGORY_THRESHOLDS
    safety_thresh_val = current_thresholds["safety"]
    target_thresh_val = current_thresholds["target_lower"]

    safety_schools = [s for s in schools if s["probability"] >= safety_thresh_val]
    target_schools = [
        s for s in schools if target_thresh_val <= s["probability"] < safety_thresh_val
    ]
    reach_schools = [s for s in schools if s["probability"] < target_thresh_val]

    safety_schools = sorted(safety_schools, key=lambda x: x["probability"], reverse=True)
    target_schools = sorted(target_schools, key=lambda x: x["probability"], reverse=True)
    reach_schools = sorted(reach_schools, key=lambda x: x["probability"], reverse=True)

    min_safety = min(1, len(safety_schools))
    min_target = min(1, len(target_schools))
    min_reach = min(1, len(reach_schools))

    remaining = count - min_safety - min_target - min_reach

    if remaining < 0:
        selected_safety = safety_schools[:min_safety]
        selected_target = target_schools[:min_target]
        selected_reach = reach_schools[:min_reach]
        result = selected_reach + selected_target + selected_safety
        return result[:count]

    ideal_safety = round(remaining * 0.3) + min_safety
    ideal_target = round(remaining * 0.4) + min_target
    ideal_reach = count - ideal_safety - ideal_target

    selected_safety = safety_schools[: min(ideal_safety, len(safety_schools))]
    selected_target = target_schools[: min(ideal_target, len(target_schools))]
    selected_reach = reach_schools[: min(ideal_reach, len(reach_schools))]

    result = selected_safety + selected_target + selected_reach

    while len(result) < count:
        current_safety = len([s for s in result if s["probability"] >= safety_thresh_val])
        current_target = len(
            [s for s in result if target_thresh_val <= s["probability"] < safety_thresh_val]
        )
        current_reach = len([s for s in result if s["probability"] < target_thresh_val])

        diff_safety = ideal_safety - current_safety
        diff_target = ideal_target - current_target
        diff_reach = ideal_reach - current_reach

        if (
            diff_reach > 0
            and diff_reach >= diff_target
            and diff_reach >= diff_safety
            and len(selected_reach) < len(reach_schools)
        ):
            next_reach = [s for s in reach_schools if s not in selected_reach]
            if next_reach:
                selected_reach.append(next_reach[0])

        elif (
            diff_target > 0
            and diff_target >= diff_safety
            and len(selected_target) < len(target_schools)
        ):
            next_target = [s for s in target_schools if s not in selected_target]
            if next_target:
                selected_target.append(next_target[0])

        elif diff_safety > 0 and len(selected_safety) < len(safety_schools):
            next_safety = [s for s in safety_schools if s not in selected_safety]
            if next_safety:
                selected_safety.append(next_safety[0])
        else:
            remaining_reach = [s for s in reach_schools if s not in result]
            if remaining_reach:
                result.append(remaining_reach[0])
                continue

            remaining_target = [s for s in target_schools if s not in result]
            if remaining_target:
                result.append(remaining_target[0])
                continue

            remaining_safety = [s for s in safety_schools if s not in result]
            if remaining_safety:
                result.append(remaining_safety[0])
                continue

            remaining_schools = [s for s in schools if s not in result]
            if not remaining_schools:
                break

            result.append(remaining_schools[0])
            continue

        result = selected_safety + selected_target + selected_reach

    if len(result) > count:
        result = _trim_to_balanced_count(result, count, safety_thresh_val, target_thresh_val)

    if not any(s["probability"] < target_thresh_val for s in result) and reach_schools:
        safety_in_result = [s for s in result if s["probability"] >= safety_thresh_val]
        if safety_in_result and len(safety_in_result) > 1:
            result.remove(safety_in_result[-1])
            result.append(reach_schools[0])

    return result


def _trim_to_balanced_count(
    result: list[dict[str, Any]],
    count: int,
    safety_thresh_val: float,
    target_thresh_val: float,
) -> list[dict[str, Any]]:
    reach_schools_in_result = [s for s in result if s["probability"] < target_thresh_val]
    target_schools_in_result = [
        s for s in result if target_thresh_val <= s["probability"] < safety_thresh_val
    ]
    safety_schools_in_result = [s for s in result if s["probability"] >= safety_thresh_val]

    target_to_keep = min(len(target_schools_in_result), math.ceil(count * BALANCE_RATIOS["target"]))
    safety_to_keep = min(len(safety_schools_in_result), math.ceil(count * BALANCE_RATIOS["safety"]))
    reach_to_keep = count - target_to_keep - safety_to_keep

    if reach_to_keep < 0:
        min_reach_ideal = math.ceil(count * BALANCE_RATIOS["reach"])
        reach_to_keep = min(
            max(
                1 if min_reach_ideal > 0 else 0,
                min_reach_ideal if len(reach_schools_in_result) > 0 else 0,
            ),
            len(reach_schools_in_result),
        )

        remaining_for_ts = count - reach_to_keep
        if remaining_for_ts < 0:
            remaining_for_ts = 0

        if remaining_for_ts > 0:
            target_share_of_remaining = BALANCE_RATIOS["target"] / (
                BALANCE_RATIOS["target"] + BALANCE_RATIOS["safety"]
            )
            target_to_keep = min(
                len(target_schools_in_result),
                math.ceil(remaining_for_ts * target_share_of_remaining),
            )
            safety_to_keep = count - reach_to_keep - target_to_keep
            safety_to_keep = min(len(safety_schools_in_result), max(0, safety_to_keep))
            if safety_to_keep < (count - reach_to_keep - target_to_keep):
                target_to_keep = min(
                    len(target_schools_in_result), count - reach_to_keep - safety_to_keep
                )
        else:
            target_to_keep = 0
            safety_to_keep = 0

    target_to_keep = min(len(target_schools_in_result), max(0, target_to_keep))
    safety_to_keep = min(len(safety_schools_in_result), max(0, safety_to_keep))
    reach_to_keep = min(len(reach_schools_in_result), max(0, reach_to_keep))

    current_sum = target_to_keep + safety_to_keep + reach_to_keep
    deficit = count - current_sum

    if deficit > 0:
        can_add = len(target_schools_in_result) - target_to_keep
        add = min(deficit, can_add)
        target_to_keep += add
        deficit -= add

        if deficit > 0:
            can_add = len(safety_schools_in_result) - safety_to_keep
            add = min(deficit, can_add)
            safety_to_keep += add
            deficit -= add

            if deficit > 0:
                can_add = len(reach_schools_in_result) - reach_to_keep
                add = min(deficit, can_add)
                reach_to_keep += add

    kept_reach = sorted(reach_schools_in_result, key=lambda x: x["probability"], reverse=True)[
        :reach_to_keep
    ]
    kept_target = sorted(target_schools_in_result, key=lambda x: x["probability"], reverse=True)[
        :target_to_keep
    ]
    kept_safety = sorted(safety_schools_in_result, key=lambda x: x["probability"], reverse=True)[
        :safety_to_keep
    ]

    return kept_safety + kept_target + kept_reach
