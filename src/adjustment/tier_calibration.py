from typing import Any

import numpy as np

from src.adjustment.config import (
    TIER_BOUNDARIES,
    TIER_RANK_REPAIR_MAX_ITER,
    TIER_RANK_REPAIR_MIN_VIOLATION,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce, clip_scalar, float_eq

logger = setup_logger("page3", "prediction")


def build_tier_map(difficulty_order: tuple[str, ...]) -> dict[str, int]:
    boundaries = list(TIER_BOUNDARIES)
    tier_map: dict[str, int] = {}
    for i, name in enumerate(difficulty_order):
        if i < boundaries[0]:
            tier_map[name] = 1
        elif i < boundaries[1]:
            tier_map[name] = 2
        elif i < boundaries[2]:
            tier_map[name] = 3
        else:
            tier_map[name] = 4
    return tier_map


def _solve_adjacent_tier_pair(
    q_lower: list[float],
    q_upper: list[float],
    min_violation: float = 0.08,
) -> tuple[list[float], list[float], float | None]:
    lower = np.asarray(q_lower, dtype=float)
    upper = np.asarray(q_upper, dtype=float)

    if len(lower) == 0 or len(upper) == 0:
        return q_lower, q_upper, None

    max_lower = float(lower.max())
    min_upper = float(upper.min())

    if max_lower <= min_upper + min_violation:
        return q_lower, q_upper, None

    mid = (max_lower + min_upper) / 2.0

    adj_lower = lower.copy()
    adj_upper = upper.copy()

    adj_lower[np.abs(adj_lower - max_lower) < 1e-9] = mid
    adj_upper[np.abs(adj_upper - min_upper) < 1e-9] = mid

    return adj_lower.tolist(), adj_upper.tolist(), mid


def apply_tier_rank_repair(
    results: list[dict[str, Any]],
    tier_map: dict[str, int],
) -> list[dict[str, Any]]:
    """启发式 tier 排序修复：相邻 extrema 拉平，非 isotonic regression。

    允许 TIER_RANK_REPAIR_MIN_VIOLATION (默认 0.08) 以内的跨 tier 倒挂。
    当 max(T_lower) > min(T_upper) + min_violation 时，将极值拉至中点。
    这不是真正的 isotonic calibration / Dykstra 投影——仅做相邻 pair 修复。
    """
    if len(results) < 2:
        return [dict(r) for r in results]

    tier_indices: dict[int, list[int]] = {}
    untiered_indices: list[int] = []

    for i, r in enumerate(results):
        tier = tier_map.get(str(r.get("university", "")))
        if tier is None:
            untiered_indices.append(i)
        else:
            tier_indices.setdefault(tier, []).append(i)

    available_tiers = sorted(tier_indices.keys())
    if len(available_tiers) < 2:
        logger.debug(
            "Tier Rank Repair 跳过 | n_results=%d n_tiered=%d available_tiers=%d (需≥2)",
            len(results),
            len(results) - len(untiered_indices),
            len(available_tiers),
        )
        return [dict(r) for r in results]

    tier_sizes = {t: len(tier_indices[t]) for t in available_tiers}
    logger.info(
        "Tier Rank Repair 开始 | n_results=%d tiers=%s tier_sizes=%s untiered=%d",
        len(results),
        available_tiers,
        tier_sizes,
        len(untiered_indices),
    )

    tier_probs: dict[int, list[float]] = {}
    for t in available_tiers:
        tier_probs[t] = [float(results[i]["probability"]) for i in tier_indices[t]]

    max_violation = 0.0
    for t_idx in range(len(available_tiers) - 1):
        t_lower = available_tiers[t_idx]
        t_upper = available_tiers[t_idx + 1]
        lo = max(tier_probs[t_lower]) if tier_probs[t_lower] else 0.0
        hi = min(tier_probs[t_upper]) if tier_probs[t_upper] else 1.0
        violation = lo - hi
        if violation > max_violation:
            max_violation = violation

    if max_violation < TIER_RANK_REPAIR_MIN_VIOLATION:
        logger.info(
            "Tier Rank Repair 跳过 | max_violation=%.4f < min_violation=%.2f "
            "(概率已保序或偏差在噪声范围内)",
            max_violation,
            TIER_RANK_REPAIR_MIN_VIOLATION,
        )
        return [dict(r) for r in results]

    n_adjustments = 0
    for iteration in range(TIER_RANK_REPAIR_MAX_ITER):
        max_delta = 0.0
        pair_violations = 0
        for t_idx in range(len(available_tiers) - 1):
            t_lower = available_tiers[t_idx]
            t_upper = available_tiers[t_idx + 1]

            old_lower = tier_probs[t_lower][:]
            old_upper = tier_probs[t_upper][:]

            adj_lower, adj_upper, midpoint = _solve_adjacent_tier_pair(
                tier_probs[t_lower],
                tier_probs[t_upper],
                min_violation=TIER_RANK_REPAIR_MIN_VIOLATION,
            )

            tier_probs[t_lower] = adj_lower
            tier_probs[t_upper] = adj_upper

            for old, new in zip(old_lower, adj_lower, strict=False):
                max_delta = max(max_delta, abs(new - old))
            for old, new in zip(old_upper, adj_upper, strict=False):
                max_delta = max(max_delta, abs(new - old))

            if midpoint is not None:
                pair_violations += 1
                n_adjustments += 1
                logger.debug(
                    "Tier Rank Repair 修正 | iter=%d T%d→T%d midpoint=%.4f "
                    "max(T%d)=%.4f min(T%d)=%.4f",
                    iteration + 1,
                    t_lower,
                    t_upper,
                    midpoint,
                    t_lower,
                    max(old_lower),
                    t_upper,
                    min(old_upper),
                )

        if pair_violations > 0:
            logger.debug(
                "Tier Rank Repair iter=%d | violations=%d max_delta=%.6f",
                iteration + 1,
                pair_violations,
                max_delta,
            )

        if max_delta < 1e-9:
            logger.info(
                "Tier Rank Repair 收敛 | iter=%d total_adjustments=%d",
                iteration + 1,
                n_adjustments,
            )
            break
    else:
        logger.warning(
            "Tier Rank Repair 未收敛 | max_iter=%d total_adjustments=%d",
            TIER_RANK_REPAIR_MAX_ITER,
            n_adjustments,
        )

    adjusted = [dict(r) for r in results]
    for t in available_tiers:
        indices = tier_indices[t]
        probs = tier_probs[t]
        for idx, new_prob in zip(indices, probs, strict=False):
            old_prob = clip_probability_coerce(adjusted[idx].get("probability"))
            # step 的 after/delta 必须与最终赋值完全一致（clip+round 一次到位）。
            final_prob = round(clip_scalar(new_prob, 0.005, 0.95), 6)
            delta = final_prob - old_prob
            if float_eq(delta, 0.0):
                continue
            adjusted[idx]["probability"] = final_prob

            trace = dict(adjusted[idx].get("_adjustment_trace", {}))
            trace["tier_rank_repair"] = round(delta, 6)
            adjusted[idx]["_adjustment_trace"] = trace

            steps = list(adjusted[idx].get("_adjustment_steps", []))
            steps.append(
                {
                    "name": "Tier Rank Repair",
                    "before": round(old_prob, 6),
                    "after": final_prob,
                    "delta": round(delta, 6),
                    "type": "calibration",
                    "description": (
                        f"T{t} 排序修复 | 相邻极值拉平 (容忍 ≤{TIER_RANK_REPAIR_MIN_VIOLATION:.0%} 倒挂)"
                    ),
                }
            )
            adjusted[idx]["_adjustment_steps"] = steps

    return adjusted
