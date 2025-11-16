from typing import Any

import pandas as pd

from src.pages.prediction.result_modifier.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.pages.prediction.result_modifier.config import (
    AGENT_BOUNDARY_SIMILARITY_RANGE,
    AGENT_EXPLORATION_MAX_ROUNDS,
    AGENT_MAX_BOUNDARY_CASES,
    AGENT_NO_CHANGE_THRESHOLD,
    AGENT_TAIL_PERCENTAGE,
    CROSS_MAJOR_SIMILARITY_MIN,
    HIGHER_SIMILARITY_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD,
    TOP_N_RECOMMENDATIONS,
    UNIVERSITY_COUNT_THRESHOLD,
)
from src.pages.prediction.result_modifier.utils import clip_probability
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

def get_similar_major_recommendations(
    results_with_similarity: list[dict[str, Any]], num_target_universities: int
) -> list[dict[str, Any]]:
    if not results_with_similarity:
        return []

    current_threshold = MIN_SIMILARITY_THRESHOLD
    if num_target_universities > 0 and num_target_universities <= UNIVERSITY_COUNT_THRESHOLD:
        current_threshold = HIGHER_SIMILARITY_THRESHOLD

    filtered_by_similarity = [
        res for res in results_with_similarity if res.get("similarity", 0.0) >= current_threshold
    ]

    if not filtered_by_similarity:
        return []

    if len(filtered_by_similarity) <= TOP_N_RECOMMENDATIONS:
        filtered_by_similarity.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        top_candidates = filtered_by_similarity
    else:
        import heapq

        top_candidates = heapq.nlargest(
            TOP_N_RECOMMENDATIONS,
            filtered_by_similarity,
            key=lambda x: x.get("similarity", 0.0),
        )

    for c in top_candidates:
        if isinstance(c, dict) and "probability" in c:
            c["probability"] = clip_probability(c.get("probability", 0.0))

    top_candidates.sort(key=lambda x: x.get("probability", 0.0), reverse=True)

    return top_candidates


def get_cross_major_recommendations(
    results_with_similarity: list[dict[str, Any]],
    background_major: str,
    cases_df: pd.DataFrame | None = None,
    user_specified_combinations: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if not background_major or not results_with_similarity:
        return []

    bg_major_clean = str(background_major).strip()
    admitted_combinations = get_admitted_combinations_from_dataframe(cases_df, bg_major_clean)

    if admitted_combinations:
        admitted_results = [
            res
            for res in results_with_similarity
            if (res.get("university"), res.get("major")) in admitted_combinations
            and res.get("similarity", 1.0) < MIN_SIMILARITY_THRESHOLD
            and res.get("similarity", 0.0) >= CROSS_MAJOR_SIMILARITY_MIN
        ]

        if admitted_results:
            admitted_results.sort(key=lambda x: x.get("similarity", 1.0))
            top_least_similar = admitted_results[:TOP_N_RECOMMENDATIONS]
            top_least_similar.sort(key=lambda x: x.get("probability", 0), reverse=True)
            return top_least_similar
    return []


def adjust_similarity_results_with_agent(
    top_similarity_results: list[dict[str, Any]],
    results_with_similarity: list[dict[str, Any]],
    balance_diff: int,
    background_major: str,
    current_threshold: float,
    agent: Any,
) -> list[dict[str, Any]]:
    if not top_similarity_results or not agent:
        return top_similarity_results

    mode = "relax" if balance_diff > 0 else "tighten"
    top_set = {(r.get("university"), r.get("major")) for r in top_similarity_results}

    evaluated_cases = set()
    no_change_count = 0
    exploration_rounds = 0
    in_exploration_mode = False
    exploration_no_change_count = 0
    result = top_similarity_results.copy()
    adjusted_count = 0
    EARLY_STOP_THRESHOLD = 2

    if mode == "relax":
        boundary_range = (
            AGENT_BOUNDARY_SIMILARITY_RANGE * 1.5
            if current_threshold >= HIGHER_SIMILARITY_THRESHOLD
            else AGENT_BOUNDARY_SIMILARITY_RANGE
        )
        lower_bound = current_threshold - boundary_range
        lower_bound = max(lower_bound, CROSS_MAJOR_SIMILARITY_MIN)
        
        boundary_candidates = [
            r
            for r in results_with_similarity
            if (r.get("university"), r.get("major")) not in top_set
            and lower_bound <= r.get("similarity", 0.0) < current_threshold
        ]
        boundary_candidates.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        boundary_candidates = boundary_candidates[:AGENT_MAX_BOUNDARY_CASES]

        pool_for_exploration = [
            r
            for r in results_with_similarity
            if (r.get("university"), r.get("major")) not in top_set
            and r.get("similarity", 0.0) < current_threshold
            and r.get("similarity", 0.0) >= CROSS_MAJOR_SIMILARITY_MIN
        ]
        pool_for_exploration.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
    else:
        tail_count = max(1, int(len(top_similarity_results) * AGENT_TAIL_PERCENTAGE))
        tail_candidates = sorted(top_similarity_results, key=lambda x: x.get("similarity", 0.0))[
            :tail_count
        ]
        boundary_candidates = tail_candidates[:AGENT_MAX_BOUNDARY_CASES]

        pool_for_exploration = sorted(
            top_similarity_results, key=lambda x: x.get("similarity", 0.0)
        )

    process_count = min(abs(balance_diff), len(boundary_candidates), AGENT_MAX_BOUNDARY_CASES)
    boundary_cases = boundary_candidates[:process_count]

    evaluation_round = 0
    while boundary_cases and exploration_rounds <= AGENT_EXPLORATION_MAX_ROUNDS:
        evaluation_round += 1
        
        if adjusted_count >= abs(balance_diff):
            logger.info(
                f"Agent 已调整 {adjusted_count} 个专业，达到目标 {abs(balance_diff)}，提前停止"
            )
            break
        
        cases_to_evaluate = [
            c
            for c in boundary_cases
            if (c.get("university"), c.get("major")) not in evaluated_cases
        ]

        if not cases_to_evaluate:
            break

        cases_to_evaluate = cases_to_evaluate[:AGENT_MAX_BOUNDARY_CASES]

        for case in cases_to_evaluate:
            evaluated_cases.add((case.get("university"), case.get("major")))

        evaluation_result = agent.evaluate_boundary_cases(background_major, cases_to_evaluate, mode)

        decisions = evaluation_result.get("decisions", [])
        needs_adjustment = evaluation_result.get("needs_adjustment", False)

        if not needs_adjustment:
            if in_exploration_mode:
                exploration_no_change_count += 1
                if exploration_no_change_count >= EARLY_STOP_THRESHOLD:
                    logger.info(
                        f"Agent 探索模式下连续 {exploration_no_change_count} 次无调整，停止评估"
                    )
                    break
            else:
                no_change_count += 1
                if no_change_count >= EARLY_STOP_THRESHOLD:
                    if adjusted_count > 0:
                        logger.info(
                            f"Agent 连续 {no_change_count} 次无调整，已调整 {adjusted_count} 个专业，提前停止评估"
                        )
                        break
                    elif adjusted_count == 0 and abs(balance_diff) > 0:
                        logger.info(
                            f"Agent 连续 {no_change_count} 次无调整，但尚未调整任何专业，进入探索模式"
                        )
                        in_exploration_mode = True
                        exploration_rounds = 1
                        exploration_no_change_count = 0

                        if mode == "tighten":
                            remaining_for_exploration = [
                                r
                                for r in result
                                if (r.get("university"), r.get("major")) not in evaluated_cases
                            ]
                            remaining_for_exploration.sort(key=lambda x: x.get("similarity", 0.0))
                            next_candidates = remaining_for_exploration[:AGENT_MAX_BOUNDARY_CASES]
                        else:
                            next_candidates = [
                                c
                                for c in pool_for_exploration
                                if (c.get("university"), c.get("major")) not in evaluated_cases
                            ][:AGENT_MAX_BOUNDARY_CASES]
                        
                        if not next_candidates:
                            break
                        
                        boundary_cases = next_candidates
                        continue
                    else:
                        logger.info(
                            f"Agent 连续 {no_change_count} 次无调整，无需调整，停止评估"
                        )
                        break
                elif no_change_count >= AGENT_NO_CHANGE_THRESHOLD:
                    logger.info(
                        f"Agent 连续 {no_change_count} 次无调整，进入探索模式"
                    )
                    in_exploration_mode = True
                    exploration_rounds = 1
                    exploration_no_change_count = 0

                    if mode == "tighten":
                        remaining_for_exploration = [
                            r
                            for r in result
                            if (r.get("university"), r.get("major")) not in evaluated_cases
                        ]
                        remaining_for_exploration.sort(key=lambda x: x.get("similarity", 0.0))
                        next_candidates = remaining_for_exploration[:AGENT_MAX_BOUNDARY_CASES]
                    else:
                        next_candidates = [
                            c
                            for c in pool_for_exploration
                            if (c.get("university"), c.get("major")) not in evaluated_cases
                        ][:AGENT_MAX_BOUNDARY_CASES]
                    
                    if not next_candidates:
                        break
                    
                    boundary_cases = next_candidates
                    continue
        else:
            if in_exploration_mode:
                exploration_no_change_count = 0
            else:
                no_change_count = 0

        if mode == "relax":
            for i, decision in enumerate(decisions):
                if i < len(cases_to_evaluate) and decision:
                    case = cases_to_evaluate[i]
                    similarity = case.get("similarity", 0.0)
                    if similarity < CROSS_MAJOR_SIMILARITY_MIN:
                        logger.warning(
                            f"Agent 添加相似度过低的专业到结果中: {case.get('major')} "
                            f"(相似度: {similarity:.3f}, 背景专业: {background_major})"
                        )
                    result.append(case)
                    adjusted_count += 1
        else:
            cases_to_remove = [
                cases_to_evaluate[i]
                for i, decision in enumerate(decisions)
                if i < len(cases_to_evaluate) and decision
            ]
            for case in cases_to_remove:
                similarity = case.get("similarity", 0.0)
                if similarity >= current_threshold:
                    logger.warning(
                        f"Agent 移除相似度较高的专业到结果中: {case.get('major')} "
                        f"(相似度: {similarity:.3f}, 阈值: {current_threshold:.3f}, 背景专业: {background_major})"
                    )
            remove_keys = {(c.get("university"), c.get("major")) for c in cases_to_remove}
            result = [r for r in result if (r.get("university"), r.get("major")) not in remove_keys]
            adjusted_count += len(cases_to_remove)
        
        if adjusted_count >= abs(balance_diff):
            logger.info(
                f"Agent 已调整 {adjusted_count} 个专业，达到目标 {abs(balance_diff)}，提前停止"
            )
            break

        if in_exploration_mode:
            exploration_rounds += 1
            if exploration_rounds > AGENT_EXPLORATION_MAX_ROUNDS:
                logger.info(f"Agent 探索模式已达到最大轮数 {AGENT_EXPLORATION_MAX_ROUNDS}，停止评估")
                break

            if mode == "tighten":
                remaining_for_exploration = [
                    r
                    for r in result
                    if (r.get("university"), r.get("major")) not in evaluated_cases
                ]
                remaining_for_exploration.sort(key=lambda x: x.get("similarity", 0.0))
                next_candidates = remaining_for_exploration[:AGENT_MAX_BOUNDARY_CASES]
            else:
                next_candidates = [
                    c
                    for c in pool_for_exploration
                    if (c.get("university"), c.get("major")) not in evaluated_cases
                ][:AGENT_MAX_BOUNDARY_CASES]
            
            if not next_candidates:
                break
            
            boundary_cases = next_candidates
        else:
            if mode == "relax":
                boundary_candidates = [
                    c
                    for c in boundary_candidates
                    if (c.get("university"), c.get("major")) not in evaluated_cases
                ]
                if not boundary_candidates:
                    break
                boundary_cases = boundary_candidates[:AGENT_MAX_BOUNDARY_CASES]
            else:
                remaining_tail = [
                    r
                    for r in result
                    if (r.get("university"), r.get("major")) not in evaluated_cases
                ]
                remaining_tail.sort(key=lambda x: x.get("similarity", 0.0))
                tail_count = max(1, int(len(remaining_tail) * AGENT_TAIL_PERCENTAGE))
                boundary_cases = remaining_tail[:min(tail_count, AGENT_MAX_BOUNDARY_CASES)]

    for c in result:
        if isinstance(c, dict) and "probability" in c:
            c["probability"] = clip_probability(c.get("probability", 0.0))

    result.sort(key=lambda x: x.get("probability", 0.0), reverse=True)

    return result
