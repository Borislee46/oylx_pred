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

boundary_ranker_logger = setup_logger("page3", "prediction")


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
        boundary_ranker_logger.warning(
            "[边界调整] 跳过处理 - top_similarity_results为空或agent为空"
        )
        return top_similarity_results

    mode = "relax" if balance_diff > 0 else "tighten"
    top_set = {(r.get("university"), r.get("major")) for r in top_similarity_results}

    boundary_ranker_logger.info(
        f"[边界调整] 开始处理 - 模式: {mode}, 平衡差: {balance_diff}, "
        f"当前阈值: {current_threshold}, 背景专业: {background_major}, "
        f"top结果数: {len(top_similarity_results)}, 总候选数: {len(results_with_similarity)}"
    )

    evaluated_cases = set()
    no_change_count = 0
    exploration_rounds = 0
    in_exploration_mode = False
    exploration_no_change_count = 0
    result = top_similarity_results.copy()

    if mode == "relax":
        lower_bound = current_threshold - AGENT_BOUNDARY_SIMILARITY_RANGE
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
        ]
        pool_for_exploration.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        boundary_ranker_logger.info(
            f"[边界调整-放宽] 边界候选范围: [{lower_bound:.3f}, {current_threshold:.3f}), "
            f"边界候选数: {len(boundary_candidates)}, 探索池大小: {len(pool_for_exploration)}"
        )
        if boundary_candidates:
            sim_range = (
                boundary_candidates[-1].get("similarity", 0.0),
                boundary_candidates[0].get("similarity", 0.0),
            )
            boundary_ranker_logger.debug(
                f"[边界调整-放宽] 边界候选相似度范围: [{sim_range[0]:.3f}, {sim_range[1]:.3f}]"
            )
    else:
        tail_count = max(1, int(len(top_similarity_results) * AGENT_TAIL_PERCENTAGE))
        tail_candidates = sorted(top_similarity_results, key=lambda x: x.get("similarity", 0.0))[
            :tail_count
        ]
        boundary_candidates = tail_candidates[:AGENT_MAX_BOUNDARY_CASES]

        pool_for_exploration = sorted(
            top_similarity_results, key=lambda x: x.get("similarity", 0.0)
        )

        boundary_ranker_logger.info(
            f"[边界调整-收紧] 末端百分比: {AGENT_TAIL_PERCENTAGE}, "
            f"末端候选数: {len(boundary_candidates)}, 总结果数: {len(top_similarity_results)}"
        )
        if boundary_candidates:
            sim_range = (
                boundary_candidates[0].get("similarity", 0.0),
                boundary_candidates[-1].get("similarity", 0.0),
            )
            boundary_ranker_logger.debug(
                f"[边界调整-收紧] 末端候选相似度范围: [{sim_range[0]:.3f}, {sim_range[1]:.3f}]"
            )

    process_count = min(abs(balance_diff), len(boundary_candidates), AGENT_MAX_BOUNDARY_CASES)
    boundary_cases = boundary_candidates[:process_count]

    boundary_ranker_logger.info(
        f"[边界调整] 初始处理 - 处理数量: {process_count}, "
        f"边界候选总数: {len(boundary_candidates)}, "
        f"最大处理数: {AGENT_MAX_BOUNDARY_CASES}"
    )

    evaluation_round = 0
    while boundary_cases and exploration_rounds <= AGENT_EXPLORATION_MAX_ROUNDS:
        evaluation_round += 1
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

        boundary_ranker_logger.info(
            f"[边界调整] 评估轮次 #{evaluation_round} - "
            f"评估数量: {len(cases_to_evaluate)}, "
            f"探索模式: {in_exploration_mode}, "
            f"探索轮次: {exploration_rounds}/{AGENT_EXPLORATION_MAX_ROUNDS}, "
            f"无需调整计数: {no_change_count if not in_exploration_mode else exploration_no_change_count}/{AGENT_NO_CHANGE_THRESHOLD}"
        )

        if cases_to_evaluate:
            case_info = [
                f"{c.get('university', '')}-{c.get('major', '')[:30]}"
                f"(sim:{c.get('similarity', 0.0):.3f},prob:{c.get('probability', 0.0):.3f})"
                for c in cases_to_evaluate[:5]
            ]
            boundary_ranker_logger.debug(f"[边界调整] 待评估案例(前5个): {', '.join(case_info)}")

        evaluation_result = agent.evaluate_boundary_cases(background_major, cases_to_evaluate, mode)

        decisions = evaluation_result.get("decisions", [])
        needs_adjustment = evaluation_result.get("needs_adjustment", False)
        reason = evaluation_result.get("reason", "")

        boundary_ranker_logger.info(
            f"[边界调整] Agent评估结果 - 需要调整: {needs_adjustment}, "
            f"决策数量: {len(decisions)}, 原因: {reason[:100] if reason else '无'}"
        )

        if decisions:
            true_count = sum(decisions)
            boundary_ranker_logger.debug(
                f"[边界调整] 决策详情 - 通过: {true_count}/{len(decisions)}, "
                f"拒绝: {len(decisions) - true_count}/{len(decisions)}"
            )

        if not needs_adjustment:
            if in_exploration_mode:
                exploration_no_change_count += 1
                boundary_ranker_logger.info(
                    f"[边界调整] 探索模式无需调整 - 计数: {exploration_no_change_count}/{AGENT_NO_CHANGE_THRESHOLD}"
                )
                if exploration_no_change_count >= AGENT_NO_CHANGE_THRESHOLD:
                    boundary_ranker_logger.info("[边界调整] 探索模式连续无需调整达到阈值，停止处理")
                    break
            else:
                no_change_count += 1
                boundary_ranker_logger.info(
                    f"[边界调整] 无需调整计数 - {no_change_count}/{AGENT_NO_CHANGE_THRESHOLD}"
                )
                if no_change_count >= AGENT_NO_CHANGE_THRESHOLD:
                    in_exploration_mode = True
                    exploration_rounds = 1
                    exploration_no_change_count = 0

                    boundary_ranker_logger.info(
                        f"[边界调整] 进入探索模式 - 连续{AGENT_NO_CHANGE_THRESHOLD}次无需调整，"
                        f"从池子中捞取下一个候选"
                    )

                    next_candidates = [
                        c
                        for c in pool_for_exploration
                        if (c.get("university"), c.get("major")) not in evaluated_cases
                    ]
                    if not next_candidates:
                        boundary_ranker_logger.warning("[边界调整] 探索池中无更多候选，停止处理")
                        break

                    boundary_cases = next_candidates[:AGENT_MAX_BOUNDARY_CASES]
                    boundary_ranker_logger.info(
                        f"[边界调整] 从探索池捞取 {len(boundary_cases)} 个候选"
                    )
                    continue
        else:
            if in_exploration_mode:
                exploration_no_change_count = 0
            else:
                no_change_count = 0

        if mode == "relax":
            added_count = 0
            for i, decision in enumerate(decisions):
                if i < len(cases_to_evaluate) and decision:
                    result.append(cases_to_evaluate[i])
                    added_count += 1
            if added_count > 0:
                boundary_ranker_logger.info(f"[边界调整-放宽] 新增 {added_count} 个案例到结果中")
        else:
            cases_to_remove = [
                cases_to_evaluate[i]
                for i, decision in enumerate(decisions)
                if i < len(cases_to_evaluate) and decision
            ]
            remove_keys = {(c.get("university"), c.get("major")) for c in cases_to_remove}
            removed_count = len(remove_keys)
            result = [r for r in result if (r.get("university"), r.get("major")) not in remove_keys]
            if removed_count > 0:
                boundary_ranker_logger.info(f"[边界调整-收紧] 从结果中移除 {removed_count} 个案例")

        if in_exploration_mode:
            exploration_rounds += 1
            if exploration_rounds > AGENT_EXPLORATION_MAX_ROUNDS:
                boundary_ranker_logger.info(
                    f"[边界调整] 探索轮次达到上限 {AGENT_EXPLORATION_MAX_ROUNDS}，停止处理"
                )
                break

            next_candidates = [
                c
                for c in pool_for_exploration
                if (c.get("university"), c.get("major")) not in evaluated_cases
            ]
            if not next_candidates:
                boundary_ranker_logger.warning("[边界调整] 探索池中无更多候选，停止处理")
                break

            boundary_cases = next_candidates[:AGENT_MAX_BOUNDARY_CASES]
            boundary_ranker_logger.debug(
                f"[边界调整] 探索轮次 {exploration_rounds} - 捞取 {len(boundary_cases)} 个候选"
            )
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
                boundary_cases = remaining_tail[:tail_count][:AGENT_MAX_BOUNDARY_CASES]

    for c in result:
        if isinstance(c, dict) and "probability" in c:
            c["probability"] = clip_probability(c.get("probability", 0.0))

    result.sort(key=lambda x: x.get("probability", 0.0), reverse=True)

    boundary_ranker_logger.info(
        f"[边界调整] 处理完成 - 最终结果数: {len(result)}, "
        f"原始结果数: {len(top_similarity_results)}, "
        f"变化: {len(result) - len(top_similarity_results):+d}, "
        f"总评估轮次: {evaluation_round}, "
        f"已评估案例数: {len(evaluated_cases)}"
    )

    return result
