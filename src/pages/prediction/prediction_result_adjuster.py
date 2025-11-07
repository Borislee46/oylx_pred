from typing import Optional

from src.pages.prediction.prediction_utils import is_new_major
from src.pages.prediction.result_modifier.probability_adjuster import ProbabilityAdjuster
from src.pages.prediction.result_modifier.text_boost_provider import TextBoostProvider
from src.utils.logger import setup_logger

prediction_handler_logger = setup_logger("page3", "prediction")


def pipeline_adjust_results(
    results: list[dict[str, float | str]],
    probability_adjuster: Optional[ProbabilityAdjuster],
    text_boost_provider: Optional[TextBoostProvider],
    experience_details: dict[str, str],
    gpa: Optional[float],
    language_score: Optional[float],
    background_university: Optional[str],
    is_new_major_cache: Optional[dict[tuple[str, str], bool]] = None,
) -> list[dict[str, float | str]]:
    if not results or not isinstance(results, list):
        return results

    dict_indices = [idx for idx, r in enumerate(results) if isinstance(r, dict)]
    if not dict_indices:
        return results

    base_probs: list[float] = []
    for i in dict_indices:
        prob_value = results[i].get("probability", 0.0)
        if isinstance(prob_value, (int, float)):
            base_probs.append(float(prob_value))
        else:
            base_probs.append(0.0)

    if probability_adjuster and gpa is not None and language_score is not None:
        adjusted_probs: list[float] = []
        failed_count = 0
        for idx, p in enumerate(base_probs):
            try:
                if not isinstance(p, (int, float)) or p < 0 or p > 1:
                    prediction_handler_logger.warning(
                        f"无效的概率值 {p} (索引 {dict_indices[idx]}), 将被限制到 [0, 1]"
                    )
                    p = max(0.0, min(1.0, float(p)))

                ap = probability_adjuster.adjust_probability(
                    p,
                    gpa,
                    language_score,
                    background_university_name=background_university,
                )
                adjusted_prob = float(ap)

                if adjusted_prob < 0 or adjusted_prob > 1:
                    prediction_handler_logger.warning(
                        f"概率调整后超出范围: {adjusted_prob}, 将被限制到 [0, 1]"
                    )
                    adjusted_prob = max(0.0, min(1.0, adjusted_prob))

                adjusted_probs.append(adjusted_prob)
            except Exception as e:
                failed_count += 1
                prediction_handler_logger.warning(
                    f"概率调整失败 (索引 {dict_indices[idx]}, 概率 {p}): {e}", exc_info=True
                )
                adjusted_probs.append(max(0.0, min(1.0, float(p))))

        if failed_count > 0:
            prediction_handler_logger.warning(
                f"概率调整阶段有 {failed_count}/{len(base_probs)} 个结果失败"
            )
    else:
        adjusted_probs = base_probs
        if probability_adjuster is None:
            prediction_handler_logger.debug("跳过概率调整：缺少概率调整器或 GPA/语言分数")

    if text_boost_provider is not None and isinstance(experience_details, dict):
        try:
            boosted_probs, boost_info = text_boost_provider.apply(
                adjusted_probs, experience_details
            )
            if boost_info:
                prediction_handler_logger.debug(f"文本增强应用成功: {boost_info}")
        except Exception as e:
            prediction_handler_logger.warning(f"文本增强失败，使用调整后的概率: {e}", exc_info=True)
            boosted_probs = adjusted_probs
    else:
        boosted_probs = adjusted_probs

    failed_assignments = 0
    for pos, idx in enumerate(dict_indices):
        if pos < len(boosted_probs):
            try:
                final_prob = float(boosted_probs[pos])
                if final_prob < 0 or final_prob > 1:
                    prediction_handler_logger.warning(
                        f"最终概率超出范围: {final_prob}, 将被限制到 [0, 1]"
                    )
                    final_prob = max(0.0, min(1.0, final_prob))
                results[idx]["probability"] = final_prob
            except (ValueError, TypeError, KeyError) as e:
                failed_assignments += 1
                prediction_handler_logger.warning(f"概率赋值失败 (索引 {idx}): {e}", exc_info=True)
                if "probability" not in results[idx]:
                    results[idx]["probability"] = 0.0
        else:
            prediction_handler_logger.warning(
                f"索引越界: 结果索引 {idx} 超出增强概率列表长度 {len(boosted_probs)}"
            )

    if failed_assignments > 0:
        prediction_handler_logger.warning(
            f"概率赋值阶段有 {failed_assignments}/{len(dict_indices)} 个结果失败"
        )

    if is_new_major_cache is not None:
        for idx in dict_indices:
            r = results[idx]
            uni_value = r.get("university")
            major_value = r.get("major")
            if isinstance(uni_value, str) and isinstance(major_value, str):
                key: tuple[str, str] = (uni_value, major_value)
                r["is_new_major"] = is_new_major_cache.get(key, False)
            else:
                r["is_new_major"] = False

    return results


def batch_adjust_results(
    results_list: list[list[dict[str, float | str]]],
    probability_adjuster: Optional[ProbabilityAdjuster],
    text_boost_provider: Optional[TextBoostProvider],
    experience_details: dict[str, str],
    gpa: Optional[float],
    language_score: Optional[float],
    background_university: Optional[str],
) -> list[list[dict[str, float | str]]]:
    if not results_list or not any(results_list):
        return results_list

    all_combinations: set[tuple[str, str]] = set()
    for results in results_list:
        if results and isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    uni_value = r.get("university")
                    major_value = r.get("major")
                    if isinstance(uni_value, str) and isinstance(major_value, str):
                        all_combinations.add((uni_value, major_value))

    is_new_major_cache: dict[tuple[str, str], bool] = {}
    if all_combinations:
        try:
            for uni, major in all_combinations:
                is_new_major_cache[(uni, major)] = is_new_major(uni, major)
        except Exception as e:
            prediction_handler_logger.warning(f"批量查询新专业失败: {e}")

    adjusted_results_list = []
    for results in results_list:
        adjusted = pipeline_adjust_results(
            results,
            probability_adjuster,
            text_boost_provider,
            experience_details,
            gpa,
            language_score,
            background_university,
            is_new_major_cache,
        )
        adjusted_results_list.append(adjusted)

    return adjusted_results_list
