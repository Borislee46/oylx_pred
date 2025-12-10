from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st

from src.pages.prediction.prediction_utils import is_new_major
from src.pages.prediction.result_modifier.probability_adjuster import ProbabilityAdjuster
from src.pages.prediction.result_modifier.text_boost_provider import TextBoostProvider
from src.utils.logger import setup_logger

prediction_handler_logger = setup_logger("page3", "prediction")


def _validate_probability(prob: Any) -> float:
    try:
        val = float(prob)
    except (ValueError, TypeError):
        return 0.0

    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def pipeline_adjust_results(
    results: List[Dict[str, Union[float, str]]],
    probability_adjuster: Optional[ProbabilityAdjuster],
    text_boost_provider: Optional[TextBoostProvider],
    experience_details: Dict[str, str],
    gpa: Optional[float],
    language_score: Optional[float],
    background_university: Optional[str],
    is_new_major_cache: Optional[Dict[Tuple[str, str], bool]] = None,
) -> List[Dict[str, Union[float, str]]]:
    if not results or not isinstance(results, list):
        return results

    dict_indices = [idx for idx, r in enumerate(results) if isinstance(r, dict)]
    if not dict_indices:
        return results

    base_probs: List[float] = []
    for i in dict_indices:
        prob_value = results[i].get("probability", 0.0)
        base_probs.append(_validate_probability(prob_value))

    adjusted_probs: List[float] = []
    if probability_adjuster and gpa is not None and language_score is not None:
        for idx, p in enumerate(base_probs):
            try:
                ap = probability_adjuster.adjust_probability(p, gpa, language_score)
                adjusted_probs.append(_validate_probability(ap))
            except Exception as e:
                prediction_handler_logger.warning(
                    f"概率调整计算出错 (索引 {dict_indices[idx]}, 概率 {p}): {e}"
                )
                adjusted_probs.append(p)
    else:
        adjusted_probs = base_probs
        if probability_adjuster is None:
            prediction_handler_logger.debug("跳过概率调整：缺少概率调整器或 GPA/语言分数")

    boosted_probs = adjusted_probs
    if text_boost_provider is not None and isinstance(experience_details, dict):
        try:
            apply_result = text_boost_provider.apply(adjusted_probs, experience_details)
            if apply_result:
                result_probs, boost_info = apply_result
                if boost_info:
                    prediction_handler_logger.debug(f"文本增强应用成功: {boost_info}")
                boosted_probs = [_validate_probability(p) for p in result_probs]
        except Exception as e:
            prediction_handler_logger.warning(f"文本增强失败，使用调整后的概率: {e}")

    for pos, idx in enumerate(dict_indices):
        if pos < len(boosted_probs):
            results[idx]["probability"] = boosted_probs[pos]
        else:
            prediction_handler_logger.warning(
                f"索引越界: 结果索引 {idx} 超出增强概率列表长度 {len(boosted_probs)}"
            )

    if is_new_major_cache is not None:
        for idx in dict_indices:
            r = results[idx]
            uni_value = str(r.get("university", ""))
            major_value = str(r.get("major", ""))
            key = (uni_value, major_value)
            r["is_new_major"] = is_new_major_cache.get(key, False)
    else:
        for idx in dict_indices:
            results[idx]["is_new_major"] = False

    return results


def _get_text_boost_message(experience_details: Dict[str, str]) -> str:
    items = []
    if experience_details.get("research_details"):
        items.append("科研经历")
    if experience_details.get("internship_details"):
        items.append("实习经验")
    if experience_details.get("award_details"):
        items.append("获奖经历")
    if experience_details.get("paper_details"):
        items.append("论文发表")

    if not items:
        return "正在分析您的背景经历"

    return f"正在分析您的{'、'.join(items)}对申请的加成效果"


DOTS_CSS = """
<style>
.dots::after {
    content: '.';
    animation: dots 1.2s steps(3, end) infinite;
}
@keyframes dots {
    0% { content: '.'; }
    33% { content: '..'; }
    66% { content: '...'; }
}
</style>
"""


def _render_animated_message(placeholder, message: str):
    placeholder.markdown(
        f'{DOTS_CSS}<span style="color:#888;font-size:0.85em">{message}<span class="dots"></span></span>',
        unsafe_allow_html=True,
    )


def batch_adjust_results(
    results_list: List[List[Dict[str, Union[float, str]]]],
    probability_adjuster: Optional[ProbabilityAdjuster],
    text_boost_provider: Optional[TextBoostProvider],
    experience_details: Dict[str, str],
    gpa: Optional[float],
    language_score: Optional[float],
    background_university: Optional[str],
) -> List[List[Dict[str, Union[float, str]]]]:
    if not results_list:
        return results_list

    all_combinations: set[Tuple[str, str]] = set()
    for results in results_list:
        if results and isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    uni_value = r.get("university")
                    major_value = r.get("major")
                    if isinstance(uni_value, str) and isinstance(major_value, str):
                        all_combinations.add((uni_value, major_value))

    is_new_major_cache: Dict[Tuple[str, str], bool] = {}
    if all_combinations:
        try:
            for uni, major in all_combinations:
                is_new_major_cache[(uni, major)] = is_new_major(uni, major)
        except Exception as e:
            prediction_handler_logger.warning(f"批量查询新专业失败: {e}")

    placeholder = None
    if text_boost_provider is not None and experience_details:
        placeholder = st.empty()
        message = _get_text_boost_message(experience_details)
        _render_animated_message(placeholder, message)

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

    if placeholder is not None:
        placeholder.empty()

    return adjusted_results_list
