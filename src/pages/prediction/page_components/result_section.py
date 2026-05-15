"""
预测结果展示区：表格 + 竞争力面板 + 增量对比 + 交互日志。

职责：
  - display_results_section: 编排结果展示流程
  - _tag_results: 为三个来源的结果加中文标签
  - _compute_results_hash: 计算结果 MD5 用于变更检测
  - DeltaCalculator: 判断是否展示"相比上次"的概率变化
"""

import hashlib
import json
from typing import Any

import pandas as pd

from src.pages.prediction.handler_config import DEFAULT_SESSION_KEYS, DEFAULT_UI_KEYS
from src.pages.prediction.result_display import ResultsDisplay  # 结果表格渲染器
from src.pages.prediction.result_display.competitiveness import render_competitiveness_panel  # 竞争力雷达图
from src.pages.prediction.result_display.delta_calculator import DeltaCalculator  # 增量对比计算
from src.utils import log_interaction_event
from src.utils.session_manager import SessionManager

PROBABILITY_PRECISION = 6  # 概率精度（小数点后 6 位）


def _normalize_target_list(v) -> list[str]:
    """归一化目标列表为 list[str]。

    session_state 中可能存储了逗号拼接的字符串（from 日志）或 list（from multiselect）。
    统一转换为 list[str]。
    """
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [x.strip() for x in v.replace("，", ",").split(",") if x.strip()]
    return []


def _tag_results(
    sim: list[dict] | None,
    cross: list[dict] | None,
    user: list[dict] | None,
) -> list[dict]:
    """为三个来源的结果打中文标签（用于竞争力面板展示来源）。"""
    tagged: list[dict] = []
    for r in sim or []:
        tagged.append({**r, "_source": "相似专业"})
    for r in cross or []:
        tagged.append({**r, "_source": "潜力跨专业"})
    for r in user or []:
        tagged.append({**r, "_source": "指定专业"})
    return tagged


def _compute_results_hash(
    sim_results: list[dict[str, Any]] | None,
    cross_results: list[dict[str, Any]] | None,
    user_specified_results: list[dict[str, Any]] | None,
) -> str:
    """计算三源结果的 MD5 哈希（用于检测结果是否与上次相同）。

    相同结果 → 相同哈希 → 不重复记录交互日志 → 避免日志膨胀。
    """
    def _extract(res):
        return [
            (
                str(r.get("university")),
                str(r.get("major")),
                round(float(r.get("probability", 0)), 4),
            )
            for r in (res or [])
            if isinstance(r, dict) and r.get("university")
        ]

    combined = {
        "s": _extract(sim_results),
        "c": _extract(cross_results),
        "u": _extract(user_specified_results),
    }
    return hashlib.md5(json.dumps(combined, sort_keys=True).encode()).hexdigest()


def display_results_section(
    input_data: dict[str, Any],
    sim_results: list[dict[str, Any]] | None,
    cross_results: list[dict[str, Any]] | None,
    user_specified_results: list[dict[str, Any]] | None,
    cases_df: pd.DataFrame,
    submitted: bool = True,
) -> None:
    """渲染完整的结果展示区。

    组件顺序：
    1. ResultsDisplay — 概率表格（含相似专业/跨专业/用户指定三个 tab）
    2. render_competitiveness_panel — 竞争力分析面板（雷达图 + 百分位）
    3. Delta 对比 — 如果用户之前预测过且目标重叠，显示概率变化
    4. 交互日志 — 首次展示结果时记录（相同结果不重复记录）
    """
    if not any([sim_results, cross_results, user_specified_results]):
        return

    all_results = _tag_results(sim_results, cross_results, user_specified_results)

    # ── Delta 对比：检测是否有上一次预测结果可对比 ──
    session_manager = SessionManager()
    prev_model = session_manager.get(DEFAULT_UI_KEYS.previous_prediction_results, None)
    prev_results_list = prev_model.unified_results if prev_model is not None else None

    current_unis = _normalize_target_list(input_data.get("target_universities", []))
    current_majors = _normalize_target_list(input_data.get("target_majors", []))

    has_prev, prev_prob_map, has_overlap = DeltaCalculator.should_show_delta(
        current_unis,
        current_majors,
        input_data.get("background_university", ""),
        input_data.get("background_major", ""),
        prev_results_list,
        session_manager.get(DEFAULT_UI_KEYS.previous_input_data, None),
    )

    show_delta = has_prev and has_overlap

    # ── 渲染结果表格 ──
    ResultsDisplay(
        top_similarity_results=sim_results,
        top_cross_major_results=cross_results,
        user_specified_results=user_specified_results,
        prev_prob_map=prev_prob_map if show_delta else None,
        delta_calculator=DeltaCalculator() if show_delta else None,
        cases_df=cases_df,
    ).display()

    # ── 渲染竞争力面板 ──
    render_competitiveness_panel(all_results, input_data, cases_df)

    # ── 变更检测 + 日志记录 ──
    current_hash = _compute_results_hash(sim_results, cross_results, user_specified_results)

    # 条件：结果变更 + 非表单变更路径 → 记录交互日志
    if current_hash != session_manager.get(
        DEFAULT_UI_KEYS.last_saved_results_hash, ""
    ) and not session_manager.get(DEFAULT_SESSION_KEYS.form_data_changed, False):
        session_manager.set(last_saved_results_hash=current_hash)
        log_interaction_event(
            "prediction_results",
            {
                "results_hash": current_hash,
                "similarity_count": len(sim_results or []),
                "cross_count": len(cross_results or []),
                "user_specified_count": len(user_specified_results or []),
                "best_probability": _best_probability(all_results),
                "target_universities": len(input_data.get("target_universities", [])),
                "target_majors": len(input_data.get("target_majors", [])),
            },
        )


def _best_probability(results: list[dict[str, Any]]) -> float:
    """返回结果列表中最高录取概率（用于日志摘要）。"""
    if not results:
        return 0.0
    return round(max(float(r.get("probability", 0) or 0) for r in results), PROBABILITY_PRECISION)
