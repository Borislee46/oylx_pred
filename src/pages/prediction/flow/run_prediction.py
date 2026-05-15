"""
单次预测执行器 — 候选组合生成 → 特征构建 → XGBoost 推理 → 结果处理。

这是 predict 链路的最核心执行层。被 pipeline.py 的 _execute_prediction_pipeline 调用。

流程（5 步）：
  1. 候选组合生成：generate_prediction_combinations（语义匹配 + 模糊匹配 + 用户指定）
    或复用 cached_combinations（快速路径）
  2. 特征向量构建：prepare_model_inputs（归一化特征 → 模型格式）
  3. 批量推理：PredictionExecutor.execute_parallel（ThreadPoolExecutor 并行 XGBoost）
  4. Fallback 检测：特征缺失 → 标记 fallback_eligible，pipeline 上层处理
  5. 结果处理：process_prediction_results（分路、排序、相似度调整、学部过滤）
"""

from typing import Any

import pandas as pd

from src.pages.prediction.config.ui_messages import (
    PIPELINE_PHASE_MAP,
    format_pipeline_compute_progress,
)
from src.pages.prediction.core.types import PredictionInput
from src.pages.prediction.core.utils import get_background_faculty
from src.pages.prediction.flow.processor import (
    generate_prediction_combinations,  # 候选池构建（语义+模糊+用户指定三路）
    process_prediction_results,        # 结果处理（分路、排序、相似度调整、Agent 排序）
)
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.modeling.model import PredictionModel
from src.pages.prediction.page_data_loader import machine_learning_model
from src.pages.prediction.prediction_execution import PredictionExecutor  # 并行推理执行器
from src.pages.prediction.prediction_preparation import (
    get_user_specified_combinations,  # 从用户输入解析指定院校-专业组合
    prepare_model_inputs,             # 归一化 → 模型特征向量
)
from src.utils.logger import setup_logger

prediction_runner_logger = setup_logger("page3", "prediction")


def run_single_prediction(
    current_input_data: dict[str, Any],
    prediction_model: PredictionModel,
    cases_df: pd.DataFrame,
    bg_target_similarity_cache: dict[str, float],
    expected_features: list[str],
    all_universities_target: list[str],
    all_majors_target: list[str],
    num_target_universities: int,
    cross_faculty_confirmed: bool = False,
    probability_adjuster: Any | None = None,
    gpa: float | None = None,
    language_score: float | None = None,
    language_type: str | None = None,
    background_university: str | None = None,
    progress_reporter: ProgressReporter | None = None,
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
    page_state: machine_learning_model | None = None,
    cached_combinations: list[tuple[str, str]] | None = None,
) -> tuple[
    list[dict[str, float | str]],       # similarity_results
    list[dict[str, float | str]],       # cross_major_results
    list[dict[str, float | str]] | None,  # user_specified_results
    dict[str, Any] | None,              # meta
]:
    """执行单次预测：候选生成 → 特征构建 → 推理 → 结果处理。

    Returns:
        (similarity, cross_major, user_specified, meta)

    meta 的特殊键：
    - error: 错误类型标识（no_valid_combinations/model_unavailable/execution_failed）
    - fallback_eligible: 是否触发 fallback（特征缺失）
    - _fallback_combinations: fallback 时使用的候选组合（传给 pipeline 上层）
    """
    prediction_input: PredictionInput = current_input_data

    bg_major = prediction_input.get("background_major", "")
    bg_major_orig = str(current_input_data.get("background_major_original") or bg_major)

    # ── Step 1: 候选组合生成（或复用缓存）────────────────
    if cached_combinations:
        # 快速路径：跳过 processor.py 的候选池构建（语义+模糊匹配）
        # 用于目标院校与前次完全相同的重跑场景
        combinations = cached_combinations
        meta = {
            "combination_count": len(combinations),
            "cached": True,
            "progress_hints": {
                "target_unis": list({u for u, _ in combinations}),
                "target_majors": list({m for _, m in combinations}),
                "user_locked_majors": True,
            },
        }
    else:
        # 正常路径：语义匹配 + 模糊匹配 + 用户指定 → 候选池
        combinations, meta = generate_prediction_combinations(
            input_data=prediction_input,
            all_universities_target=all_universities_target,
            all_majors_target=all_majors_target,
            bg_target_similarity_cache=bg_target_similarity_cache,
            background_major_original=bg_major_orig,
        )

    meta = meta or {}
    if not combinations:
        prediction_runner_logger.warning("有效组合为空：请检查候选池或筛选条件。")
        meta["error"] = "no_valid_combinations"
        return [], [], None, meta

    # 进度报告：展示推理阶段概览
    if progress_reporter is not None:
        hints = meta.get("progress_hints") or {}
        progress_reporter.emit(
            format_pipeline_compute_progress(combinations, hints),
            force=True,
            phase=PIPELINE_PHASE_MAP["running_calc"],
        )

    # ── Step 2: 特征向量构建 ────────────────────────────
    model_input_features, missing_inputs = prepare_model_inputs(
        current_input_data, expected_features
    )

    if prediction_model is None:
        meta["error"] = "model_unavailable"
        return [], [], None, meta

    # ── Step 2.5: Fallback 检测 ──────────────────────────
    # 关键特征缺失（GPA/语言）→ 标记 fallback_eligible
    # 不在此层处理 fallback，返回让 pipeline 上层统一处理
    if missing_inputs:
        meta["fallback_eligible"] = True
        meta["missing_features"] = missing_inputs
        meta["_fallback_combinations"] = combinations
        return [], [], None, meta

    # ── Step 3: 批量 XGBoost 推理（并行）─────────────────
    # PredictionExecutor 内部使用 ThreadPoolExecutor，
    # 每个 (u, m) 组合构造一行特征 → XGBoost.predict_proba → Platt 校准概率
    all_prediction_outputs = PredictionExecutor(len(combinations)).execute_parallel(
        prediction_model, combinations, model_input_features, expected_features
    )

    if not all_prediction_outputs:
        meta["error"] = "execution_failed"
        return [], [], None, meta

    # ── Step 4: 用户指定组合解析 ─────────────────────────
    user_specified_combinations = get_user_specified_combinations(
        current_input_data, all_universities_target
    )

    # 背景学部（三级优先级）
    bg_faculty = background_faculty or current_input_data.get("faculty")
    if bg_faculty is None:
        bg_faculty = get_background_faculty(bg_major, cases_df)

    if page_state is None:
        page_state = machine_learning_model.resource_loader()

    # ── Step 5: 结果处理 ─────────────────────────────────
    # process_prediction_results 完成：
    #   - 按目标院校分路（similarity/cross_major/user_specified）
    #   - 初始相似度调整（adjust_similarity_results_with_agent）
    #   - 学部过滤（get_allowed_target_faculties）
    #   - 排序 + 截断
    results = process_prediction_results(
        results=all_prediction_outputs,
        background_major=bg_major,
        background_major_original=bg_major_orig,
        bg_target_similarity_cache=bg_target_similarity_cache,
        num_target_universities=num_target_universities,
        cases_df=cases_df,
        user_specified_combinations=user_specified_combinations,
        background_faculty=bg_faculty if isinstance(bg_faculty, str) else None,
        allow_degraded_user_specified=cross_faculty_confirmed,  # 跨学部确认后放宽过滤
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        language_type=language_type or prediction_input.get("language_type"),
        background_university=background_university,
        progress_reporter=progress_reporter,
        agent=None,
        admitted_combinations=admitted_combinations,
    )

    return (*results, meta)
