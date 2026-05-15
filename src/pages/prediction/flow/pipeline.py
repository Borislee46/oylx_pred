"""
预测管道核心 — 三段式 pipeline 的编排层。

这是整个预测系统的"心脏"。_execute_prediction_pipeline 将原始输入
经过以下阶段转换为最终概率：

  Pipeline 4 阶段：
  ┌──────────┬─────────────────────────────────────────────────────┐
  │ Phase    │ 做什么                                              │
  ├──────────┼─────────────────────────────────────────────────────┤
  │ prep     │ 输入清洗、指纹校验、相似度缓存加载、GPA/语言提取    │
  │ match    │ 候选组合生成（语义+模糊+用户指定）、特征向量构建    │
  │ infer    │ XGBoost 并行推理 → 原始概率                         │
  │ deliver  │ 调整链应用（GPA/语言/跨专业/跨学部/文本提升）       │
  │          │ → 仲裁器衰减 → 三源合并去重 → 最终结果              │
  └──────────┴─────────────────────────────────────────────────────┘

  两条特殊路径：
  1. Fallback 路径：关键特征缺失（GPA/语言）→ 跳过 XGBoost，
     用 Wilson CI 级联兜底估算录取率
  2. 跨学部确认：handler 层设置 _cross_faculty_confirmed=True，
     pipeline 放宽用户指定专业的过滤条件

  两个入口：
  - run_prediction_pipeline: 无进度回调（测试/后台用）
  - run_prediction_pipeline_with_progress: 带进度回调（生产 UI 用）
"""

import random
from collections.abc import Callable
from typing import Any

import pandas as pd

from src.pages.prediction.config.ui_messages import (
    PIPELINE_MESSAGES,
    PIPELINE_PHASE_MAP,
    format_pipeline_done_progress,
    format_pipeline_empty_progress,
    format_pipeline_prep_progress,
    format_pipeline_refine_progress,
)
from src.pages.prediction.core.utils import get_background_faculty, is_new_major
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.flow.run_prediction import run_single_prediction  # XGBoost 推理 + 初始结果处理
from src.pages.prediction.page_data_loader import (
    cached_get_prediction_model,
    machine_learning_model,
)
from src.pages.prediction.prediction_preparation import validate_and_clean_input  # 输入清洗 + 归一化
from src.pages.prediction.result_modifier import (
    AdjustmentContext,              # 调整链所需的上下文（GPA/语言/学部等20+字段）
    ProbabilityAdjustmentPipeline,  # 调整链编排器（批量应用5层惩罚+文本提升）
)
from src.pages.prediction.result_modifier.admission_cache import (
    get_admitted_combinations_from_dataframe,  # 从历史数据提取已录取组合（用于调整链的 admitted flag）
)
from src.pages.prediction.result_modifier.config import DEFAULT_TEXT_BOOST_CONFIG
from src.pages.prediction.result_modifier.fallback import (
    compute_fallback_probabilities,  # Wilson CI 级联兜底（无模型推理能力时用）
)
from src.pages.prediction.result_modifier.probability_adjuster import (
    ProbabilityAdjuster,  # GPA/语言惩罚计算器（需要 cases_df 统计量）
)
from src.pages.prediction.result_modifier.text_boost_provider import (
    get_text_boost_provider,  # TF-IDF 文本提升模型（单例）
)
from src.pages.prediction.results_handler import combine_and_deduplicate_results
from src.utils.app_data_loader import load_bg_target_similarity_cache
from src.utils.logger import setup_logger
from src.utils.session_manager import PredictionResultModel

prediction_handler_logger = setup_logger("page3", "prediction")

ProgressCallback = Callable[[str], None]


def _execute_prediction_pipeline(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    all_universities_target: list[str],
    all_majors_target: list[str],
    reporter: ProgressReporter,
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
    page_state: machine_learning_model | None = None,
    cached_combinations: list[tuple[str, str]] | None = None,
) -> PredictionResultModel:
    """预测管道主函数 — prep → match → infer → deliver 四阶段。

    输入：来自 handle_form_submission 的归一化表单数据
    输出：PredictionResultModel（含三源结果 + unified 合并结果 + meta）

    两条特殊路径：
    - Fallback：GPA 或语言缺失 → 跳过 XGBoost → Wilson CI 兜底
    - Normal：完整特征 → XGBoost → 调整链 → 合并去重

    cached_combinations 参数：
      当 handler 检测到目标与前次相同 → 跳过候选组合生成（processor.py），
      直接复用前次 unified_results 中的 (u, m) 对，大幅加速重跑。
    """
    # ── Phase: prep — 输入准备 ──────────────────────────
    prediction_model = cached_get_prediction_model(model_name)

    if prediction_model is None:
        return PredictionResultModel(meta={"error": "model_load_failed"})

    if page_state is None:
        page_state = machine_learning_model.resource_loader()
    cases_df = page_state.cases_df

    # 数据版本一致性校验：防止模型和数据不同步
    if cases_df_fingerprint != page_state.cases_df_fingerprint:
        prediction_handler_logger.warning(
            f"案例数据指纹不匹配: 期望 {cases_df_fingerprint}, 实际 {page_state.cases_df_fingerprint}"
        )
        return PredictionResultModel(
            meta={
                "error": "cases_df_fingerprint_mismatch",
                "expected_cases_df_fingerprint": cases_df_fingerprint,
                "actual_cases_df_fingerprint": page_state.cases_df_fingerprint,
            }
        )

    # 输入清洗：validate_and_clean_input 处理缺失值、类型转换、院校名模糊匹配等
    cleaned_input = validate_and_clean_input(input_data)
    current_input_data = {**input_data, **cleaned_input}

    # ── Phase: match — 相似度缓存 + 进度报告 ─────────────
    bg_target_similarity_cache = load_bg_target_similarity_cache()

    # 进度报告：展示当前输入的概览（force=True 确保此条立即发送）
    reporter.emit(
        format_pipeline_prep_progress(
            bg_university=cleaned_input.get("background_university"),
            bg_major=cleaned_input.get("background_major"),
            language_type=cleaned_input.get("language_type"),
            language_score=cleaned_input.get("language_score"),
            target_universities=cleaned_input.get("target_universities"),
            similarity_cache_loaded=bool(bg_target_similarity_cache),
        ),
        force=True,
        phase=PIPELINE_PHASE_MAP["init_engine"],
    )

    num_target_universities = len(cleaned_input.get("target_universities", []))
    cross_faculty_confirmed = input_data.get("_cross_faculty_confirmed", False)

    # 提取调整链所需的核心特征
    gpa = cleaned_input.get("gpa")
    language_score = cleaned_input.get("language_score")
    background_university = cleaned_input.get("background_university")
    background_major = cleaned_input.get("background_major", "")

    # ProbabilityAdjuster 初始化：需要 cases_df 计算 GPA 均值/标准差（z-score）和语言分布
    # GPA 或语言缺失时设为 None → 调整链跳过（fallback 路径用）
    probability_adjuster = (
        ProbabilityAdjuster(
            cases_df if cases_df is not None else pd.DataFrame(),
            data_hash=cases_df_fingerprint,
        )
        if gpa is not None and language_score is not None
        else None
    )

    # ── Phase: infer — XGBoost 推理 ────────────────────
    # run_single_prediction 内部执行：
    #   1. generate_prediction_combinations（候选组合生成）
    #   2. prepare_model_inputs（特征向量构建）
    #   3. PredictionExecutor.execute_parallel（批量 XGBoost 推理）
    #   4. process_prediction_results（初步结果处理：分路、排序）
    sim_results, cross_results, user_specified_results, meta = run_single_prediction(
        current_input_data=current_input_data,
        prediction_model=prediction_model,
        cases_df=cases_df,
        bg_target_similarity_cache=bg_target_similarity_cache,
        expected_features=loaded_feature_names,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        num_target_universities=num_target_universities,
        cross_faculty_confirmed=cross_faculty_confirmed,
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        language_type=cleaned_input.get("language_type"),
        background_university=background_university,
        progress_reporter=reporter,
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
        page_state=page_state,
        cached_combinations=cached_combinations,
    )

    # ── Fallback 路径：关键特征缺失 → Wilson CI 级联兜底 ──
    # 触发条件：GPA 或语言成绩缺失（prepare_model_inputs 返回 missing_inputs 非空）
    # fallback_eligible 和 _fallback_combinations 由 run_single_prediction 内部设置
    if meta and meta.get("fallback_eligible") and meta.get("_fallback_combinations"):
        prediction_handler_logger.info(
            "触发人口统计兜底 | missing=%s combinations=%d",
            meta.get("missing_features"),
            len(meta["_fallback_combinations"]),
        )
        fallback_combinations = meta["_fallback_combinations"]
        # Wilson CI 级联：精确匹配 → 同背景院校 → 同目标组合 → 同目标院校 → 全局
        fallback_results = compute_fallback_probabilities(
            fallback_combinations,
            cases_df,
            background_university or "",
            background_major,
            similarity_scores=bg_target_similarity_cache,
        )
        if fallback_results:
            # Fallback 结果的调整链：仅文本提升（不做 GPA/语言惩罚，因为本身就是缺失的）
            pipeline = ProbabilityAdjustmentPipeline(
                probability_adjuster=None,       # 无 GPA/语言 → 跳过后5层惩罚
                text_boost_provider=(
                    get_text_boost_provider(DEFAULT_TEXT_BOOST_CONFIG)
                    if input_data.get("_has_valid_experience")
                    else None
                ),
            )
            bg_faculty_fb = (
                background_faculty
                or current_input_data.get("faculty")
                or get_background_faculty(background_major, cases_df)
            )
            fb_adj_ctx = AdjustmentContext(
                gpa=gpa,
                language_score=language_score,
                background_university=background_university,
                background_major=background_major,
                background_faculty=bg_faculty_fb,
                internship_count=cleaned_input.get("internship_count", 0),
                user_specified_majors=cleaned_input.get("target_majors", []),
                experience_details=cleaned_input.get("experience_details", {}),
                cases_df=cases_df,
                admitted_combinations=admitted_combinations,
            )
            fallback_results = pipeline.adjust_batch(
                fallback_results, fb_adj_ctx,
                progress_reporter=reporter, batch_tag="兜底估算",
            )
            unified = combine_and_deduplicate_results(fallback_results, [], None)
            # 记录 fallback 层级（用于 UI 通知 _check_and_render_fallback_notice）
            meta["fallback_level"] = min(
                (r.get("_fallback_level", 4) for r in fallback_results),
                default=4,
            )
            reporter.emit(
                format_pipeline_done_progress(),
                force=True,
                phase=PIPELINE_PHASE_MAP["done"],
            )
            return PredictionResultModel(
                similarity_results=fallback_results,
                cross_major_results=[],
                user_specified_results=None,
                unified_results=unified,
                meta=meta,
            )

    # ── Normal 路径：XGBoost 推理已完成，进入调整链 ────
    if meta and meta.get("error"):
        prediction_handler_logger.info(f"预测未生成有效结果: {meta.get('error')}")
        return PredictionResultModel(meta=meta)

    # 录取组合缓存（用于调整链判断某个组合在历史数据中有无录取记录）
    admitted_combos = (
        admitted_combinations
        if admitted_combinations is not None
        else get_admitted_combinations_from_dataframe(cases_df, background_major)
    )

    # 新增专业缓存（用于 UI 标记"新专业"标签）
    all_res = sim_results + cross_results + (user_specified_results or [])
    new_major_cache = {
        (r.get("university"), r.get("major")): is_new_major(r.get("university"), r.get("major"))
        for r in all_res
        if r.get("university") and r.get("major")
    }

    # 背景学部（三级优先级：handler 传入 > 输入数据中的 faculty > 查表计算）
    bg_faculty = (
        background_faculty
        if background_faculty
        else (
            current_input_data.get("faculty") or get_background_faculty(background_major, cases_df)
        )
    )

    # ── Phase: deliver — 调整链 + 合并去重 ──────────────
    route_labels: list[str] = []
    if sim_results:
        route_labels.append("相似专业")
    if cross_results:
        route_labels.append("跨专业")
    if user_specified_results:
        route_labels.append("用户指定")

    reporter.emit(
        format_pipeline_refine_progress(
            route_labels=route_labels,
            bg_faculty=bg_faculty if isinstance(bg_faculty, str) else None,
            soft_background_on=bool(input_data.get("_has_valid_experience")),
        ),
        force=True,
        phase=PIPELINE_PHASE_MAP["initial_filter"],
    )

    # AdjustmentContext 打包调整链所需的全部上下文（20+ 字段）
    adj_ctx = AdjustmentContext(
        gpa=gpa,
        language_score=language_score,
        background_university=background_university,
        background_major=background_major,
        background_faculty=bg_faculty,
        internship_count=cleaned_input.get("internship_count", 0),
        user_specified_majors=cleaned_input.get("target_majors", []),
        experience_details=cleaned_input.get("experience_details", {}),
        cases_df=cases_df,
        admitted_combinations=admitted_combos,
        is_new_major_cache=new_major_cache,
    )

    # 调整链 Pipeline：GPA惩罚 → 语言惩罚 → 跨专业(×0.5) → 跨学部(×0.3)
    #                  → 职业学位(×0.7) → 仲裁器衰减(0.85/layer) → 文本提升(+0~15%)
    pipeline = ProbabilityAdjustmentPipeline(
        probability_adjuster=probability_adjuster,
        text_boost_provider=(
            get_text_boost_provider(DEFAULT_TEXT_BOOST_CONFIG)
            if input_data.get("_has_valid_experience")  # 无有效经历文本 → 跳过文本提升
            else None
        ),
    )

    # 调整链对三路结果分别执行（每路可能有不同的调整参数）
    sim_results = pipeline.adjust_batch(
        sim_results, adj_ctx, progress_reporter=reporter, batch_tag="相似专业"
    )
    cross_results = pipeline.adjust_batch(
        cross_results, adj_ctx, progress_reporter=reporter, batch_tag="跨专业"
    )
    if user_specified_results:
        user_specified_results = pipeline.adjust_batch(
            user_specified_results, adj_ctx, progress_reporter=reporter, batch_tag="用户指定"
        )

    # 三源合并去重：优先级 user_specified > cross > similarity
    unique_results = combine_and_deduplicate_results(
        sim_results, cross_results, user_specified_results
    )

    if not unique_results:
        meta = meta or {}
        meta["error"] = "empty_results"
        empty_msg = PIPELINE_MESSAGES["empty_results"]
        meta.setdefault(
            "user_message",
            random.choice(empty_msg) if isinstance(empty_msg, list) else empty_msg,
        )
        prediction_handler_logger.info("预测结果为空")
        reporter.emit(
            format_pipeline_empty_progress(
                bg_major=background_major,
                target_universities=cleaned_input.get("target_universities"),
            ),
            force=True,
            phase=PIPELINE_PHASE_MAP["empty_results"],
        )
        return PredictionResultModel(meta=meta)

    reporter.emit(
        format_pipeline_done_progress(),
        force=True,
        phase=PIPELINE_PHASE_MAP["done"],
    )

    return PredictionResultModel(
        similarity_results=sim_results,
        cross_major_results=cross_results,
        user_specified_results=user_specified_results,
        unified_results=unique_results,
        meta=meta,
    )


# ── 辅助函数 ──────────────────────────────────────────────
def _prepare_list_args(input_data: dict[str, Any]) -> tuple[list[str], list[str]]:
    """从 input_data 的特殊前缀键提取目标列表。

    _all_universities_target / _all_majors_target 是 handler 层注入的元数据键，
    区别于 normalized 后的 target_universities / target_majors（可能被 normalize 修改）。
    """
    def _to_list(key: str) -> list[str]:
        raw = input_data.get(key)
        return [str(x) for x in raw] if isinstance(raw, list) else []

    return _to_list("_all_universities_target"), _to_list("_all_majors_target")


# ── 公共入口（无进度回调版）─────────────────────────────
def run_prediction_pipeline(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
) -> PredictionResultModel:
    """预测管道入口（无进度回调）。

    用于测试、脚本、批量预测等不需要 UI 进度展示的场景。
    ProgressReporter(None) → 所有进度消息被静默丢弃。
    """
    all_universities_target, all_majors_target = _prepare_list_args(input_data)
    reporter = ProgressReporter(None)

    return _execute_prediction_pipeline(
        input_data=input_data,
        model_name=model_name,
        cases_df_fingerprint=cases_df_fingerprint,
        loaded_feature_names=loaded_feature_names,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        reporter=reporter,
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
    )


# ── 公共入口（带进度回调版）─────────────────────────────
def run_prediction_pipeline_with_progress(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    *,
    progress_cb: ProgressCallback | None = None,
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
    page_state: machine_learning_model | None = None,
    cached_combinations: list[tuple[str, str]] | None = None,
) -> PredictionResultModel:
    """预测管道入口（带进度回调）。

    用于生产 UI（hk.py → handler.py → 此函数）。
    progress_cb 非 None 时，pipeline 各阶段通过 ProgressReporter 发送进度文本，
    最终由 render_thought_bubble_with_wait_pulse 渲染到前端。
    """
    all_universities_target, all_majors_target = _prepare_list_args(input_data)
    reporter = ProgressReporter(progress_cb)

    return _execute_prediction_pipeline(
        input_data=input_data,
        model_name=model_name,
        cases_df_fingerprint=cases_df_fingerprint,
        loaded_feature_names=loaded_feature_names,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        reporter=reporter,
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
        page_state=page_state,
        cached_combinations=cached_combinations,
    )
