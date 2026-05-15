"""
表单提交流程编排 — handler.py 是预测系统的"总调度"。

职责：
  handle_form_submission: 从表单数据到预测结果的主流程
  run_prediction_with_guard: 带守卫的预测执行（经历校验 + pipeline 调用 + 异常处理）

完整流程（handle_form_submission）：
  1. 前置校验：缺少背景院校/专业 → 跳过
  2. 跨学部检测：quick_cross_faculty_check → 需确认则弹窗 + 暂存数据
  3. 输入准备：prepare_input_data 归一化 → persist_input_state 持久化
  4. 提交日志：log_first_submission_if_needed（每会话一次）
  5. 缓存复用：与前次目标相同 → 复用 unified_results 加速匹配
  6. 预测执行：run_prediction_with_guard → pipeline + 结果写入 session_state

跨学部确认流的 session_state 协议：
  检测到跨学部 → hk_ui_phase="awaiting_confirm" + pending_prediction_data
  → hk.py 检测到 awaiting_confirm → _dispatch_prediction 处理确认
  → 用户确认后重跑 → handler 检测 cross_faculty_confirmed=True → 跳过检测
"""

import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.pages.prediction.config.ui_messages import PIPELINE_MESSAGES
from src.pages.prediction.core.utils import get_background_faculty
from src.pages.prediction.flow.pipeline import run_prediction_pipeline_with_progress
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.handler_config import (
    DEFAULT_FORM_KEYS,
    DEFAULT_UI_KEYS,
    FormSubmissionContext,
    SessionKeys,
)
from src.pages.prediction.input_form_components.cross_faculty_guard import (
    cross_faculty_confirm_dialog,   # Streamlit dialog：提示跨学部风险 + 确认/取消按钮
    quick_cross_faculty_check,      # 快速检测：背景专业 vs 目标专业的学部是否一致
)
from src.pages.prediction.page_components.submission_logger import (
    log_first_submission_if_needed,
)
from src.pages.prediction.prediction_preparation import prepare_input_data  # 表单数据 → 模型输入归一化
from src.pages.prediction.result_modifier.admission_cache import (
    get_admitted_combinations_from_dataframe,  # 从历史数据中提取已录取的院校-专业组合
)
from src.pages.prediction.result_modifier.experience_text_validator import (
    has_meaningful_experience_text,  # 检测经历文本是否包含有效内容（非空、非默认）
)
from src.pages.prediction.results_handler import reset_prediction_results
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from src.pages.prediction.page_data_loader import machine_learning_model
    from src.utils.session_manager import SessionManager

from src.utils.session_manager import PredictionResultModel

prediction_handler_logger = setup_logger("page3", "prediction")

ProgressCallback = Callable[[str], None]


def _update_progress(progress_cb: ProgressCallback | None, text: str | list[str]) -> None:
    """向进度回调发送消息（支持随机变体避免重复感）。"""
    if progress_cb is not None:
        if isinstance(text, list):
            t = str(random.choice(text) if text else "").strip()
        else:
            t = str(text or "").strip()
        progress_cb(t)


def persist_input_state(
    session_manager: "SessionManager",
    current_input_data: dict,
    session_keys: SessionKeys,
) -> None:
    """将归一化后的输入数据持久化到 session_state。

    此数据用于：
    - display_content 回显（has_predicted=True 时渲染结果区）
    - 下次预测的 previous_input_data 对比
    """
    session_manager.set(
        **{
            session_keys.input_data: current_input_data,
            session_keys.is_school_selection_submit: False,
        }
    )


def run_prediction_with_guard(
    session_manager: "SessionManager",
    page_state: "machine_learning_model",
    current_input_data: dict,
    all_universities_target: list[str],
    all_majors_target: list[str],
    session_keys: SessionKeys,
    progress_cb: ProgressCallback | None = None,
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
    cached_combinations: list[tuple[str, str]] | None = None,
) -> bool:
    """带守卫的预测执行：经历校验 → pipeline 调用 → 结果处理。

    内部步骤：
    1. 构建带元数据的 input_data（_all_universities_target、_cross_faculty_confirmed 等）
    2. 经历文本校验：has_meaningful_experience_text 检测四段经历是否有有效内容
    3. 调用 run_prediction_pipeline_with_progress 执行完整 pipeline
    4. 成功 → 写入 prediction_results + usage_stats
    5. 失败 → 按 error_type 映射用户消息 + 重置状态

    cached_combinations 优化：
      当用户的目标院校与前次完全相同时，复用前次 unified_results 中的
      (university, major) 组合来加速匹配阶段。避免重复的候选池构建。

    Returns:
        bool: True=成功生成结果，False=失败（已设置 user_message）
    """
    # 1. 构建带元数据标记的 input_data（下划线前缀 = pipeline 内部用，不输出给用户）
    input_data_with_lists = current_input_data.copy()
    input_data_with_lists["_all_universities_target"] = all_universities_target
    input_data_with_lists["_all_majors_target"] = all_majors_target
    input_data_with_lists["_cross_faculty_confirmed"] = session_manager.get(
        DEFAULT_UI_KEYS.cross_faculty_confirmed, False
    )

    # 2. 经历文本校验
    experience_details = current_input_data.get("experience_details", {})
    pre_reporter = ProgressReporter(progress_cb)
    has_valid_experience = has_meaningful_experience_text(
        experience_details, progress_reporter=pre_reporter
    )
    input_data_with_lists["_has_valid_experience"] = has_valid_experience

    # 3. 数据版本指纹（用于一致性校验）
    cases_df_fingerprint = page_state.cases_df_fingerprint

    # 4. 执行预测 pipeline
    prediction_result_model = run_prediction_pipeline_with_progress(
        input_data_with_lists,
        "xgboost",
        cases_df_fingerprint,
        page_state.loaded_feature_names,
        progress_cb=progress_cb,
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
        page_state=page_state,
        cached_combinations=cached_combinations,
    )
    if prediction_result_model and prediction_result_model.meta:
        session_manager.set(**prediction_result_model.meta)

    # 5. 有结果 → 写入 session_state
    unified = getattr(prediction_result_model, "unified_results", None)
    if isinstance(unified, list) and len(unified) > 0:
        prediction_handler_logger.info("预测成功 | 结果数=%d", len(unified))
        session_manager.set(
            prediction_results=prediction_result_model,
            **{session_keys.has_predicted: True, session_keys.predict_lock: False},
            fresh_prediction_result=True,
            student_background_chart_visible=True,
        )
        # 用量统计（排除开发者本人）
        _user_info = session_manager.get("user_info", {})
        _user_email = str(_user_info.get("email", "") or "").lower()
        if _user_email:
            try:
                from src.pages.prediction.usage_stats import increment as _incr_usage

                _incr_usage(unified)
            except Exception:
                prediction_handler_logger.warning("usage_stats 写入失败", exc_info=True)
        from src.pages.prediction.input_form_components.form_state import FormStateManager

        FormStateManager.update_form_snapshot_hash_after_prediction(session_manager)
        return True

    # 6. 无结果 → 错误映射 + 重置
    error_type = (
        getattr(prediction_result_model, "meta", {}).get("error", "unknown")
        if prediction_result_model
        else "null_result"
    )
    prediction_handler_logger.warning("预测未生成有效结果 | error=%s", error_type)

    _error_messages = {
        "no_valid_combinations": "当前条件下无可匹配的院校-专业组合。请尝试调整目标院校或专业选择。",
        "model_unavailable": "预测服务暂时不可用，请稍后重试。如反复出现请联系技术支持。",
        "execution_failed": "预测计算失败，请稍后重试。如反复出现请联系技术支持。",
        "missing_features": "关键数据缺失（GPA、语言成绩），无法完成模型预测。",
        "empty_results": "当前条件下无匹配结果，请尝试调整目标范围。",
        "cases_df_fingerprint_mismatch": "数据版本不一致，请刷新页面后重试。",
    }
    _msg = _error_messages.get(error_type, "预测未完成，请稍后重试。如反复出现请联系技术支持。")
    session_manager.set(user_message=_msg)
    reset_prediction_results(session_manager)
    session_manager.set(**{session_keys.predict_lock: False})
    return False


def handle_form_submission(
    ctx: FormSubmissionContext, progress_cb: ProgressCallback | None = None
) -> None:
    """表单提交主流程 — 从表单数据到预测结果的完整编排。

    流程分为 6 个阶段：

    Phase 1: 前置校验
      缺少背景院校或专业 → 设置 user_message + 重置状态 + 返回

    Phase 2: 跨学部检测
      quick_cross_faculty_check 比较背景专业 vs 目标专业的学部
      - 无跨学部 → 继续
      - 有跨学部 + agent 已批准 → 标记 cross_faculty_confirmed + 继续
      - 有跨学部 + 未经确认 → 暂存数据 + 弹窗 + 返回（由 hk.py 处理后续）

    Phase 3: 输入准备
      prepare_input_data 归一化表单数据 → 模型输入格式
      persist_input_state 持久化到 session_state

    Phase 4: 提交日志
      log_first_submission_if_needed（每会话记录一次）

    Phase 5: 缓存复用
      与前次预测目标院校相同 → 复用 unified_results 中的组合加速匹配

    Phase 6: 预测执行
      run_prediction_with_guard → pipeline + 结果写入 session_state
    """
    session_manager = ctx.session_manager
    page_state = ctx.page_state
    input_data_from_form = ctx.input_data_from_form
    session_keys = ctx.session_keys

    session_manager.set(**{session_keys.form_data_changed: False})

    # ── Phase 1: 前置校验 ──
    bg_major = input_data_from_form.get("background_major")
    if not all([input_data_from_form.get("background_university"), bg_major]):
        prediction_handler_logger.info("表单提交缺少背景院校或专业，跳过预测")
        reset_prediction_results(session_manager)
        session_manager.set(
            user_message="缺少背景院校或专业信息，无法进行预测。请填写完整的背景信息后再试。"
        )
        session_manager.delete(session_keys.input_data)
        return

    prediction_handler_logger.info(
        "表单提交 | 院校=%s 专业=%s",
        input_data_from_form.get("background_university", "")[:40],
        bg_major[:40],
    )

    # ── Phase 2: 跨学部检测 ──
    # 延迟计算：只在首次调用时计算 background_faculty 和 admitted_combinations
    if not ctx.background_faculty:
        ctx.background_faculty = get_background_faculty(bg_major, page_state.cases_df)
    if not ctx.admitted_combinations:
        ctx.admitted_combinations = get_admitted_combinations_from_dataframe(
            page_state.cases_df, bg_major
        )

    user_selected_categories = (
        session_manager.get(DEFAULT_FORM_KEYS.selected_major_categories, []) or []
    )
    user_selected_majors = session_manager.get(DEFAULT_FORM_KEYS.selected_target_majors, []) or []

    if bg_major and (user_selected_categories or user_selected_majors):
        _update_progress(progress_cb, PIPELINE_MESSAGES["cross_check"])
        is_cross_faculty, bg_faculty, target_faculties, agent_approved = quick_cross_faculty_check(
            bg_major,
            user_selected_categories,
            user_selected_majors,
            page_state.cases_df,
        )

        if is_cross_faculty:
            if agent_approved:
                # 智能体已自动批准跨学部（如理→工的合理转换）
                prediction_handler_logger.info(
                    "跨学科检测: 智能体已批准 | bg=%s target=%s",
                    bg_faculty,
                    target_faculties,
                )
                session_manager.set(cross_faculty_confirmed=True)
            elif not session_manager.get(DEFAULT_UI_KEYS.cross_faculty_confirmed, False):
                # 需用户确认：暂存数据 + 弹窗 + 返回
                # hk.py 在检测到 awaiting_confirm 后处理 confirm_dialog 的确认/取消
                prediction_handler_logger.warning(
                    "跨学科检测: 需用户确认 | bg=%s target=%s",
                    bg_faculty,
                    target_faculties,
                )
                _update_progress(progress_cb, "检测到跨学科申请跨度较大，需进一步评估风险...")
                session_manager.set(
                    hk_ui_phase="awaiting_confirm",
                    pending_prediction_data={
                        "input_data": input_data_from_form,
                        "all_universities": ctx.all_universities_target,
                        "all_majors": ctx.all_majors_target,
                        "original_form": ctx.original_form_data,
                    },
                )
                cross_faculty_confirm_dialog(session_manager, bg_faculty, target_faculties)
                return

    # ── Phase 3: 输入准备 ──
    session_manager.set(**{session_keys.predict_lock: True})
    session_manager.set(hk_ui_phase="running", hk_last_error=None)

    # 保存前次结果（用于 Delta 对比）
    _prev_model = session_manager.get(DEFAULT_UI_KEYS.prediction_results)
    if isinstance(_prev_model, PredictionResultModel) and _prev_model.unified_results:
        session_manager.set(
            previous_prediction_results=_prev_model,
            previous_input_data=session_manager.get(session_keys.input_data),
        )

    current_input_data = prepare_input_data(input_data_from_form, cases_df=page_state.cases_df)
    persist_input_state(session_manager, current_input_data, session_keys)

    # ── Phase 4: 提交日志（每会话一次）──
    log_first_submission_if_needed(
        session_manager,
        ctx.original_form_data,
        input_data_from_form,
        session_keys.last_submission_logged,
    )

    # ── Phase 5: 缓存复用 ──
    # 目标院校与前次完全相同时 → 复用前次的 unified_results 组合加速匹配
    _prev_model = session_manager.get(DEFAULT_UI_KEYS.prediction_results)
    _cached_combos: list[tuple[str, str]] | None = None
    if isinstance(_prev_model, PredictionResultModel) and _prev_model.unified_results:
        _prev_target_unis = sorted(
            {str(r.get("university", "")) for r in _prev_model.unified_results}
        )
        _curr_target_unis = sorted(str(u) for u in ctx.all_universities_target)
        if _prev_target_unis == _curr_target_unis:
            _cached_combos = [
                (str(r.get("university", "")), str(r.get("major", "")))
                for r in _prev_model.unified_results
            ]

    # ── Phase 6: 预测执行 ──
    run_prediction_with_guard(
        session_manager,
        page_state,
        current_input_data,
        ctx.all_universities_target,
        ctx.all_majors_target,
        session_keys,
        progress_cb=progress_cb,
        background_faculty=ctx.background_faculty,
        admitted_combinations=ctx.admitted_combinations,
        cached_combinations=_cached_combos,
    )
