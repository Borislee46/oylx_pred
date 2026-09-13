import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.adjustment.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.adjustment.experience_text_validator import (
    has_meaningful_experience_text,
)
from src.pages.prediction.core.ui_messages import PIPELINE_MESSAGES
from src.pages.prediction.core.utils import get_background_faculty
from src.pages.prediction.flow import prepare_input_data
from src.pages.prediction.flow.dual_major_pipeline import run_dual_major_pipeline
from src.pages.prediction.flow.pipeline import run_prediction_pipeline_with_progress
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.handler_config import (
    DEFAULT_FORM_KEYS,
    DEFAULT_UI_KEYS,
    FormSubmissionContext,
    SessionKeys,
)
from src.pages.prediction.input_form_components.cross_faculty_guard import (
    cross_faculty_confirm_dialog,
    quick_cross_faculty_check,
)
from src.pages.prediction.results_handler import reset_prediction_results
from src.pages.prediction.ui.submission_logger import (
    log_first_submission_if_needed,
)
from src.utils.analytics import _bump_prediction_id
from src.utils.analytics import track as _track
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce, prob_to_pct

if TYPE_CHECKING:
    from src.pages.prediction.page_data_loader import machine_learning_model
    from src.utils.session_manager import SessionManager

from src.pages.prediction.ui.page_state_machine import HKPagePhase, PageStateMachine
from src.utils.session_manager import PredictionResultModel

prediction_handler_logger = setup_logger("page3", "prediction")

ProgressCallback = Callable[[str], None]


def _update_progress(progress_cb: ProgressCallback | None, text: str | list[str]) -> None:
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
    input_data_with_lists = current_input_data.copy()
    input_data_with_lists["_all_universities_target"] = all_universities_target
    input_data_with_lists["_all_majors_target"] = all_majors_target
    input_data_with_lists["_cross_faculty_confirmed"] = session_manager.get(
        DEFAULT_UI_KEYS.cross_faculty_confirmed, False
    )

    experience_details = current_input_data.get("experience_details", {})
    pre_reporter = ProgressReporter(progress_cb)
    has_valid_experience = has_meaningful_experience_text(
        experience_details, progress_reporter=pre_reporter
    )
    input_data_with_lists["_has_valid_experience"] = has_valid_experience

    cases_df_fingerprint = page_state.cases_df_fingerprint

    if current_input_data.get("background_major_2"):
        _track("dual_major_used", major2=str(current_input_data.get("background_major_2", ""))[:30])
        prediction_result_model = run_dual_major_pipeline(
            input_data_with_lists,
            "xgboost",
            cases_df_fingerprint,
            page_state.loaded_feature_names,
            progress_cb=progress_cb,
            background_faculty=background_faculty,
            admitted_combinations=admitted_combinations,
            page_state=page_state,
        )
    else:
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

    unified = getattr(prediction_result_model, "unified_results", None)
    if isinstance(unified, list) and len(unified) > 0:
        prediction_handler_logger.info("预测成功 | 结果数=%d", len(unified))
        _top3 = [
            f"{r.get('university', '')[:10]}|{prob_to_pct(r.get('probability'))}%"
            for r in unified[:3]
        ]
        _probs = [clip_probability_coerce(r.get("probability")) for r in unified]
        _prob_range = f"{prob_to_pct(min(_probs))}%-{prob_to_pct(max(_probs))}%"
        _flag_count = sum(1 for r in unified if r.get("_business_flags"))
        _track(
            "prediction_complete",
            result_count=len(unified),
            top3=_top3,
            prob_range=_prob_range,
            business_flag_count=_flag_count,
            fallback_used=bool(
                prediction_result_model.meta.get("fallback_level")
                if prediction_result_model.meta
                else False
            ),
        )
        session_manager.set(
            prediction_results=prediction_result_model,
            **{session_keys.has_predicted: True, session_keys.predict_lock: False},
            fresh_prediction_result=True,
            student_background_chart_visible=True,
        )
        from src.pages.prediction.input_form_components.form_state import FormStateManager

        FormStateManager.update_form_snapshot_hash_after_prediction(session_manager)
        return True

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
        "model_load_failed": "预测模型加载失败，请刷新页面后重试。如反复出现请联系技术支持。",
    }
    _msg = _error_messages.get(error_type, "预测未完成，请稍后重试。如反复出现请联系技术支持。")
    session_manager.set(user_message=_msg)
    reset_prediction_results(session_manager)
    session_manager.set(**{session_keys.predict_lock: False})
    return False


def handle_form_submission(
    ctx: FormSubmissionContext, progress_cb: ProgressCallback | None = None
) -> None:
    session_manager = ctx.session_manager
    page_state = ctx.page_state
    input_data_from_form = ctx.input_data_from_form
    session_keys = ctx.session_keys

    session_manager.set(**{session_keys.form_data_changed: False})

    bg_major = input_data_from_form.get("background_major")
    if not all([input_data_from_form.get("background_university"), bg_major]):
        prediction_handler_logger.info("表单提交缺少背景院校或专业，跳过预测")
        reset_prediction_results(session_manager)
        session_manager.set(
            user_message="缺少背景院校或专业信息，无法进行预测。请填写完整的背景信息后再试。"
        )
        session_manager.delete(session_keys.input_data)
        return

    gpa_val = input_data_from_form.get("gpa")
    lang_val = input_data_from_form.get("language_score")
    has_gpa = isinstance(gpa_val, (int, float)) and gpa_val > 0
    has_lang = isinstance(lang_val, (int, float)) and lang_val > 0
    if not (has_gpa and has_lang):
        missing_continuous = []
        if not has_gpa:
            missing_continuous.append("GPA")
        if not has_lang:
            missing_continuous.append("语言成绩")
        prediction_handler_logger.info(
            "表单提交缺少连续变量 %s，跳过预测（不跑 fallback）", missing_continuous
        )
        reset_prediction_results(session_manager)
        session_manager.set(
            user_message=f"缺少{'和'.join(missing_continuous)}，无法进行预测。"
            "这两项是连续变量，缺失时不能用历史均值替代，请补充后再试。"
        )
        session_manager.delete(session_keys.input_data)
        return

    prediction_handler_logger.info(
        "表单提交 | 院校=%s 专业=%s",
        input_data_from_form.get("background_university", "")[:40],
        bg_major[:40],
    )
    _bump_prediction_id()

    bg_uni_raw = input_data_from_form.get("background_university", "")
    if bg_uni_raw and hasattr(page_state.prediction_model, "check_cross_level_blocked"):
        pm = page_state.prediction_model
        level_before = pm.level_override.get(str(bg_uni_raw))
        is_blocked = pm.check_cross_level_blocked(str(bg_uni_raw))
        level_after = pm.level_override.get(str(bg_uni_raw))

        if level_after and level_after != "未知" and level_before != level_after:
            import streamlit as st

            st.toast(
                f"你的院校「{bg_uni_raw}」不在院校库中，AI 推断层次为"
                f"「{level_after}」。如不准确，请手动选择。",
                icon=":material/psychology:",
            )

        if is_blocked:
            resolved_level = pm.level_override.get(
                str(bg_uni_raw)
            ) or pm.school_level_service.get_school_level(str(bg_uni_raw))
            prediction_handler_logger.info(
                "跨 level 代理拒绝预测 | school=%s level=%s",
                str(bg_uni_raw)[:40],
                resolved_level,
            )
            reset_prediction_results(session_manager)
            if resolved_level == "未知":
                msg = (
                    f"你的院校「{bg_uni_raw}」层次无法确定，"
                    "暂无法给出科学可靠的预测。建议联系顾问做人工评估。"
                )
            else:
                msg = (
                    f"你的院校「{bg_uni_raw}」层次为「{resolved_level}」，"
                    "该层次样本不足，暂无法给出科学可靠的预测。"
                    "建议联系顾问做人工评估。"
                )
            session_manager.set(user_message=msg)
            import streamlit as st

            st.toast(msg, icon=":material/gpp_maybe:")
            session_manager.delete(session_keys.input_data)
            return

        known_bg_unis = getattr(page_state, "background_universities", None) or set()
        if bg_uni_raw and known_bg_unis and bg_uni_raw not in known_bg_unis:
            if not (level_after and level_after != "未知" and level_before != level_after):
                import streamlit as st

                st.toast(
                    f"你的院校「{bg_uni_raw}」不在模型训练集中，"
                    "将用同层次院校作为代理，预测结果仅供参考。",
                    icon=":material/info:",
                )

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
            _track(
                "cross_faculty_popup",
                bg_faculty=str(bg_faculty),
                target_faculties=str(target_faculties),
                agent_approved=agent_approved,
            )
            if agent_approved:
                prediction_handler_logger.info(
                    "跨学科检测: 智能体已批准 | bg=%s target=%s",
                    bg_faculty,
                    target_faculties,
                )
                session_manager.set(cross_faculty_confirmed=True)
            elif not session_manager.get(DEFAULT_UI_KEYS.cross_faculty_confirmed, False):
                prediction_handler_logger.warning(
                    "跨学科检测: 需用户确认 | bg=%s target=%s",
                    bg_faculty,
                    target_faculties,
                )
                _update_progress(progress_cb, "检测到跨学科申请跨度较大，需进一步评估风险...")
                sm = PageStateMachine(session_manager)
                sm.transition(HKPagePhase.AWAITING_CONFIRM)
                session_manager.set(
                    pending_prediction_data={
                        "input_data": input_data_from_form,
                        "all_universities": ctx.all_universities_target,
                        "all_majors": ctx.all_majors_target,
                        "original_form": ctx.original_form_data,
                    },
                )
                cross_faculty_confirm_dialog(session_manager, bg_faculty, target_faculties)
                return

    session_manager.set(**{session_keys.predict_lock: True})
    sm = PageStateMachine(session_manager)
    sm.transition(HKPagePhase.RUNNING)
    session_manager.set(hk_last_error=None)

    _prev_model = session_manager.get(DEFAULT_UI_KEYS.prediction_results)
    if isinstance(_prev_model, PredictionResultModel) and _prev_model.unified_results:
        session_manager.set(
            previous_prediction_results=_prev_model,
            previous_input_data=session_manager.get(session_keys.input_data),
        )

    current_input_data = prepare_input_data(input_data_from_form, cases_df=page_state.cases_df)
    persist_input_state(session_manager, current_input_data, session_keys)

    log_first_submission_if_needed(
        session_manager,
        ctx.original_form_data,
        input_data_from_form,
        session_keys.last_submission_logged,
    )

    _cached_combos: list[tuple[str, str]] | None = None
    if isinstance(_prev_model, PredictionResultModel) and _prev_model.unified_results:
        _prev_input = session_manager.get("previous_input_data") or {}
        _prev_target_unis = sorted(str(u) for u in (_prev_input.get("target_universities") or []))
        _curr_target_unis = sorted(str(u) for u in ctx.all_universities_target)
        if _prev_target_unis and _prev_target_unis == _curr_target_unis:
            _cached_combos = [
                (str(r.get("university", "")), str(r.get("major", "")))
                for r in _prev_model.unified_results
            ]

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
    if not session_manager.get(session_keys.has_predicted, False):
        sm = PageStateMachine(session_manager)
        sm.transition(HKPagePhase.ERROR)
