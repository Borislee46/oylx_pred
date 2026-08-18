from src.pages.prediction.handler_config import DEFAULT_UI_KEYS
from src.utils.numeric import clip_probability_coerce
from src.utils.session_manager import PredictionResultModel, SessionManager


def reset_prediction_results(session_manager: SessionManager):
    session_manager.set(
        has_predicted=False,
        prediction_results=PredictionResultModel(),
        prediction_submit_lock=False,
        processing_lock=False,
        lock_start_time=0,
    )
    session_manager.prune_states(["pending_", "last_"])
    from src.pages.prediction.result_display.portfolio_service import (
        clear_portfolio_admit_cache,
    )

    clear_portfolio_admit_cache()


def clear_pending_prediction_state(
    session_manager: SessionManager,
    *,
    reset_cross_faculty_confirmed: bool = False,
    reset_cross_faculty_cancelled: bool = False,
):
    updates = {
        DEFAULT_UI_KEYS.pending_cross_faculty_prediction: False,
        DEFAULT_UI_KEYS.pending_prediction_data: None,
    }
    if reset_cross_faculty_confirmed:
        updates[DEFAULT_UI_KEYS.cross_faculty_confirmed] = False
    if reset_cross_faculty_cancelled:
        updates[DEFAULT_UI_KEYS.cross_faculty_cancelled] = False
    session_manager.set(**updates)


def combine_and_deduplicate_results_with_sources(
    sim_results, cross_results, user_specified_results
):
    """合并去重并返回每个 key 的代表条目（原始 dict 引用）。

    Returns:
        (stripped_results, representatives)
        stripped_results: 与 combine_and_deduplicate_results 一致的展示列表（剥离 _ 前缀）。
        representatives: key -> 实际进入 unified 的那条原始 result。
        供 L6 回写使用，避免把同一 key 的概率/step 错误覆盖到非代表条目。
    """
    sources = [
        (sim_results, "similarity", 1, None),
        (cross_results, "cross_major", 2, lambda r: r.get("admitted", 0) == 1),
        (user_specified_results, "user_specified", 3, None),
    ]

    unique_results = {}
    representatives: dict = {}

    for results, source, priority, filter_fn in sources:
        for result in results or []:
            if filter_fn and not filter_fn(result):
                continue

            key = (result.get("university"), result.get("major"))
            new_prob = clip_probability_coerce(result.get("probability"))

            existing = unique_results.get(key)
            if not existing:
                unique_results[key] = {**result, "_source": source, "_priority": priority}
                representatives[key] = result
            else:
                existing_priority = existing.get("_priority", 0)
                existing_prob = clip_probability_coerce(existing.get("probability"))

                if priority > existing_priority or (
                    priority == existing_priority and new_prob > existing_prob
                ):
                    unique_results[key] = {**result, "_source": source, "_priority": priority}
                    representatives[key] = result

    stripped_results = [
        {k: v for k, v in res.items() if not k.startswith("_")} for res in unique_results.values()
    ]
    return stripped_results, representatives


def combine_and_deduplicate_results(sim_results, cross_results, user_specified_results):
    stripped_results, _ = combine_and_deduplicate_results_with_sources(
        sim_results, cross_results, user_specified_results
    )
    return stripped_results


def initialize_session_states(session_manager: SessionManager):
    if session_manager.get(DEFAULT_UI_KEYS.app_initialized):
        return

    session_manager.set(
        has_predicted=False,
        is_school_selection_submit=False,
        processing_lock=False,
        prediction_submit_lock=False,
        input_data=None,
        cross_faculty_confirmed=False,
        cross_faculty_cancelled=False,
        pending_cross_faculty_prediction=False,
        pending_prediction_data=None,
        hk_run_id=None,
        hk_last_error=None,
        hk_view_mode="sales",
        app_initialized=True,
    )
    from src.pages.prediction.ui.page_state_machine import HKPagePhase, PageStateMachine

    sm = PageStateMachine(session_manager)
    sm.transition(HKPagePhase.IDLE)
