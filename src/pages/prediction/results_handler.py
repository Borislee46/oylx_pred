from src.utils.session_manager import PredictionResultModel, SessionManager


def reset_prediction_results(session_manager: SessionManager):
    session_manager.set(
        has_predicted=False,
        prediction_results=PredictionResultModel(),
        prediction_submit_lock=False,
        processing_lock=False,
        lock_start_time=0,
    )


def combine_and_deduplicate_results(sim_results, cross_results, user_specified_results):
    sources = [
        (sim_results, "similarity", 1, None),
        (cross_results, "cross_major", 2, lambda r: r.get("admitted", 0) == 1),
        (user_specified_results, "user_specified", 3, None),
    ]

    unique_results = {}

    for results, source, priority, filter_fn in sources:
        for result in results or []:
            if filter_fn and not filter_fn(result):
                continue

            key = (result.get("university"), result.get("major"))
            new_prob = result.get("probability", 0.0) or 0.0

            existing = unique_results.get(key)
            if not existing:
                unique_results[key] = {**result, "_source": source, "_priority": priority}
            else:
                existing_priority = existing.get("_priority", 0)
                existing_prob = existing.get("probability", 0.0) or 0.0

                if priority > existing_priority or (
                    priority == existing_priority and new_prob > existing_prob
                ):
                    unique_results[key] = {**result, "_source": source, "_priority": priority}

    return [
        {k: v for k, v in res.items() if not k.startswith("_")} for res in unique_results.values()
    ]


def initialize_session_states(session_manager: SessionManager):
    if session_manager.get("app_initialized"):
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
        hk_ui_phase="idle",
        hk_run_id=None,
        hk_last_error=None,
        app_initialized=True,
    )
