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
    RESULT_SOURCES = [
        (sim_results, "similarity", 1),
        (
            cross_results,
            "cross_major",
            2,
            lambda res: res.get("admitted", 0) == 1,
        ),
        (user_specified_results, "user_specified", 3),
    ]

    unique_results = {}

    def _get_result_key(result: dict):
        return (result.get("university"), result.get("major"))

    def _should_replace_existing(existing: dict, new_priority: int, new_prob: float):
        existing_priority = existing.get("_priority", 0)
        existing_prob = existing.get("probability", 0.0) or 0.0

        return new_priority > existing_priority or (
            new_priority == existing_priority and new_prob > existing_prob
        )

    def _process_result_source(results, source, priority, filter_condition=None):
        if not results:
            return

        for result in results:
            if filter_condition and not filter_condition(result):
                continue

            key = _get_result_key(result)
            if not all(key):
                continue

            result_with_meta = result.copy()
            result_with_meta.update({"_source": source, "_priority": priority})
            new_prob = result.get("probability", 0.0) or 0.0

            if key not in unique_results:
                unique_results[key] = result_with_meta
            else:
                existing = unique_results[key]
                if _should_replace_existing(existing, priority, new_prob):
                    unique_results[key] = result_with_meta

    for source_config in RESULT_SOURCES:
        _process_result_source(*source_config)

    return [
        {k: v for k, v in result.items() if not k.startswith("_")}
        for result in unique_results.values()
    ]


def initialize_session_states(session_manager: SessionManager):
    if session_manager.get("app_initialized") is not None:
        return

    initial_states = {
        "has_predicted": False,
        "is_school_selection_submit": False,
        "processing_lock": False,
        "prediction_submit_lock": False,
        "input_data": None,
        "cross_faculty_confirmed": False,
        "cross_faculty_cancelled": False,
        "pending_cross_faculty_prediction": False,
        "pending_prediction_data": None,
        "hk_ui_phase": "idle",
        "hk_run_id": None,
        "hk_last_error": None,
        "app_initialized": True,
    }

    session_manager.set(**initial_states)
