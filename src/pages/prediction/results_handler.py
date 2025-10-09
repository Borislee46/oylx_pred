from src.pages.prediction.prediction_utils import is_new_major
from src.utils.session_manager import PredictionResultModel, SessionManager


def reset_prediction_results(session_manager: SessionManager):
    session_manager.set(
        has_predicted=False,
        prediction_results=PredictionResultModel(),
        prediction_submit_lock=False,
    )


def combine_and_deduplicate_results(sim_results, cross_results, user_specified_results):
    unique_results = {}

    def _add_or_update_result(result: dict, source: str, priority: int):
        key = (result.get("university"), result.get("major"))
        if not key[0] or not key[1]:
            return

        result_with_meta = result.copy()
        result_with_meta["_source"] = source
        result_with_meta["_priority"] = priority

        if key not in unique_results:
            unique_results[key] = result_with_meta
        else:
            existing = unique_results[key]
            existing_priority = existing.get("_priority", 0)
            existing_prob = existing.get("probability", 0.0) or 0.0
            new_prob = result.get("probability", 0.0) or 0.0

            should_replace = False
            if priority > existing_priority:
                should_replace = True
            elif priority == existing_priority and new_prob > existing_prob:
                should_replace = True

            if should_replace:
                unique_results[key] = result_with_meta

    if sim_results:
        for res in sim_results:
            _add_or_update_result(res, "similarity", priority=1)

    if cross_results:
        for res in cross_results:
            if res.get("admitted", 0) == 1:
                _add_or_update_result(res, "cross_major", priority=2)

    if user_specified_results:
        for res in user_specified_results:
            _add_or_update_result(res, "user_specified", priority=3)

    final_results = []
    for result in unique_results.values():
        result.pop("_priority", None)
        final_results.append(result)

    return final_results


def add_new_major_marks_to_results(results):
    if not results:
        return results

    marked_results = []
    is_new_cache: dict[tuple[str, str], bool] = {}

    for result in results:
        if isinstance(result, dict):
            result_copy = result.copy()
            university = result_copy.get("university")
            major = result_copy.get("major")
            if university and major:
                cache_key = (university, major)
                is_new = is_new_cache.get(cache_key)
                if is_new is None:
                    is_new = is_new_major(university, major)
                    is_new_cache[cache_key] = is_new
                result_copy["is_new_major"] = is_new
            else:
                result_copy["is_new_major"] = False
            marked_results.append(result_copy)
        else:
            marked_results.append(result)

    return marked_results


def initialize_session_states(session_manager: SessionManager):
    if session_manager.get("app_initialized") is None:
        session_manager.set(
            has_predicted=False,
            is_school_selection_submit=False,
            processing_lock=False,
            prediction_submit_lock=False,
            input_data=None,
            app_initialized=True,
        )
