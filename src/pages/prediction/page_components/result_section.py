import hashlib
import json

from src.pages.prediction.prediction_result_display import ResultsDisplay
from src.utils.session_manager import SessionManager


def _compute_results_hash(sim_results, cross_results, user_specified_results) -> str:
    def extract_key_fields(results):
        if not results:
            return []
        return [
            (
                r.get("university"),
                r.get("major"),
                round(float(r.get("probability", 0.0) or 0.0), 6),
            )
            for r in results
            if isinstance(r, dict)
        ]

    combined = {
        "sim": extract_key_fields(sim_results),
        "cross": extract_key_fields(cross_results),
        "user": extract_key_fields(user_specified_results),
    }
    content = json.dumps(combined, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()


def display_results_section(
    input_data,
    sim_results,
    cross_results,
    user_specified_results,
    cases_df,
    submitted=True,
):
    if all(x is None for x in [sim_results, cross_results, user_specified_results]):
        return

    session_manager = SessionManager()
    background_university = input_data.get("background_university")
    background_major = input_data.get("background_major")

    results_display = ResultsDisplay(
        top_similarity_results=sim_results,
        top_cross_major_results=cross_results,
        user_specified_results=user_specified_results,
    )

    target_universities = input_data.get("target_universities", [])
    target_majors = input_data.get("target_majors", [])

    results_display.display(
        target_universities,
        target_majors,
        background_university=background_university,
        background_major=background_major,
    )

    current_results_hash = _compute_results_hash(sim_results, cross_results, user_specified_results)
    last_saved_hash = session_manager.get("last_saved_results_hash", "")
    form_data_changed = bool(session_manager.get("form_data_changed", False))

    if current_results_hash and current_results_hash != last_saved_hash and not form_data_changed:
        target_unis_sorted = tuple(sorted(target_universities)) if target_universities else ()
        target_majs_sorted = tuple(sorted(target_majors)) if target_majors else ()
        cur_context_key = (
            background_university,
            background_major,
            target_unis_sorted,
            target_majs_sorted,
        )

        old_prev_map = session_manager.get("previous_prob_map", {})
        current_prob_map: dict = {}
        for result in (user_specified_results or []) + (sim_results or []) + (cross_results or []):
            if isinstance(result, dict):
                key = (result.get("university"), result.get("major"))
                if key[0] and key[1]:
                    p = float(result.get("probability", 0.0) or 0.0)
                    current_prob_map[key] = p
        session_manager.set(
            previous_context_key=cur_context_key,
            prev_prev_prob_map=old_prev_map,
            previous_prob_map=current_prob_map,
            last_saved_results_hash=current_results_hash,
        )
