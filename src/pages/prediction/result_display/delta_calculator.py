from src.utils.session_manager import SessionManager


class DeltaCalculator:
    @staticmethod
    def calculate_delta(result, prev_prob_map):
        key = (result.get("university"), result.get("major"))
        prev_p = (
            float(prev_prob_map.get(key, 0.0))
            if prev_prob_map and prev_prob_map.get(key) is not None
            else None
        )
        cur_p = float(result.get("probability", 0.0) or 0.0)

        if prev_p is None:
            return ""

        diff_pct = (cur_p - prev_p) * 100.0
        if abs(diff_pct) < 0.05:
            return ""
        elif diff_pct > 0:
            return f"+{diff_pct:.1f}%"
        else:
            return f"{diff_pct:.1f}%"

    @staticmethod
    def should_show_delta(
        target_universities,
        target_majors,
        background_university,
        background_major,
    ):
        session_manager = SessionManager()
        prev_context_key = session_manager.get("previous_context_key")
        prev_prob_map = session_manager.get("previous_prob_map", {})
        form_data_changed = session_manager.get("form_data_changed", False)

        prob_map_to_use = (
            session_manager.get("prev_prev_prob_map", {}) if form_data_changed else prev_prob_map
        )

        target_unis_sorted = tuple(sorted(target_universities)) if target_universities else ()
        target_majs_sorted = tuple(sorted(target_majors)) if target_majors else ()
        cur_context_key = (
            background_university,
            background_major,
            target_unis_sorted,
            target_majs_sorted,
        )

        return (
            isinstance(prev_context_key, tuple)
            and prev_context_key == cur_context_key
            and isinstance(prob_map_to_use, dict)
            and bool(prob_map_to_use)
        ), prob_map_to_use
