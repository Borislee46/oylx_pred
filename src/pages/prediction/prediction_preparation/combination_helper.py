from typing import Any, Optional

from src.utils.session_manager import SessionManager


def get_user_specified_combinations(
    current_input_data: dict[str, Any],
    all_universities_target: list[str],
    session_manager: SessionManager,
) -> Optional[list[tuple[str, str]]]:
    has_user_specification = session_manager.get(
        "selected_target_majors"
    ) or session_manager.get("selected_major_categories")

    if not has_user_specification:
        return None

    target_unis = current_input_data.get("target_universities")
    target_majors = current_input_data.get("target_majors")

    if not target_majors or not isinstance(target_majors, list):
        return None

    if target_unis and isinstance(target_unis, list):
        unis_to_use = target_unis
    else:
        unis_to_use = all_universities_target

    return [(uni, major) for uni in unis_to_use for major in target_majors]

