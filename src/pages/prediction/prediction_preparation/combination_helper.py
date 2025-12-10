from typing import Any, Optional


def get_user_specified_combinations(
    current_input_data: dict[str, Any],
    all_universities_target: list[str],
) -> Optional[list[tuple[str, str]]]:
    target_majors = current_input_data.get("target_majors")

    if not target_majors or not isinstance(target_majors, list) or len(target_majors) == 0:
        return None

    target_unis = current_input_data.get("target_universities")

    if target_unis and isinstance(target_unis, list) and len(target_unis) > 0:
        unis_to_use = target_unis
    else:
        unis_to_use = all_universities_target

    return [(uni, major) for uni in unis_to_use for major in target_majors]
