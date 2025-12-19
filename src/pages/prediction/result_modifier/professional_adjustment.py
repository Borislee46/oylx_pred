from typing import Any

from src.pages.prediction.result_modifier.config import (
    PROFESSIONAL_MAJORS_LOWER,
    PROFESSIONAL_REDUCTION_FACTOR,
    PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR,
)
from src.pages.prediction.result_modifier.utils import clip_basic


def adjust_for_professional_majors(
    results: list[dict[str, Any]],
    internship_count: int,
    user_specified_majors: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not results:
        return []

    if internship_count > 0:
        return results

    adjusted_results = []
    user_majors_lower = [m.lower() for m in user_specified_majors] if user_specified_majors else []

    for result in results:
        target_major = result.get("major")
        if not target_major:
            adjusted_results.append(result)
            continue

        target_major_lower = target_major.lower()
        is_professional = any(
            prof_major in target_major_lower for prof_major in PROFESSIONAL_MAJORS_LOWER
        )

        if not is_professional:
            adjusted_results.append(result)
            continue

        is_user_specified = any(
            spec_major in target_major_lower for spec_major in user_majors_lower
        )

        result_copy = result.copy()
        p = float(result_copy.get("probability", 0.0) or 0.0)
        factor = (
            PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR
            if is_user_specified
            else PROFESSIONAL_REDUCTION_FACTOR
        )
        p = clip_basic(p * factor)
        result_copy["probability"] = p
        adjusted_results.append(result_copy)

    return adjusted_results
