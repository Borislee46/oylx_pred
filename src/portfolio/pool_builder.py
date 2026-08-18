from __future__ import annotations

from typing import Any

from src.utils.numeric import clip_probability_coerce, safe_float


def prediction_results_to_schools(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schools: list[dict[str, Any]] = []
    for row in results:
        uni = row.get("university") or row.get("target_university")
        major = row.get("major") or row.get("target_major")
        if not uni or not major:
            continue
        schools.append(
            {
                "university": str(uni),
                "major": str(major),
                "probability": clip_probability_coerce(row.get("probability")),
                "similarity": safe_float(row.get("similarity"), default=0.0),
            }
        )
    return sorted(schools, key=lambda s: s["probability"], reverse=True)
