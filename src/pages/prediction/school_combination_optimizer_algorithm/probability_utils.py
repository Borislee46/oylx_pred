def calibrate_cross_major_probabilities(
    schools: list[dict],
    background_faculty: str | None,
    major_category_cache: dict[str, str] | None,
) -> list[dict]:
    if not schools or not background_faculty or not major_category_cache:
        return schools

    cross_faculty_multiplier = 0.85
    shrinkage_alpha = 0.5
    prior_p = 0.1

    adjusted_schools: list[dict] = []
    for school in schools:
        university = school.get("university", "")
        major = school.get("major", "")
        prob = float(school.get("probability", 0.0))

        cache_key = f"{university}|{major}"
        target_faculty = major_category_cache.get(cache_key)

        if not target_faculty or target_faculty == background_faculty:
            adjusted_schools.append(school)
            continue

        calibrated_prob = shrinkage_alpha * prob + (1.0 - shrinkage_alpha) * prior_p
        calibrated_prob *= cross_faculty_multiplier

        school_copy = dict(school)
        school_copy["probability"] = max(0.0, min(1.0, calibrated_prob))
        adjusted_schools.append(school_copy)

    return adjusted_schools
