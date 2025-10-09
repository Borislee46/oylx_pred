import numpy as np

from src.pages.prediction.school_combination_optimizer_algorithm.major_category_config import (
    RELATED_GROUPS,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer_config import (
    CROSS_MAJOR_CALIBRATION,
)


def calibrate_cross_major_probabilities(
    schools: list[dict],
    background_major_category: str | None,
    major_category_cache: dict[str, str] | None,
    school_cross_major_factor: dict[str, float] | None = None,
    calibration_cfg: dict | None = None,
) -> list[dict]:
    if not schools or not background_major_category or not major_category_cache:
        return schools

    cfg = calibration_cfg or CROSS_MAJOR_CALIBRATION
    prior_p = cfg.get("prior_p", 0.12)
    shrinkage_alpha = cfg.get("shrinkage_alpha", 0.5)
    clip_q = cfg.get("clip_quantile", 0.95)
    same_group_mul = cfg.get("same_group_multiplier", 0.9)
    cross_group_mul = cfg.get("cross_group_multiplier", 0.8)
    strict_mul = cfg.get("strict_multiplier", 0.8)

    def in_same_group(cat_a: str, cat_b: str) -> bool:
        for members in RELATED_GROUPS.values():
            if cat_a in members and cat_b in members:
                return True
        return False

    probs = np.array([s.get("probability", 0.0) for s in schools], dtype=float)
    if len(probs) > 0:
        cap = float(np.quantile(probs, clip_q))
    else:
        cap = 1.0

    adjusted: list[dict] = []
    for s in schools:
        university = s.get("university", "")
        major = s.get("major", "")
        prob = float(s.get("probability", 0.0))
        cache_key = f"{university}|{major}"
        target_cat = major_category_cache.get(cache_key, "")
        if not target_cat or target_cat == background_major_category:
            adjusted.append(s)
            continue

        prob = min(prob, cap)

        prob = shrinkage_alpha * prob + (1.0 - shrinkage_alpha) * prior_p

        mul = (
            same_group_mul
            if in_same_group(background_major_category, target_cat)
            else cross_group_mul
        )
        prob *= mul

        if school_cross_major_factor and university in school_cross_major_factor:
            prob *= float(school_cross_major_factor[university])

        prob *= strict_mul

        s2 = dict(s)
        s2["probability"] = max(0.0, min(1.0, prob))
        adjusted.append(s2)

    return adjusted
