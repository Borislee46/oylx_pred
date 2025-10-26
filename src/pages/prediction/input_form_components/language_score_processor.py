from src.pages.prediction.input_form_components.form_config import (
    DEFAULT_LANGUAGE_SCORES,
    LANGUAGE_SCORE_RANGES,
)
from src.utils.school_level_service import get_school_level_service


def apply_overseas_language_boost(school_name: str, language_type: str) -> float:
    school_service = get_school_level_service()
    
    if not school_service.is_overseas_school(school_name):
        return DEFAULT_LANGUAGE_SCORES.get(language_type, 6.5)
    
    base_score = DEFAULT_LANGUAGE_SCORES.get(language_type)
    if base_score is None:
        return 6.5 if language_type == "雅思" else 90
    
    multiplier = school_service.get_language_boost_multiplier(school_name)
    boosted_score = base_score * multiplier
    
    max_score = LANGUAGE_SCORE_RANGES[language_type]["max"]
    return min(boosted_score, max_score)

