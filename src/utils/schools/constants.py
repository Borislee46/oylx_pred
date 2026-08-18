from src.utils.config_models import SchoolConstants

_cfg = SchoolConstants.load()

SCHOOL_LEVEL_PRIORITY: dict[str | None, int] = {
    (k if k != "null" else None): v for k, v in _cfg.school_level_priority.items()
}
SCHOOL_LEVEL_SCORES: dict[str | None, float] = {
    (k if k != "null" else None): v for k, v in _cfg.school_level_scores.items()
}
OVERSEAS_SCHOOL_LEVELS: list[str] = _cfg.overseas_school_levels
LANGUAGE_BOOST_MULTIPLIERS: dict[str, float] = _cfg.language_boost_multipliers
