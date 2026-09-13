from src.utils.schools.config_loader import UNIVERSITY_DISPLAY_ORDER

UNIVERSITY_SORT_ORDER: list[str] = list(UNIVERSITY_DISPLAY_ORDER)
UNIVERSITY_ORDER_MAP: dict[str, int] = {name: i for i, name in enumerate(UNIVERSITY_SORT_ORDER)}
