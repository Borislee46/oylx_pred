from functools import lru_cache
from pathlib import Path

from src.utils.schools.config_loader import UNIVERSITY_DIFFICULTY_ORDER


@lru_cache(maxsize=1)
def get_university_difficulty_order(
    config_path: Path,
    default_order: tuple[str, ...],
) -> list[str]:
    return list(UNIVERSITY_DIFFICULTY_ORDER)
