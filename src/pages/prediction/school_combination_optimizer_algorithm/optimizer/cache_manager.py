# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
import hashlib
import json
from collections.abc import Callable
from typing import Any, TypeVar

from src.pages.prediction.school_combination_optimizer_algorithm.utils import LRUCache
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

T = TypeVar("T")


def get_cached_data(
    caches: dict[str, LRUCache],
    cache_type: str,
    key: str,
    calculation_func: Callable[[], T],
) -> T:
    cache = caches[cache_type]
    if cached := cache.get(key):
        if isinstance(cached, (int, float, str, tuple, bool, type(None))):
            return cached
        return cached.copy() if hasattr(cached, "copy") else cached

    result = calculation_func()
    if isinstance(result, (int, float, str, tuple, bool, type(None))):
        cache.put(key, result)
    else:
        cache.put(key, result.copy() if hasattr(result, "copy") else result)
    return result


def build_optimization_input_hash(
    all_schools_data: list[dict[str, Any]],
    background_major: str,
    background_faculty: str | None,
    school_level: str | None,
    gpa: float | None,
) -> str:
    input_data = {
        "background_major": background_major,
        "background_faculty": background_faculty,
        "school_level": school_level,
        "gpa": gpa,
        "schools": sorted(
            [
                {
                    "university": s.get("university", ""),
                    "major": s.get("major", ""),
                    "probability": s.get("probability", 0.0),
                }
                for s in all_schools_data
            ],
            key=lambda x: (x["university"], x["major"]),
        ),
    }

    input_str = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(input_str.encode("utf-8")).hexdigest()


def clear_all_caches(caches: dict[str, LRUCache]) -> None:
    for cache in caches.values():
        cache.clear()
