from collections import OrderedDict
from functools import lru_cache
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
import pandas as pd
from pymoo.util.ref_dirs import get_reference_directions

from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    ADAPTIVE_THRESHOLD_PERCENTILES,
    SCHOOL_CATEGORY_THRESHOLDS,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def clip_probability(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_school_name(name: str | None) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    s = s.replace("\u00a0", "").replace("\u3000", "")
    s = "".join(ch for ch in s if not ch.isspace())
    s = s.replace("（", "(").replace("）", ")")
    return s


def calibrate_cross_major_probabilities(
    schools: list[dict],
    background_faculty: str | None,
) -> list[dict]:
    if not schools or not background_faculty:
        return schools

    cross_faculty_multiplier = 0.85
    shrinkage_alpha = 0.5
    prior_p = 0.1

    def calibrate_school(school: dict) -> dict:
        prob = float(school.get("probability", 0.0))
        target_faculty = school.get("faculty", "")

        if not target_faculty or target_faculty == background_faculty:
            return school

        calibrated_prob = (
            shrinkage_alpha * prob + (1.0 - shrinkage_alpha) * prior_p
        ) * cross_faculty_multiplier
        return {**school, "probability": max(0.0, min(1.0, calibrated_prob))}

    return [calibrate_school(school) for school in schools]


K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int = 256):
        if capacity <= 0:
            raise ValueError("LRU 缓存容量必须为正数")
        self._capacity = capacity
        self._data: MutableMapping[K, V] = OrderedDict()

    def get(self, key: K) -> Optional[V]:
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.pop(key)
        elif len(self._data) >= self._capacity:
            self._data.popitem(last=False)
        self._data[key] = value

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> Iterator[Tuple[K, V]]:
        return iter(self._data.items())


def build_school_key(university: str, major: str) -> str:
    return f"{university}|{major}"


def _get_school_keys(schools: Iterable[dict[str, str]]) -> list[str]:
    return sorted(build_school_key(s.get("university", ""), s.get("major", "")) for s in schools)


def build_selection_key(background_major: str, schools: Iterable[dict[str, str]]) -> str:
    school_keys = _get_school_keys(schools)
    return f"{background_major}|{'|'.join(school_keys)}"


def build_school_set_key(schools: Iterable[dict[str, str]]) -> str:
    school_keys = _get_school_keys(schools)
    return "|".join(school_keys)


def build_major_category_cache(details_df: pd.DataFrame | None) -> Dict[str, str]:
    if details_df is None or details_df.empty:
        return {}

    required_cols = ["学校", "专业英文名称", "专业大类"]
    if not all(col in details_df.columns for col in required_cols):
        return {}

    df_clean = details_df[required_cols].dropna()
    df_clean = df_clean[df_clean[required_cols].ne("").all(axis=1)]

    keys = df_clean["学校"] + "|" + df_clean["专业英文名称"]
    return dict(zip(keys, df_clean["专业大类"]))


def build_new_major_cache(all_schools_data: list[dict[str, Any]]) -> Dict[str, Any]:
    return {
        f"{school['university']}|{school['major']}": school.get("is_new_major", False)
        for school in all_schools_data
        if school.get("university") and school.get("major")
    }


@lru_cache(maxsize=32)
def _fetch_reference_directions(method: str, n_dim: int, n_points: int) -> np.ndarray:
    return get_reference_directions(method, n_dim=n_dim, n_points=n_points)


def get_cached_reference_directions(method: str, n_dim: int, n_points: int) -> np.ndarray:
    ref_dirs = _fetch_reference_directions(method, n_dim, n_points)
    return ref_dirs.copy()


def calculate_adaptive_thresholds(
    all_school_probabilities: list[float],
    reach_percentile_val: int = None,
    safety_percentile_val: int = None,
) -> dict[str, float]:
    reach_percentile_val = (
        reach_percentile_val or ADAPTIVE_THRESHOLD_PERCENTILES["reach_percentile_val"]
    )
    safety_percentile_val = (
        safety_percentile_val or ADAPTIVE_THRESHOLD_PERCENTILES["safety_percentile_val"]
    )

    if not all_school_probabilities or len(all_school_probabilities) < 3:
        return {
            "safety": SCHOOL_CATEGORY_THRESHOLDS["safety"],
            "target_lower": SCHOOL_CATEGORY_THRESHOLDS["target_lower"],
        }

    target_lower_threshold = float(np.percentile(all_school_probabilities, reach_percentile_val))
    safety_threshold = float(np.percentile(all_school_probabilities, safety_percentile_val))

    if target_lower_threshold > safety_threshold:
        if reach_percentile_val > safety_percentile_val:
            target_lower_threshold, safety_threshold = (
                safety_threshold,
                target_lower_threshold,
            )
        else:
            return {
                "safety": SCHOOL_CATEGORY_THRESHOLDS["safety"],
                "target_lower": SCHOOL_CATEGORY_THRESHOLDS["target_lower"],
            }

    if safety_threshold == target_lower_threshold:
        safety_threshold = min(safety_threshold + 0.005, 1.0)
        if target_lower_threshold > 0.0 and safety_threshold == target_lower_threshold:
            target_lower_threshold = max(target_lower_threshold - 0.005, 0.0)

        if target_lower_threshold >= safety_threshold:
            return {
                "safety": SCHOOL_CATEGORY_THRESHOLDS["safety"],
                "target_lower": SCHOOL_CATEGORY_THRESHOLDS["target_lower"],
            }

    max_probability = max(all_school_probabilities)
    if safety_percentile_val >= 70 and safety_threshold > max_probability:
        safety_threshold = max_probability * 0.9
        if target_lower_threshold >= safety_threshold * 0.8:
            target_lower_threshold = safety_threshold * 0.7

    return {"safety": safety_threshold, "target_lower": target_lower_threshold}
