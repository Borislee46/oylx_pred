from functools import lru_cache
from typing import Any

import pandas as pd

from src.utils.schools.constants import (
    LANGUAGE_BOOST_MULTIPLIERS,
    OVERSEAS_SCHOOL_LEVELS,
    SCHOOL_LEVEL_PRIORITY,
    SCHOOL_LEVEL_SCORES,
)


def _normalize_school_level(level: Any) -> str:
    if level is None:
        return "未知"
    cleaned = str(level).strip()
    if not cleaned or cleaned.lower() in {"nan", "none"}:
        return "未知"
    return cleaned


@lru_cache(maxsize=1)
def _get_school_level_mapping() -> dict[str, dict[str, Any]]:
    from src.utils.schools.data import load_school_base_data

    school_mapping = {}

    df = load_school_base_data()
    if isinstance(df, pd.DataFrame) and "学校名称" in df.columns and "school_level" in df.columns:
        df_local = df[["学校名称", "school_level"]].copy()

        df_local["学校名称"] = df_local["学校名称"].astype(str).str.strip()
        df_local["school_level"] = df_local["school_level"].apply(_normalize_school_level)

        names = df_local["学校名称"].tolist()
        levels = df_local["school_level"].tolist()
        for school_name, level in zip(names, levels, strict=True):
            info = {
                "school_level": level,
                "priority": SCHOOL_LEVEL_PRIORITY.get(level, SCHOOL_LEVEL_PRIORITY["未知"]),
            }
            school_mapping[school_name] = info

    return school_mapping


class SchoolLevelService:
    def get_school_info(self, school_name: str) -> dict[str, Any]:
        if not school_name:
            return {
                "school_level": "未知",
                "priority": SCHOOL_LEVEL_PRIORITY["未知"],
            }

        mapping = _get_school_level_mapping()
        cleaned_name = str(school_name).strip()

        if cleaned_name in mapping:
            return mapping[cleaned_name]

        return {
            "school_level": "未知",
            "priority": SCHOOL_LEVEL_PRIORITY["未知"],
        }

    def get_school_level(self, school_name: str) -> str:
        return self.get_school_info(school_name)["school_level"]

    def get_school_priority(self, school_name: str) -> int:
        return self.get_school_info(school_name)["priority"]

    def compare_schools(self, school1: str, school2: str) -> int:
        p1 = self.get_school_priority(school1)
        p2 = self.get_school_priority(school2)
        return (p1 > p2) - (p1 < p2)

    def is_overseas_school(self, school_name: str) -> bool:
        return self.get_school_level(school_name) in OVERSEAS_SCHOOL_LEVELS

    def get_language_boost_multiplier(self, school_name: str) -> float:
        return LANGUAGE_BOOST_MULTIPLIERS.get(self.get_school_level(school_name), 1.0)

    def get_school_score(self, school_name: str | None) -> float:
        level = self.get_school_level(school_name or "")
        return float(SCHOOL_LEVEL_SCORES.get(level, SCHOOL_LEVEL_SCORES["未知"]))


_school_level_service = None


def get_school_level_service() -> SchoolLevelService:
    global _school_level_service
    if _school_level_service is None:
        _school_level_service = SchoolLevelService()
    return _school_level_service
