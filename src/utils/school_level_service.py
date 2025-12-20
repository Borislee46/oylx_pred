from typing import Any

import pandas as pd
import streamlit as st

from src.utils.app_data_loader import load_school_base_data
from src.utils.school_constants import (
    LANGUAGE_BOOST_MULTIPLIERS,
    OVERSEAS_SCHOOL_LEVELS,
    SCHOOL_LEVEL_ALIASES,
    SCHOOL_LEVEL_PRIORITY,
    SCHOOL_LEVEL_SCORES,
)


def _normalize_school_level(level: Any) -> str:
    if level is None:
        return "未知"
    cleaned = str(level).strip()
    if not cleaned or cleaned.lower() in {"nan", "none"}:
        return "未知"
    return SCHOOL_LEVEL_ALIASES.get(cleaned, cleaned)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_school_level_mapping() -> dict[str, dict[str, Any]]:
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
        mapping = _get_school_level_mapping()
        if not school_name:
            return {
                "school_level": "未知",
                "priority": SCHOOL_LEVEL_PRIORITY["未知"],
            }

        cleaned_name = str(school_name).strip()

        if cleaned_name in mapping:
            info = mapping[cleaned_name]
            level = _normalize_school_level(info.get("school_level"))
            return {
                "school_level": level,
                "priority": SCHOOL_LEVEL_PRIORITY.get(level, SCHOOL_LEVEL_PRIORITY["未知"]),
            }

        if "985" in cleaned_name or "211" in cleaned_name:
            inferred_level = "985" if "985" in cleaned_name else "211"
            return {
                "school_level": inferred_level,
                "priority": SCHOOL_LEVEL_PRIORITY[inferred_level],
            }

        return {
            "school_level": "未知",
            "priority": SCHOOL_LEVEL_PRIORITY["未知"],
        }

    def get_school_level(self, school_name: str) -> str:
        return self.get_school_info(school_name)["school_level"]

    def get_school_priority(self, school_name: str) -> int:
        return self.get_school_info(school_name)["priority"]

    def compare_schools(self, school1: str, school2: str) -> int:
        priority1 = self.get_school_priority(school1)
        priority2 = self.get_school_priority(school2)

        if priority1 < priority2:
            return -1
        elif priority1 > priority2:
            return 1
        else:
            return 0

    def is_overseas_school(self, school_name: str) -> bool:
        school_level = _normalize_school_level(self.get_school_level(school_name))
        return school_level in OVERSEAS_SCHOOL_LEVELS

    def get_language_boost_multiplier(self, school_name: str) -> float:
        school_level = _normalize_school_level(self.get_school_level(school_name))
        return LANGUAGE_BOOST_MULTIPLIERS.get(school_level, 1.0)

    def get_school_score(self, school_name: str | None) -> float:
        if not school_name:
            return SCHOOL_LEVEL_SCORES["未知"]
        level = _normalize_school_level(self.get_school_level(school_name))
        return float(SCHOOL_LEVEL_SCORES.get(level, SCHOOL_LEVEL_SCORES["未知"]))


_school_level_service = None


def get_school_level_service() -> SchoolLevelService:
    global _school_level_service
    if _school_level_service is None:
        _school_level_service = SchoolLevelService()
    return _school_level_service
