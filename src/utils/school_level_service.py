from typing import Any

import pandas as pd
import streamlit as st

from src.utils.app_data_loader import load_school_base_data

SCHOOL_LEVEL_PRIORITY = {
    "985": 1,
    "1-50": 2,
    "51-100": 3,
    "211": 4,
    "101-200": 5,
    "201-300": 6,
    "301-500": 6,
    "500之后": 7,
    "500+": 7,
    "普通本科": 8,
    "专接本": 9,
    "三本/民办本科": 10,
    "专科": 11,
    "未知": 12,
    None: 12,
}


@st.cache_data(ttl=3600)
def _get_school_level_mapping() -> dict[str, dict[str, Any]]:
    school_mapping = {}

    try:
        df = load_school_base_data()
        if (
            isinstance(df, pd.DataFrame)
            and "学校名称" in df.columns
            and "school_level" in df.columns
        ):
            df_local = df[["学校名称", "school_level"]].copy()

            df_local["学校名称"] = df_local["学校名称"].astype(str).str.strip()
            df_local["school_level"] = df_local["school_level"].astype(str).str.strip()
            mask_unknown = (
                df_local["school_level"].str.lower().isin(["nan", "none", ""])
                | df_local["school_level"].isna()
            )
            df_local.loc[mask_unknown, "school_level"] = "未知"

            names = df_local["学校名称"].tolist()
            levels = df_local["school_level"].tolist()
            for school_name, level in zip(names, levels):
                info = {
                    "school_level": level,
                    "priority": SCHOOL_LEVEL_PRIORITY.get(level, SCHOOL_LEVEL_PRIORITY["未知"]),
                }
                school_mapping[school_name] = info
                normalized = _normalize_school_name(school_name)
                if normalized and normalized not in school_mapping:
                    school_mapping[normalized] = info
    except Exception:
        pass

    return school_mapping


class SchoolLevelService:
    def __init__(self):
        pass

    def get_school_info(self, school_name: str) -> dict[str, Any]:
        mapping = _get_school_level_mapping()
        if not school_name:
            return {
                "school_level": "未知",
                "priority": SCHOOL_LEVEL_PRIORITY["未知"],
            }

        cleaned_name = str(school_name).strip()
        canonical = SCHOOL_ALIASES.get(cleaned_name, cleaned_name)
        normalized = _normalize_school_name(canonical)

        if cleaned_name in mapping:
            return mapping[cleaned_name]
        if canonical in mapping:
            return mapping[canonical]
        if normalized in mapping:
            return mapping[normalized]

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


_school_level_service = None


def get_school_level_service() -> SchoolLevelService:
    global _school_level_service
    if _school_level_service is None:
        _school_level_service = SchoolLevelService()
    return _school_level_service


def get_school_level_for_analyzer(background_uni_name: str) -> str:
    return get_school_level_service().get_school_level(background_uni_name)


def _normalize_school_name(name: str) -> str:
    try:
        if name is None:
            return ""
        s = str(name).strip()
        s = s.replace("\u00a0", "").replace("\u3000", "")
        s = "".join(ch for ch in s if not ch.isspace())
        s = s.replace("（", "(").replace("）", ")")
        return s
    except Exception:
        return str(name or "").strip()


SCHOOL_ALIASES: dict[str, str] = {
    "新加坡 南洋理工大学": "新加坡南洋理工大学",
    "香港理 工大学": "香港理工大学",
    "香港岭南 大学": "香港岭南大学",
    "香港教育 大学": "香港教育大学",
    "香港中文 大学": "香港中文大学",
    "香港中文大学（深圳校区）": "香港中文大学(深圳校区)",
}
