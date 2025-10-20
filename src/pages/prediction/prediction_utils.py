from functools import lru_cache
from typing import List, Set, Tuple, Union

import pandas as pd
import streamlit as st

from src.pages.prediction.input_form_components.form_validator import FormValidator
from src.utils.app_data_loader import load_school_major_details_df
from src.utils.logger import setup_logger

normalize_language_score = FormValidator.normalize_language_score
denormalize_language_score = FormValidator.denormalize_language_score

_utils_logger = setup_logger("page3", "prediction")


class SchoolMajorDataManager:
    def __init__(self):
        self._details_df = None
        self._valid_combinations = None
        self._details_version = None

    @property
    def details_df(self):
        if self._details_df is None:
            self._details_df = load_school_major_details_df()
        return self._details_df

    @property
    def details_version(self) -> int:
        if self._details_version is None:
            self._details_version = self._compute_details_version()
        return self._details_version

    def _compute_details_version(self) -> int:
        df = self.details_df
        if df is None or df.empty:
            return 0
        try:
            from pandas.util import hash_pandas_object

            return int(hash_pandas_object(df[["学校", "专业英文名称"]]).sum())
        except Exception:
            return len(df)

    @property
    def valid_combinations(self) -> Set[str]:
        if self._valid_combinations is None:
            self._valid_combinations = self._load_valid_combinations()
        return self._valid_combinations

    def _load_valid_combinations(self) -> Set[str]:
        df = self.details_df
        if df is None:
            return set()

        try:
            if "学校" not in df.columns or "专业英文名称" not in df.columns:
                return set()

            valid_df = df[["学校", "专业英文名称"]].dropna().astype(str)
            combination_keys = valid_df["学校"] + "|" + valid_df["专业英文名称"]
            return set(combination_keys)
        except Exception:
            return set()


_data_manager = SchoolMajorDataManager()


def get_background_faculty(background_major: str, cases_df: pd.DataFrame | None) -> str | None:
    if not background_major or cases_df is None or cases_df.empty:
        return None
    try:
        major_match = cases_df[cases_df["background_major"] == background_major]
        if not major_match.empty and "faculty" in major_match.columns:
            faculty = major_match["faculty"].iloc[0]
            if pd.notna(faculty) and str(faculty).strip():
                return faculty
    except Exception as e:
        _utils_logger.warning(f"查询背景学院失败: {e}")
    return None


def format_list_field(field_list: list[str]) -> str:
    if not field_list:
        return ""
    return (
        ", ".join(field_list)
        if len(field_list) <= 3
        else f"{', '.join(field_list[:3])} 等 {len(field_list) - 3} 项"
    )


def format_field(value) -> str:
    if value is None or value == "":
        return ""
    return str(value)


def format_float(value, decimals: int = 2):
    try:
        if value is None or value == "":
            return ""
        return round(float(value), decimals)
    except Exception:
        return value


def format_display_value(value, value_type: str, language_type: str = None) -> str:
    try:
        if value_type == "gpa":
            return f"{float(value):.2f}"
        elif value_type == "language_score":
            round_to_half = language_type == "雅思"
            score = denormalize_language_score(
                float(value), language_type, round_to_half=round_to_half
            )
            return f"{score:.1f} ({language_type})"
        elif value_type in [
            "research_count",
            "award_count",
            "internship_count",
            "paper_count",
        ]:
            return f"{int(value)} 个"
        else:
            return str(value)
    except Exception:
        return str(value)


def _create_major_similarity_key(major1: str, major2: str) -> str:
    key_pair = tuple(sorted([major1, major2]))
    return f"{key_pair[0]}|{key_pair[1]}"


def get_cached_major_similarity(
    target_major: str = None,
    background_major: str = None,
    cache: dict = None,
    major1: str = None,
    major2: str = None,
) -> float:
    first_major = target_major or major1
    second_major = background_major or major2

    if not first_major or not second_major or cache is None:
        return 0.0

    key = _create_major_similarity_key(first_major, second_major)
    return cache.get(key, 0.0)


def get_cached_major_similarities_batch(
    pairs: List[Tuple[str, str]], cache: dict = None
) -> List[float]:
    if not pairs or cache is None:
        return [0.0] * len(pairs)

    return [
        get_cached_major_similarity(major1=target, major2=background, cache=cache)
        for target, background in pairs
    ]


def format_school_major_details_from_row(row: pd.Series) -> str:
    if row is None or row.empty:
        return "无详细信息"

    key_fields = [
        "专业中文名称",
        "开学季",
        "申请开始时间",
        "申请截止时间",
        "学习年限",
        "学费",
        "授课语言",
        "录取要求",
        "专业背景要求",
        "GPA要求",
        "IELTS",
        "TOEFL",
        "CET-6",
        "考试要求",
        "特殊要求",
        "申请方式",
        "推荐信方式",
        "成绩送分要求",
        "是否面试",
        "是否笔试",
        "考核形式",
        "申请注意事项",
        "专业网址",
    ]

    details = []
    for field in key_fields:
        field_value = row.get(field)
        if pd.notna(field_value) and str(field_value).strip():
            prefix = "专业网址: " if field == "专业网址" else f"{field}: "
            details.append(f"{prefix}{field_value}")

    if not details:
        return "无详细信息"

    if details and details[0].startswith("专业中文名称:"):
        details[0] = details[0].replace("专业中文名称: ", "", 1).strip()

    return "\n".join(details)


def get_school_major_details(
    university: str | None, major: str | None, return_df: bool = False
) -> Union[str, pd.DataFrame, None]:
    if return_df:
        return _data_manager.details_df

    return _get_school_major_details_cached(university, major, _data_manager.details_version)


@lru_cache(maxsize=500)
def _get_school_major_details_cached(university: str, major: str, version: int) -> str:
    df = _data_manager.details_df
    match = df[(df["学校"] == university) & (df["专业英文名称"] == major)]
    if not match.empty:
        return format_school_major_details_from_row(match.iloc[0])
    return "无详细信息"


def get_valid_school_major_set() -> Set[str]:
    return _data_manager.valid_combinations


@lru_cache(maxsize=1000)
def has_school_major_details(university: str, major: str) -> bool:
    cache_key = f"{university}|{major}"
    return cache_key in _data_manager.valid_combinations


@lru_cache(maxsize=1000)
def is_new_major(university: str, major: str) -> bool:
    return _is_new_major_cached(university, major, _data_manager.details_version)


def _is_new_major_cached(university: str, major: str, version: int) -> bool:
    df = _data_manager.details_df
    match = df[(df["学校"] == university) & (df["专业英文名称"] == major)]
    if not match.empty:
        is_new = match.iloc[0].get("新增专业")
        return pd.notna(is_new) and (is_new == "25fall新增" or is_new == "26fall新增")
    return False


@st.cache_data(ttl=3600)
def get_cached_data_manager() -> SchoolMajorDataManager:
    return SchoolMajorDataManager()
