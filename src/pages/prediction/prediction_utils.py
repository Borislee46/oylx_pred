from functools import lru_cache

import pandas as pd
import streamlit as st

from src.pages.prediction.input_form_components.form_validator import FormValidator
from src.utils.app_data_loader import load_school_major_details_df

normalize_language_score = FormValidator.normalize_language_score
denormalize_language_score = FormValidator.denormalize_language_score


@st.cache_data(ttl=3600, show_spinner=False)
def _get_details_df_version() -> int:
    df = load_school_major_details_df()
    if df is None or df.empty:
        return 0
    try:
        from pandas.util import hash_pandas_object

        return int(hash_pandas_object(df[["学校", "专业英文名称"]]).sum())
    except Exception:
        return len(df)


def format_display_value(value, value_type, language_type=None):
    try:
        if value_type == "gpa":
            return f"{float(value):.2f}"
        elif value_type == "language_score":
            round_to_half = language_type == "雅思"
            score = FormValidator.denormalize_language_score(
                float(value), language_type, round_to_half=round_to_half
            )
            return f"{score:.1f} ({language_type})"
        elif value_type in ["research_count", "award_count", "internship_count", "paper_count"]:
            return f"{int(value)} 个"
        else:
            return str(value)
    except Exception:
        return str(value)


def get_cached_major_similarity(
    target_major=None, background_major=None, cache=None, major1=None, major2=None
):
    first_major = target_major if target_major is not None else major1
    second_major = background_major if background_major is not None else major2
    if not first_major or not second_major or cache is None:
        return 0.0

    key_pair = tuple(sorted([first_major, second_major]))
    key = f"{key_pair[0]}|{key_pair[1]}"
    return cache.get(key, 0.0)


def get_cached_major_similarities_batch(pairs, cache=None):
    if not pairs or cache is None:
        return [0.0] * len(pairs)

    results = []
    for target_major, background_major in pairs:
        if not target_major or not background_major:
            results.append(0.0)
            continue
        key_pair = tuple(sorted([target_major, background_major]))
        key = f"{key_pair[0]}|{key_pair[1]}"
        results.append(cache.get(key, 0.0))

    return results


@lru_cache(maxsize=1000)
def get_cached_major_similarity_key(major1, major2):
    key_pair = tuple(sorted([major1, major2]))
    return f"{key_pair[0]}|{key_pair[1]}"


def format_school_major_details_from_row(row):
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
        if pd.notna(field_value) and str(field_value).strip() != "":
            prefix = "专业网址: " if field == "专业网址" else f"{field}: "
            details.append(f"{prefix}{field_value}")

    if not details:
        return "无详细信息"

    if details and details[0].startswith("专业中文名称:"):
        details[0] = details[0].replace("专业中文名称: ", "", 1).strip()

    return "\n".join(details) if details else "无详细信息"


def get_school_major_details(university, major, return_df=False):
    details_df = load_school_major_details_df()
    if details_df is None:
        return None

    if return_df:
        return details_df

    version = _get_details_df_version()
    return _get_school_major_details_cached(university, major, version)


@lru_cache(maxsize=500)
def _get_school_major_details_cached(university, major, version):
    details_df = load_school_major_details_df()
    try:
        match = details_df[
            (details_df["学校"] == university) & (details_df["专业英文名称"] == major)
        ]
        if not match.empty:
            best_match = match.iloc[0]
            return format_school_major_details_from_row(best_match)
        return "无详细信息"
    except Exception:
        return "获取信息时发生错误"


@st.cache_data(ttl=3600, show_spinner=False)
def _get_valid_combinations_set(version: int):
    details_df = load_school_major_details_df()
    if details_df is None:
        return set()

    try:
        if "学校" not in details_df.columns or "专业英文名称" not in details_df.columns:
            return set()

        valid_df = details_df[["学校", "专业英文名称"]].dropna().astype(str)
        combination_keys = valid_df["学校"] + "|" + valid_df["专业英文名称"]
        return set(combination_keys)
    except Exception:
        return set()


def preload_valid_school_major_combinations():
    version = _get_details_df_version()
    valid_set = _get_valid_combinations_set(version)
    return len(valid_set)


def get_valid_school_major_set() -> set:
    version = _get_details_df_version()
    return _get_valid_combinations_set(version)


@lru_cache(maxsize=1000)
def has_school_major_details(university, major):
    version = _get_details_df_version()
    cache_key = f"{university}|{major}"
    valid_combinations = _get_valid_combinations_set(version)
    return cache_key in valid_combinations


@lru_cache(maxsize=1000)
def is_new_major(university, major):
    version = _get_details_df_version()
    return _is_new_major_cached(university, major, version)


def _is_new_major_cached(university, major, version):
    details_df = load_school_major_details_df()
    if details_df is None:
        return False

    try:
        match = details_df[
            (details_df["学校"] == university) & (details_df["专业英文名称"] == major)
        ]
        if not match.empty:
            best_match = match.iloc[0]
            is_new = best_match.get("新增专业")
            return pd.notna(is_new) and (is_new == "25fall新增" or is_new == "26fall新增")
        return False
    except Exception:
        return False
