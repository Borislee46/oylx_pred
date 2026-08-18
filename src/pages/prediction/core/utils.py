import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_language_score(score: float | int | str, language_type: str) -> float:
    score = float(score)
    if language_type == "托福":
        return score / 120.0
    return score / 9.0


def denormalize_language_score(
    normalized_score: float,
    language_type: str,
    round_to_half: bool = False,
) -> float:
    normalized_score = float(normalized_score)
    if language_type == "托福":
        return normalized_score * 120.0
    score = normalized_score * 9.0
    if round_to_half:
        return round(score * 2) / 2.0
    return score


def get_background_faculty(
    background_major: str,
    cases_df: pd.DataFrame | None,
) -> str | None:
    if not background_major or cases_df is None:
        return None

    if cases_df.empty:
        logger.warning("cases_df 为空，无法查询 faculty")
        return None

    if "background_major" not in cases_df.columns:
        logger.warning("cases_df 缺少 background_major 列，无法查询 faculty")
        return None

    if "faculty" not in cases_df.columns:
        logger.warning("cases_df 缺少 faculty 列，无法查询 faculty")
        return None

    major_rows = cases_df[cases_df["background_major"] == background_major]
    if major_rows.empty:
        return None

    faculties = major_rows["faculty"].dropna().unique()
    if len(faculties) == 0:
        return None
    if len(faculties) > 1:
        logger.warning(
            "background_major=%s 对应多个 faculty: %s，取第一个 %s",
            background_major,
            list(faculties),
            faculties[0],
        )

    return str(faculties[0])


def format_list_field(field_list: list[str]) -> str:
    if not field_list:
        return ""
    if len(field_list) <= 3:
        return ", ".join(field_list)
    return f"{', '.join(field_list[:3])} 等 {len(field_list) - 3} 项"


def format_field(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(value)


def format_float(value: Any, decimals: int = 2) -> Any:
    if value is None or value == "":
        return ""
    return round(float(value), decimals)


def get_cached_major_similarity(
    target_major: str | None = None,
    background_major: str | None = None,
    cache: dict[tuple[str, str], float] | None = None,
    major1: str | None = None,
    major2: str | None = None,
) -> float:
    bg = background_major or major2
    target = target_major or major1

    if cache is None:
        return 0.0

    bg_key = str(bg).strip().lower()
    target_key = str(target).strip().lower()

    if bg_key == target_key:
        return 1.0

    similarity = float(cache.get((bg_key, target_key), 0.0))
    logger.debug(
        "相似度查询 | %s ↔ %s → %.3f",
        bg_key,
        target_key,
        similarity,
    )
    return similarity
