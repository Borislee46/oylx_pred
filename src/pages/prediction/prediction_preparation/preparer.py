import hashlib
from typing import Any

import pandas as pd

from src.pages.prediction.core.exceptions import InvalidInputError, MissingInputError
from src.pages.prediction.core.types import PredictionInput
from src.pages.prediction.core.utils import get_background_faculty, get_background_faculty
from src.utils.app_data_loader import load_raw_cases_data
from src.utils.logger import setup_logger
from src.utils.school_level_service import get_school_level_service

logger = setup_logger("page3", "prediction")


def _safe_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None and v != "" else None
    except (ValueError, TypeError):
        return None


def _safe_int(v: Any) -> int:
    try:
        return int(float(v)) if v is not None and v != "" else 0
    except (ValueError, TypeError):
        return 0


def validate_and_clean_input(input_data: dict[str, Any]) -> PredictionInput:
    """清洗并规范化输入数据类型"""
    cleaned: PredictionInput = {
        "background_university": str(input_data.get("background_university", "")).strip(),
        "background_major": str(input_data.get("background_major", "")).strip(),
        "target_universities": [str(v) for v in input_data.get("target_universities", []) if v],
        "target_majors": [str(v) for v in input_data.get("target_majors", []) if v],
        "experience_details": {str(k): str(v) for k, v in input_data.get("experience_details", {}).items()},
    }

    if (gpa := _safe_float(input_data.get("gpa"))) is not None:
        cleaned["gpa"] = gpa
    if (lang := _safe_float(input_data.get("language_score"))) is not None:
        cleaned["language_score"] = lang
    if lang_type := input_data.get("language_type"):
        cleaned["language_type"] = str(lang_type).strip()

    for k in ("internship_count", "research_count", "award_count", "paper_count"):
        cleaned[k] = _safe_int(input_data.get(k))
        # 同步到 experience_details 以便后续处理
        cleaned["experience_details"][k] = str(cleaned[k])

    if "school_level" in input_data:
        cleaned["school_level"] = str(input_data["school_level"]).strip()

    return cleaned


def prepare_input_data(input_data_from_form: dict) -> dict:
    """基础字段校验及背景信息补全"""
    if not isinstance(input_data_from_form, dict):
        raise InvalidInputError("_", value=type(input_data_from_form).__name__, expected="dict")

    # 必需字段检查
    required = ["background_university", "background_major"]
    if missing := [f for f in required if not input_data_from_form.get(f)]:
        logger.warning(f"缺少必需字段: {', '.join(missing)}")
        raise MissingInputError(missing)

    res = input_data_from_form.copy()
    
    # 补全学校等级和学部信息
    bg_uni = str(res["background_university"])
    res["school_level"] = get_school_level_service().get_school_level(bg_uni)

    bg_major = res.get("background_major")
    if isinstance(bg_major, str) and bg_major:
        faculty = get_background_faculty(bg_major, load_raw_cases_data())
        if faculty:
            res["faculty"] = faculty

    return res


def prepare_model_inputs(
    current_input_data: dict[str, Any],
    expected_features: list[str],
) -> tuple[dict[str, float | int | str], list[str]]:
    """筛选模型所需的特征字段"""
    base_features = [f for f in expected_features if f not in ("target_university", "target_major")]
    model_input = {
        f: current_input_data[f] 
        for f in base_features 
        if f in current_input_data and isinstance(current_input_data[f], (float, int, str))
    }
    missing = [f for f in base_features if f not in model_input]
    if missing:
        logger.error(f"缺少模型特征: {missing}")
    return model_input, missing


def get_user_specified_combinations(
    input_data: dict[str, Any],
    all_unis: list[str],
) -> list[tuple[str, str]] | None:
    """获取用户指定的 (大学, 专业) 组合"""
    majors = input_data.get("target_majors")
    if not isinstance(majors, list) or not majors:
        return None

    unis = input_data.get("target_universities")
    unis_to_use = unis if isinstance(unis, list) and unis else all_unis
    return [(uni, major) for uni in unis_to_use for major in majors]


def compute_list_fingerprint(lst: list[str]) -> tuple[int, int]:
    """计算列表内容的稳定指纹"""
    if not lst: return (0, 0)
    try:
        content = "\n".join(sorted(str(x) for x in lst)).encode("utf-8")
        stable_hash = int.from_bytes(hashlib.sha1(content).digest()[:8], "big")
        return (len(lst), stable_hash)
    except Exception as e:
        logger.warning(f"计算列表指纹失败: {e}")
        return (len(lst), 0)


def compute_df_fingerprint(df: pd.DataFrame | None) -> int:
    """计算 DataFrame 的关键列指纹"""
    if df is None or df.empty: return 0
    try:
        from pandas.util import hash_pandas_object
        keys = [c for c in ("background_university", "target_university", "target_major") if c in df.columns]
        return int(hash_pandas_object(df[keys]).sum()) if keys else len(df)
    except Exception as e:
        logger.warning(f"计算DF指纹失败: {e}")
        return len(df)

