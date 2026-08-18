import time
from typing import Any

import pandas as pd

from src.pages.prediction.app_data import load_raw_cases_data
from src.pages.prediction.core.exceptions import MissingInputError
from src.pages.prediction.core.types import PredictionInput
from src.pages.prediction.core.utils import get_background_faculty
from src.utils.logger import setup_logger
from src.utils.numeric import float_or_none
from src.utils.schools.level_service import get_school_level_service

logger = setup_logger("page3", "prediction")


def _safe_int(v: Any) -> int:
    return int(float(v)) if v is not None and v != "" else 0


def validate_and_clean_input(input_data: dict[str, Any]) -> PredictionInput:
    t0 = time.monotonic()
    cleaned: PredictionInput = {
        "background_university": str(input_data.get("background_university", "")).strip(),
        "background_major": str(input_data.get("background_major", "")).strip(),
        "target_universities": [str(v) for v in input_data.get("target_universities", []) if v],
        "target_majors": [str(v) for v in input_data.get("target_majors", []) if v],
        "experience_details": {
            str(k): str(v) for k, v in input_data.get("experience_details", {}).items()
        },
    }

    gpa_model = float_or_none(input_data.get("gpa_model"))
    gpa = float_or_none(input_data.get("gpa"))
    has_gpa = gpa is not None
    has_lang = (lang := float_or_none(input_data.get("language_score"))) is not None
    if has_gpa:
        cleaned["gpa"] = gpa
        cleaned["gpa_model"] = gpa_model if gpa_model is not None else gpa
    if has_lang:
        cleaned["language_score"] = lang
    if lang_type := input_data.get("language_type"):
        cleaned["language_type"] = str(lang_type).strip()

    for k in ("internship_count", "research_count", "award_count", "paper_count"):
        cleaned[k] = _safe_int(input_data.get(k))
        cleaned["experience_details"][k] = str(cleaned[k])

    if "school_level" in input_data:
        cleaned["school_level"] = str(input_data["school_level"]).strip()

    logger.info(
        "输入清洗完成 | bg_uni=%s bg_major=%s targets=%d gpa=%s lang=%s elapsed=%.3fs",
        cleaned["background_university"][:20],
        cleaned["background_major"][:20],
        len(cleaned["target_universities"]),
        f"{gpa:.2f}" if has_gpa else "缺失",
        f"{lang:.1f}" if has_lang else "缺失",
        time.monotonic() - t0,
    )
    return cleaned


def prepare_input_data(input_data_from_form: dict, *, cases_df=None) -> dict:
    required = ["background_university", "background_major"]
    if missing := [f for f in required if not input_data_from_form.get(f)]:
        logger.warning(f"缺少必需字段: {', '.join(missing)}")
        raise MissingInputError(missing)

    res = input_data_from_form.copy()

    bg_uni = str(res["background_university"])
    res["school_level"] = get_school_level_service().get_school_level(bg_uni)

    bg_major = res.get("background_major")
    if isinstance(bg_major, str) and bg_major:
        if cases_df is None:
            cases_df = load_raw_cases_data()
        faculty = get_background_faculty(bg_major, cases_df)
        if faculty:
            res["faculty"] = faculty

    return res


def prepare_model_inputs(
    current_input_data: dict[str, Any],
    expected_features: list[str],
) -> tuple[dict[str, float | int | str], list[str]]:
    base_features = [f for f in expected_features if f not in ("target_university", "target_major")]
    feature_source = current_input_data
    gpa_model = current_input_data.get("gpa_model")
    if isinstance(gpa_model, (float, int)):
        feature_source = {**current_input_data, "gpa": gpa_model}
    model_input = {
        f: feature_source[f]
        for f in base_features
        if f in feature_source and isinstance(feature_source[f], (float, int, str))
    }
    missing = [f for f in base_features if f not in model_input]
    if missing:
        logger.warning(
            "Model features missing | missing=%s total_expected=%d provided=%d",
            missing,
            len(base_features),
            len(model_input),
        )
    else:
        logger.info(
            "Model features built | features=%d/%d",
            len(model_input),
            len(base_features),
        )
    return model_input, missing


def get_user_specified_combinations(
    input_data: dict[str, Any],
    all_unis: list[str],
) -> list[tuple[str, str]] | None:
    majors = input_data.get("target_majors")
    if not isinstance(majors, list) or not majors:
        return None

    unis = input_data.get("target_universities")
    unis_to_use = unis if isinstance(unis, list) and unis else all_unis
    return [(uni, major) for uni in unis_to_use for major in majors]


def compute_df_fingerprint(df: pd.DataFrame | None) -> int:
    if df is None or df.empty:
        return 0
    from pandas.util import hash_pandas_object

    keys = [
        c for c in ("background_university", "target_university", "target_major") if c in df.columns
    ]
    return int(hash_pandas_object(df[keys]).sum()) if keys else len(df)
