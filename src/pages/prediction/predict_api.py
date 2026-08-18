"""预测入口适配层 — 测试 / 报告脚本使用的 dict-in / dict-out `predict()`。

背景：
  - 2026-07-17 提交 67fc0f26 删除了独立的 FastAPI 服务 `src/api/prediction/`。
  - 应用运行时从不走该 API（独立 FastAPI 服务已删除），正式确认删除后无需恢复 HTTP 层。
  - 但 data_quality 测试与报告重跑脚本仍依赖 `predict(payload)` 契约。

本模块保留旧 `json_api.predict()` 的纯函数核心
（校验 → 归一化 → 跨学部确认 → 委托 `flow.pipeline.run_prediction_pipeline`），
仅修正 import 路径并去掉任务存储/异步部分。返回格式与旧 API 兼容。
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

import pandas as pd

from src.adjustment.experience_text_validator import (
    has_meaningful_experience_text,
)
from src.pages.prediction.app_data import (
    load_raw_cases_data,
    load_school_base_data,
)
from src.pages.prediction.flow.form_normalizer import (
    normalize_form_data_for_prediction,
)
from src.pages.prediction.input_form_components import FormValidator, GPAConverter
from src.pages.prediction.input_form_components.cross_faculty_guard import (
    quick_cross_faculty_check,
)
from src.pages.prediction.input_form_components.form_config import (
    DEFAULT_GPA_SCALE,
    GPA_SCALES,
    LANGUAGE_TYPES,
)
from src.pages.prediction.input_form_components.form_validator import ValidationError
from src.pages.prediction.page_data_loader import cached_get_prediction_model
from src.utils.numeric.coerce import float_or_none
from src.utils.numeric.scalars import float_eq

logger = logging.getLogger(__name__)


def _list_str(value: Any) -> list[str]:
    return (
        [str(v).strip() for v in value if v and str(v).strip()] if isinstance(value, list) else []
    )


def _dict_str(value: Any) -> dict[str, str]:
    return (
        {str(k): str(v) for k, v in value.items() if k is not None}
        if isinstance(value, dict)
        else {}
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value)) if value is not None and value != "" else default
    except (ValueError, TypeError):
        return default


def _parse_gpa_raw(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    num = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            num += ch
        elif num:
            break

    try:
        return float(num) if num else None
    except ValueError:
        return None


def _normalize_gpa_scale(value: Any) -> str:
    if value is None or value == "":
        return DEFAULT_GPA_SCALE

    if isinstance(value, (int, float)):
        v = float(value)
        if float_eq(v, 4.0):
            return "4.0"
        if float_eq(v, 4.3):
            return "4.3"
        if float_eq(v, 4.5):
            return "4.5"
        if float_eq(v, 5.0):
            return "5.0"
        if v.is_integer():
            s = str(int(v))
            if s == "4":
                return "4.0"
            if s == "5":
                return "5.0"
            return s
        return str(value).strip()

    s = str(value).strip()
    if s == "4":
        s = "4.0"
    elif s == "5":
        s = "5.0"

    if s not in GPA_SCALES:
        return DEFAULT_GPA_SCALE

    return s


def _build_form_data(payload: dict[str, Any]) -> dict[str, Any]:
    language_type = payload.get("language_type")
    if language_type not in LANGUAGE_TYPES:
        language_type = LANGUAGE_TYPES[0]

    lang_raw = payload.get("language_score_raw")
    has_lang_raw = lang_raw not in (None, "")
    lang_user_provided = payload.get("language_score_user_provided")
    if lang_user_provided is None:
        # 兼容旧 API 契约：显式传了分数即视为用户提供（生产表单会显式传该 flag）
        lang_user_provided = has_lang_raw

    return {
        "target_majors": _list_str(payload.get("target_majors")),
        "target_universities": _list_str(payload.get("target_universities")),
        "background_university": payload.get("background_university"),
        "background_major_original": payload.get("background_major_original"),
        "background_major": payload.get("background_major"),
        "gpa_raw": _parse_gpa_raw(payload.get("gpa_raw")),
        "gpa_scale": _normalize_gpa_scale(payload.get("gpa_scale")),
        "exam_type": payload.get("exam_type"),
        "exam_score": float_or_none(payload.get("exam_score")),
        "language_type": language_type,
        "language_score_raw": float_or_none(payload.get("language_score_raw")),
        "language_score_user_provided": bool(lang_user_provided),
        "language_score_input_error": bool(payload.get("language_score_input_error", False)),
        "research_count": _safe_int(payload.get("research_count"), 0),
        "award_count": _safe_int(payload.get("award_count"), 0),
        "internship_count": _safe_int(payload.get("internship_count"), 0),
        "paper_count": _safe_int(payload.get("paper_count"), 0),
        "experience_details": _dict_str(payload.get("experience_details")),
    }


def _errors_to_dict(errors: list[ValidationError]) -> list[dict[str, Any]]:
    return [e.to_dict() for e in errors if isinstance(e, ValidationError)]


def validate_and_normalize(
    payload: dict[str, Any],
    cases_df: pd.DataFrame | None = None,
    school_base_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "errors": [
                {"field": "_", "message": "payload 必须是 JSON object", "severity": "error"}
            ],
        }

    cases_df = cases_df if cases_df is not None else load_raw_cases_data()
    school_base_df = school_base_df if school_base_df is not None else load_school_base_data()
    gpa_converter = GPAConverter(school_base_df)

    form_data = _build_form_data(payload)
    errors = FormValidator.validate_form_data(form_data, gpa_converter)
    if errors:
        return {"ok": False, "errors": _errors_to_dict(errors), "warnings": []}

    normalized_input = normalize_form_data_for_prediction(form_data, cases_df, gpa_converter)
    return {
        "ok": True,
        "errors": [],
        "warnings": [],
        "normalized_input": normalized_input,
        "original_form": form_data,
    }


@lru_cache(maxsize=1)
def _load_model_and_features():
    model = cached_get_prediction_model("xgboost")
    if model is None:
        raise ValueError("模型加载失败")
    features = model.feature_names
    if not isinstance(features, list) or not features:
        raise ValueError("模型特征列表为空")
    return model, features


def predict(
    payload: dict[str, Any],
    confirm_cross_faculty: bool = False,
    ablation_config: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """与旧 `json_api.predict` 兼容的 dict-in / dict-out 预测入口。

    校验 → 归一化 → 跨学部确认 → 委托 `flow.pipeline.run_prediction_pipeline`。
    返回：``{"ok", "task_id", "errors", "warnings", "needs_confirmation",
    "normalized_input", "result"}``，其中 ``result.unified_results`` 与旧格式一致。
    """
    try:
        v = validate_and_normalize(payload)
        if not v.get("ok"):
            return v

        confirmation_needed, confirmation_data = _check_cross_faculty_needed(
            payload, v["normalized_input"], confirm_cross_faculty
        )
        if confirmation_needed:
            return {
                "ok": True,
                "errors": [],
                "warnings": v.get("warnings", []),
                "needs_confirmation": True,
                "confirmation": confirmation_data,
                "normalized_input": v["normalized_input"],
                "result": None,
            }

        task_id = str(uuid.uuid4())
        result = _execute_prediction_core(
            payload,
            v["normalized_input"],
            confirm_cross_faculty,
            ablation_config=ablation_config,
        )

        return {
            "ok": True,
            "task_id": task_id,
            "errors": [],
            "warnings": v.get("warnings", []),
            "needs_confirmation": False,
            "normalized_input": v["normalized_input"],
            "result": result,
        }
    except Exception as e:
        logger.error(f"预测失败: {e}", exc_info=True)
        return {
            "ok": False,
            "errors": [{"field": "_", "message": f"Server Error: {str(e)}", "severity": "error"}],
            "warnings": [],
        }


def _check_cross_faculty_needed(
    payload: dict[str, Any], normalized_input: dict[str, Any], confirm_cross_faculty: bool
) -> tuple[bool, dict[str, Any] | None]:
    cases_df = load_raw_cases_data()
    selected_categories = _list_str(payload.get("selected_major_categories"))
    selected_majors = _list_str(payload.get("selected_target_majors")) or _list_str(
        normalized_input.get("target_majors")
    )

    is_cross_faculty = False
    background_faculty = None
    target_faculties: set[str] = set()
    agent_approved = False

    if normalized_input.get("background_major") and (selected_categories or selected_majors):
        is_cross_faculty, background_faculty, target_faculties, agent_approved = (
            quick_cross_faculty_check(
                normalized_input.get("background_major"),
                selected_categories,
                selected_majors,
                cases_df,
            )
        )

    if is_cross_faculty and not agent_approved and not confirm_cross_faculty:
        return True, {
            "background_faculty": background_faculty,
            "target_faculties": sorted(target_faculties),
            "agent_approved": agent_approved,
        }
    return False, None


def _execute_prediction_core(
    payload: dict[str, Any],
    normalized_input: dict[str, Any],
    confirm_cross_faculty: bool,
    ablation_config: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """委托 pipeline.run_prediction_pipeline 执行完整预测。"""
    from src.pages.prediction.core.data_manager import _data_manager
    from src.pages.prediction.flow.pipeline import run_prediction_pipeline
    from src.pages.prediction.flow.preparer import compute_df_fingerprint
    from src.pages.prediction.page_data_loader import machine_learning_model

    cases_df = load_raw_cases_data()
    page_state = machine_learning_model.resource_loader()

    # ── 指纹校验（与 pipeline._execute_prediction_pipeline 一致）────
    fingerprint = compute_df_fingerprint(cases_df)
    if fingerprint != page_state.cases_df_fingerprint:
        logger.error("案例数据指纹不匹配，模型与数据版本不一致")
        return {"ok": False, "error": "cases_df_fingerprint_mismatch"}

    # ── 构建 input_data ──────────────────────────────────────────
    input_data = dict(normalized_input)

    # 全量院校/专业（不传则 fallback 到 valid_*）
    all_unis = _list_str(payload.get("all_universities_target"))
    all_majors = _list_str(payload.get("all_majors_target"))
    if not all_unis or not all_majors:
        if not all_unis:
            all_unis = sorted(_data_manager.valid_universities)
        if not all_majors:
            all_majors = sorted(_data_manager.valid_majors)

    input_data["_all_universities_target"] = all_unis
    input_data["_all_majors_target"] = all_majors

    # ── 跨学部确认 ───────────────────────────────────────────
    _, _, _, agent_approved = quick_cross_faculty_check(
        normalized_input.get("background_major"),
        _list_str(payload.get("selected_major_categories")),
        _list_str(payload.get("selected_target_majors"))
        or _list_str(normalized_input.get("target_majors")),
        cases_df,
    )
    input_data["_cross_faculty_confirmed"] = bool(confirm_cross_faculty or agent_approved)

    # ── 文本含金量 ───────────────────────────────────────────
    experience_details = _dict_str(payload.get("experience_details"))
    has_valid_exp = has_meaningful_experience_text(experience_details)
    input_data["_has_valid_experience"] = has_valid_exp

    # ── 消融实验 → adjustment_flag_overrides ─────────────────
    flag_overrides = _ablation_to_flags(ablation_config) if ablation_config else None

    # ── 委托 pipeline ────────────────────────────────────────
    model, features = _load_model_and_features()
    result_model = run_prediction_pipeline(
        input_data=input_data,
        model_name="xgboost",
        cases_df_fingerprint=fingerprint,
        loaded_feature_names=features,
        adjustment_flag_overrides=flag_overrides,
    )

    return _prediction_model_to_dict(result_model)


def _ablation_to_flags(ablation_config: dict[str, bool]) -> dict[str, bool]:
    """将 short-key 消融配置映射为 pipeline flag names。"""
    _AC_KEY_MAP = {
        "gpa": "enable_gpa_penalty",
        "language": "enable_language_penalty",
        "cross_major": "enable_cross_major_penalty",
        "cross_faculty": "enable_cross_faculty_penalty",
        "professional": "enable_professional_penalty",
        "language_requirement": "enable_language_requirement_penalty",
        "text_boost": "enable_text_boost",
    }
    return {_AC_KEY_MAP[k]: v for k, v in ablation_config.items() if k in _AC_KEY_MAP}


def _prediction_model_to_dict(model) -> dict[str, Any]:
    """将 PredictionResultModel 转为 dict（与旧返回格式兼容）。"""
    return {
        "similarity_results": model.similarity_results,
        "cross_major_results": model.cross_major_results,
        "user_specified_results": model.user_specified_results,
        "unified_results": model.unified_results,
        "meta": model.meta,
    }
