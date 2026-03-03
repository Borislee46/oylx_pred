from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pages.prediction.result_modifier.config import (
    FUZZY_BIAS_MULTIPLIER_HIGH,
    FUZZY_BIAS_MULTIPLIER_LOW,
    FUZZY_BIAS_MULTIPLIER_MID,
    FUZZY_BIAS_THRESHOLD_HIGH,
    FUZZY_BIAS_THRESHOLD_LOW,
    FUZZY_BIAS_THRESHOLD_MID,
    SIMILARITY_ADJUSTMENT_RULES_PATH,
)
from src.pages.prediction.result_modifier.streamlit_cache import cache_resource
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


import re

from rapidfuzz import fuzz


def calculate_fuzzy_bias(
    background_major: str,
    target_major: str,
    target_major_cn: str | None = None,
    fuzzy_score: float | None = None,
) -> float:
    if not background_major:
        return 1.0

    best_score = fuzzy_score
    if best_score is None:
        bg = background_major.lower()
        tgt_en = target_major.lower()
        score_en = fuzz.token_sort_ratio(bg, tgt_en)
        score_cn = 0.0
        if target_major_cn:
            tgt_cn = target_major_cn.lower()
            score_cn = fuzz.token_sort_ratio(bg, tgt_cn)
        best_score = max(score_en, score_cn)

    if best_score > FUZZY_BIAS_THRESHOLD_HIGH:
        return FUZZY_BIAS_MULTIPLIER_HIGH
    if best_score > FUZZY_BIAS_THRESHOLD_MID:
        return FUZZY_BIAS_MULTIPLIER_MID
    if best_score > FUZZY_BIAS_THRESHOLD_LOW:
        return FUZZY_BIAS_MULTIPLIER_LOW
    return 1.0


def _normalize_keywords(items: list[str]) -> list[str]:
    if not items:
        return []
    content = ",".join(str(it) for it in items if it)
    return [s.strip().lower() for s in re.split(r"[,\n]", content) if s.strip()]


def _get_config_path() -> Path:
    config_path = SIMILARITY_ADJUSTMENT_RULES_PATH
    if not config_path.is_absolute():
        project_root = Path.cwd()
        config_path = project_root / config_path
    return config_path


@cache_resource(show_spinner=False, ttl=3600)
def _load_similarity_rules_cached() -> list[dict[str, Any]]:
    try:
        config_path = _get_config_path()
        if not config_path.exists():
            logger.warning(f"相似度调整规则文件不存在: {config_path}")
            return []

        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        rules = data.get("rules", [])
        enabled_rules: list[dict[str, Any]] = []
        for r in rules:
            if not isinstance(r, dict) or not r.get("enabled", False):
                continue

            bks = _normalize_keywords(r.get("background_keywords", []))
            tks = _normalize_keywords(r.get("target_keywords", []))

            if bks and tks:
                enabled_rules.append(
                    {
                        "background_keywords": bks,
                        "target_keywords": tks,
                        "adjustment": float(r.get("adjustment", 0.0)),
                    }
                )
        return enabled_rules

    except json.JSONDecodeError as e:
        logger.error(f"解析相似度调整规则JSON失败: {str(e)}")
        return []
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"读取相似度调整规则文件失败: {str(e)}")
        return []
    except (OSError, TypeError, ValueError, KeyError, AttributeError) as e:
        logger.error(f"加载相似度调整规则失败: {str(e)}", exc_info=True)
        return []


def get_applicable_similarity_rules(background_major: str) -> list[dict[str, Any]]:
    if not background_major:
        return []

    rules = _load_similarity_rules_cached()
    if not rules:
        return []

    bg_lower = background_major.lower()
    applicable = []
    for r in rules:
        bks = r.get("background_keywords", [])
        if any(k in bg_lower for k in bks):
            applicable.append(r)
    return applicable


def adjust_similarity_score(
    background_major: str,
    target_major: str,
    similarity: float,
    target_major_cn: str | None = None,
    fuzzy_score: float | None = None,
    applicable_rules: list[dict[str, Any]] | None = None,
) -> float:
    if not background_major or not target_major:
        return similarity

    adjusted = float(similarity)
    rules = (
        applicable_rules
        if applicable_rules is not None
        else get_applicable_similarity_rules(background_major)
    )

    if rules:
        tgt = target_major.lower()
        for rule in rules:
            tks = rule.get("target_keywords", [])
            adj = float(rule.get("adjustment", 0.0))

            tgt_match = False
            for k in tks:
                if k in tgt:
                    tgt_match = True
                    break

            if tgt_match:
                adjusted += adj
                break

    bias = calculate_fuzzy_bias(
        background_major, target_major, target_major_cn, fuzzy_score=fuzzy_score
    )
    adjusted *= bias

    return max(0.0, min(1.0, adjusted))
