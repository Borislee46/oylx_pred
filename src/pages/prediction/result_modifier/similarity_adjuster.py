from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pages.prediction.result_modifier.config import SIMILARITY_ADJUSTMENT_RULES_PATH
from src.pages.prediction.result_modifier.streamlit_cache import cache_resource
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


import re

from rapidfuzz import fuzz


def calculate_fuzzy_bias(
    background_major: str, target_major: str, target_major_cn: str | None = None
) -> float:
    if not background_major:
        return 1.0

    bg = background_major.lower()
    tgt_en = target_major.lower()

    score_en = fuzz.token_sort_ratio(bg, tgt_en)

    score_cn = 0.0
    if target_major_cn:
        tgt_cn = target_major_cn.lower()
        score_cn = fuzz.token_sort_ratio(bg, tgt_cn)

    best_score = max(score_en, score_cn)

    if best_score > 92:
        return 1.5
    if best_score > 82:
        return 1.2
    if best_score > 72:
        return 1.1
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


def adjust_similarity_score(
    background_major: str,
    target_major: str,
    similarity: float,
    target_major_cn: str | None = None,
) -> float:
    if not background_major or not target_major:
        return similarity

    rules = _load_similarity_rules_cached()
    adjusted = float(similarity)

    if rules:
        bg = background_major.lower()
        tgt = target_major.lower()

        for rule in rules:
            bks = rule.get("background_keywords", [])
            tks = rule.get("target_keywords", [])
            adj = float(rule.get("adjustment", 0.0))

            bg_match = False
            for k in bks:
                if k in bg:
                    bg_match = True
                    break

            if not bg_match:
                continue

            tgt_match = False
            for k in tks:
                if k in tgt:
                    tgt_match = True
                    break

            if bg_match and tgt_match:
                adjusted += adj
                break

    bias = calculate_fuzzy_bias(background_major, target_major, target_major_cn)
    adjusted *= bias

    return max(0.0, min(1.0, adjusted))
