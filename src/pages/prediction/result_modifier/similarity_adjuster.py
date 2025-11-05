from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.pages.prediction.result_modifier.config import SIMILARITY_ADJUSTMENT_RULES_PATH
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def _normalize_keywords(items: list[str]) -> list[str]:
    """
    规范化关键词列表，处理换行和逗号分隔

    Args:
        items: 原始关键词列表

    Returns:
        规范化后的关键词列表（小写，去重）
    """
    normalized: list[str] = []
    for it in items or []:
        if not isinstance(it, str):
            continue
        parts = []
        for seg in it.split("\n"):
            parts.extend(seg.split(","))
        for p in parts:
            s = p.strip().lower()
            if s:
                normalized.append(s)
    return normalized


def _get_config_path() -> Path:
    """
    获取相似度调整规则配置文件的路径

    Returns:
        配置文件路径
    """
    config_path = SIMILARITY_ADJUSTMENT_RULES_PATH
    # 如果是相对路径，则基于项目根目录解析
    if not config_path.is_absolute():
        project_root = Path.cwd()
        config_path = project_root / config_path
    return config_path


@lru_cache(maxsize=1)
def _load_similarity_rules() -> list[dict[str, Any]]:
    """
    加载相似度调整规则（带缓存）

    Returns:
        启用的规则列表
    """
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
            enabled_rules.append(
                {
                    "background_keywords": _normalize_keywords(r.get("background_keywords", [])),
                    "target_keywords": _normalize_keywords(r.get("target_keywords", [])),
                    "adjustment": float(r.get("adjustment", 0.0)),
                }
            )
        logger.info(f"加载了 {len(enabled_rules)} 条相似度调整规则")
        return enabled_rules
    except json.JSONDecodeError as e:
        logger.error(f"解析相似度调整规则JSON失败: {str(e)}")
        return []
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"读取相似度调整规则文件失败: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"加载相似度调整规则时发生未知错误: {str(e)}", exc_info=True)
        return []


def adjust_similarity_score(background_major: str, target_major: str, similarity: float) -> float:
    """
    根据规则调整相似度分数

    Args:
        background_major: 背景专业
        target_major: 目标专业
        similarity: 原始相似度分数

    Returns:
        调整后的相似度分数（限制在0.0-1.0范围内）
    """
    if not background_major or not target_major:
        return similarity

    bg = background_major.lower()
    tgt = target_major.lower()
    adjusted = float(similarity)

    for rule in _load_similarity_rules():
        bks = rule.get("background_keywords", [])
        tks = rule.get("target_keywords", [])
        adj = float(rule.get("adjustment", 0.0))
        if any(k in bg for k in bks) and any(k in tgt for k in tks):
            adjusted += adj
            break

    return max(0.0, min(1.0, adjusted))
