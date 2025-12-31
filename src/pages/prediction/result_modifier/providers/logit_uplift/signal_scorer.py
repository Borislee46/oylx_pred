"""
保证文本加成有足够的泛化性，添加常用的关键词词库
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _Rule:
    pattern: str
    score: float
    tag: str
    fields: tuple[str, ...] | None


class SignalScorer:
    """
    高价值信号评分器v2.6。

    该类通过硬匹配关键词（词库模式）来识别文本中的"强信号"。
    这些信号通常是统计模型（如 TF-IDF）难以捕捉到的决定性细节，例如具体的奖项名称、核心期刊等。

    设计意图：
    - **弥补泛化性**：向量相似度往往只能捕捉到语义相近，而词库可以锁定具体的成就水平（如"一等奖" vs "三等奖"）。
    - **字段隔离**：支持针对特定维度（如仅限"科研"）的规则，也可以定义全局通用的规则。
    """

    def __init__(
        self,
        lexicon_path: str | None,
        enabled_fields: tuple[str, ...] | None,
        per_field_cap: float,
        lexicon_weight: float,
    ) -> None:
        """
        初始化评分器并加载词库。

        Args:
            lexicon_path: 关键词规则 JSON 文件路径。
            enabled_fields: 允许进行关键词扫描的字段列表。
            per_field_cap: 单个字段允许的最大词库加成上限。
            lexicon_weight: 词库总权重的缩放系数。
        """
        self._lexicon_path = lexicon_path
        self._enabled_fields = enabled_fields
        self._per_field_cap = float(per_field_cap)
        self._lexicon_weight = float(lexicon_weight)
        self._rules_by_field: dict[str, list[_Rule]] = {}
        self._global_rules: list[_Rule] = []

        all_rules = self._load_rules(lexicon_path)
        for r in all_rules:
            if r.fields is None:
                self._global_rules.append(r)
            else:
                for f in r.fields:
                    if f not in self._rules_by_field:
                        self._rules_by_field[f] = []
                    self._rules_by_field[f].append(r)

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    def _load_rules(self, lexicon_path: str | None) -> tuple[_Rule, ...]:
        if not lexicon_path:
            return ()
        path = Path(lexicon_path)
        if not path.exists():
            return ()
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f) or {}
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
            return ()

        rules_raw = obj.get("rules")
        if not isinstance(rules_raw, list):
            return ()

        out: list[_Rule] = []
        for r in rules_raw:
            if not isinstance(r, dict):
                continue
            pattern = r.get("pattern")
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            score = self._safe_float(r.get("score"), 0.0)
            score = float(min(max(score, 0.0), 1.0))
            tag = r.get("tag")
            if not isinstance(tag, str) or not tag.strip():
                tag = pattern.strip()
            fields_raw = r.get("fields")
            fields: tuple[str, ...] | None
            if fields_raw is None:
                fields = None
            elif isinstance(fields_raw, list) and all(isinstance(x, str) for x in fields_raw):
                fields = tuple(fields_raw)
            else:
                continue
            out.append(
                _Rule(pattern=pattern.strip().lower(), score=score, tag=tag.strip(), fields=fields)
            )
        return tuple(out)

    def score(
        self, texts_by_field: dict[str, str]
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        if not texts_by_field or self._lexicon_weight <= 0:
            return {}, {}

        enabled = set(self._enabled_fields) if self._enabled_fields else None
        bonuses: dict[str, float] = {}
        tags_found: dict[str, list[str]] = {}

        for field, text in texts_by_field.items():
            if enabled is not None and field not in enabled:
                continue
            if not isinstance(text, str) or not text.strip():
                continue

            t = text.lower()
            best = 0.0
            matched_tags: list[str] = []

            target_rules = self._rules_by_field.get(field, []) + self._global_rules
            if not target_rules:
                continue

            for rule in target_rules:
                if rule.pattern in t:
                    matched_tags.append(rule.tag)
                    if rule.score > best:
                        best = rule.score

            if best > 0:
                # 最终得分为：max(scores) * lexicon_weight，并对单字段封顶
                bonuses[field] = min(self._per_field_cap, best * self._lexicon_weight)
                tags_found[field] = list(set(matched_tags))

        return bonuses, tags_found
