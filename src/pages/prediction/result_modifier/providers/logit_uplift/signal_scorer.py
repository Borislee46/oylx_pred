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
    def __init__(
        self,
        lexicon_path: str | None,
        enabled_fields: tuple[str, ...] | None,
        per_field_cap: float,
        max_reasons: int,
        lexicon_weight: float,
    ) -> None:
        self._lexicon_path = lexicon_path
        self._enabled_fields = enabled_fields
        self._per_field_cap = float(per_field_cap)
        self._max_reasons = int(max_reasons)
        self._lexicon_weight = float(lexicon_weight)
        self._rules = self._load_rules(lexicon_path)

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
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
        except Exception:
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

    def score(self, texts_by_field: dict[str, str]) -> tuple[dict[str, float], tuple[str, ...]]:
        if not texts_by_field or not self._rules or self._lexicon_weight <= 0:
            return {}, ()

        enabled = set(self._enabled_fields) if self._enabled_fields else None

        bonuses: dict[str, float] = {}
        reasons: list[tuple[float, str]] = []

        for field, text in texts_by_field.items():
            if enabled is not None and field not in enabled:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            t = text.lower()
            best = 0.0
            for rule in self._rules:
                if rule.fields is not None and field not in rule.fields:
                    continue
                if rule.pattern in t:
                    w = rule.score
                    if w > best:
                        best = w
                    reasons.append((w, rule.tag))
            if best > 0:
                bonuses[field] = min(self._per_field_cap, best * self._lexicon_weight)

        if not reasons:
            return bonuses, ()

        reasons.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        picked: list[str] = []
        for _, tag in reasons:
            if tag in seen:
                continue
            seen.add(tag)
            picked.append(tag)
            if len(picked) >= self._max_reasons:
                break
        return bonuses, tuple(picked)
