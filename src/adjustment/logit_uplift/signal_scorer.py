from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability, safe_float

logger = setup_logger("page3", "prediction")


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
        lexicon_weight: float,
    ) -> None:
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

    def _load_rules(self, lexicon_path: str | None) -> tuple[_Rule, ...]:
        if not lexicon_path:
            return ()
        path = Path(lexicon_path)
        if not path.exists():
            return ()
        with open(path, encoding="utf-8") as f:
            obj = json.load(f) or {}

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
            score = safe_float(r.get("score"), 0.0)
            score = clip_probability(score)
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
        logger.info(
            "SignalScorer 加载词库 | n_rules=%d path=%s",
            len(out),
            lexicon_path,
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
                bonuses[field] = min(self._per_field_cap, best * self._lexicon_weight)
                tags_found[field] = list(set(matched_tags))

        if bonuses:
            logger.debug(
                "SignalScorer 含金量命中 | bonuses=%s tags=%s",
                {k: round(v, 4) for k, v in bonuses.items()},
                tags_found,
            )
        return bonuses, tags_found
