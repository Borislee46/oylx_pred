from __future__ import annotations

import json
from typing import Any

from src.utils.numeric import safe_float


class TextProcessor:
    def __init__(
        self,
        text_keys: tuple[str, ...],
        count_keys: tuple[str, ...],
    ) -> None:
        self._text_keys = text_keys
        self._count_keys = count_keys

    @staticmethod
    def prep_text(s: str | None) -> str:
        if not isinstance(s, str):
            return ""
        return s.strip()

    def make_signature(self, details: dict[str, Any]) -> str:
        obj: dict[str, Any] = {k: self.prep_text(str(details.get(k, ""))) for k in self._text_keys}
        for k in self._count_keys:
            obj[k] = int(safe_float(details.get(k, 0), 0))
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)

    @property
    def text_keys(self) -> tuple[str, ...]:
        return self._text_keys

    @property
    def count_keys(self) -> tuple[str, ...]:
        return self._count_keys
