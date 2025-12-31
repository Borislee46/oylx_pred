from __future__ import annotations

import json
from typing import Any

from src.pages.prediction.result_modifier.providers.logit_uplift.utils import safe_float


class TextProcessor:
    """
    背景文本预处理器。

    负责将用户的输入字典标准化为模型可识别的格式，并生成缓存签名。
    """

    def __init__(
        self,
        text_keys: tuple[str, ...],
        count_keys: tuple[str, ...],
    ) -> None:
        """
        Args:
            text_keys: 文本描述字段名列表（如 research_details）。
            count_keys: 经历数量字段名列表（如 research_count）。
        """
        self._text_keys = text_keys
        self._count_keys = count_keys

    @staticmethod
    # 只做去空格，先不用分词器
    def prep_text(s: str | None) -> str:
        if not isinstance(s, str):
            return ""
        return s.strip()

    def make_signature(self, details: dict[str, Any]) -> str:
        """
        为输入数据生成唯一的 JSON 签名。

        用于 `lru_cache`。通过 `sort_keys=True` 确保即使字典顺序不同，
        相同的输入内容也会生成相同的签名。
        """
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
