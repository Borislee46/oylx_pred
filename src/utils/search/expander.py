from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput

from src.agent.runtime.model_factory import build_production_model

_log = logging.getLogger(__name__)


class SearchExpansion(BaseModel):
    original: str = ""
    terms: list[str] = Field(
        default_factory=list,
        description="Expanded English search terms (5-10 terms)",
    )


_EXPANDER_PROMPT = """\
你是香港/新加坡/澳门/马来西亚硕士项目数据库的搜索专家。
你的任务: 把用户输入的中文专业名称,扩展为一组英文搜索词,
用于在英文项目名列表中做模糊匹配。

规则:
1. 生成 5-10 个英文搜索词,覆盖该项目可能的英文变体
2. 包含: 直译、学科全称、常见缩写、相关子方向
3. 搜索词按精确度排序: 最可能精准匹配的排最前
4. 只写英文搜索词,不要翻译回中文

示例:
- "语言学" → ["Linguistics", "Applied Linguistics", "Chinese Linguistics",
              "Language Studies", "Translation", "Interpreting",
              "English Language", "Language and Culture"]
- "计算机" → ["Computer Science", "Computing", "Information Technology",
              "Software Engineering", "Data Science", "Artificial Intelligence",
              "CS", "IT"]
- "金融" → ["Finance", "Financial Engineering", "Financial Mathematics",
            "Investment", "Accounting and Finance", "Fintech",
            "Financial Economics"]
- "电子工程" → ["Electronic Engineering", "Electrical and Electronic Engineering",
               "EEE", "Electronics", "Electrical Engineering"]
"""


@lru_cache(maxsize=1)
def _build_expander_agent() -> Agent:
    model = build_production_model(timeout=10)
    return Agent(
        model,
        output_type=PromptedOutput(SearchExpansion),
        instructions=_EXPANDER_PROMPT,
        retries=1,
    )


def build_search_expander() -> Callable[[str], list[str]]:
    agent = _build_expander_agent()

    @lru_cache(maxsize=512)
    def _expand(query: str) -> tuple[str, ...]:
        try:
            result = agent.run_sync(query)
            output = result.output
            if isinstance(output, SearchExpansion) and output.terms:
                _log.debug("SEARCH_EXPAND | query=%s → %d terms", query, len(output.terms))
                return tuple(output.terms)
        except Exception:
            _log.warning("SEARCH_EXPAND | failed for query=%s", query, exc_info=True)

        return (query,)

    return lambda q: list(_expand(q.strip()))
