import logging
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel

from src.agent.base_agent import BaseAgent
from src.agent.persistent_cache import PersistentCache


class MajorCategoryDecision(BaseModel):
    canonical_major: str = Field(
        description="匹配到的规范专业名称（从候选集中选择，或'未知'表示无法匹配）"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="匹配置信度。high=专业名明确对应，medium=基本可判断，low=难以确定"
    )
    reasoning: str = Field(default="", description="简短匹配理由（供调试和trace展示，≤80字）")


_MAJOR_CATEGORY_SYSTEM_PROMPT = """你是高校专业名称标准化助手。

目标：将用户输入的"原始专业名称"映射到候选列表中最接近的"规范专业名称"。

规则：
- canonical_major 只能从下方候选集中选择，不要自创
- 如果原始名称和某个候选名称是同义词/缩写/子领域/近似学科 → 选该候选
- 如果原始名称太模糊或与所有候选都无关 → 选 "未知"
- confidence 诚实判断：名称完全相同→high，学科相似但措辞不同→medium，完全无法判断→low
- reasoning 简短说明判断依据（≤80字），用中文
- 不要解释，不要输出多余字段"""


def _build_major_category_agent(model: OpenAIChatModel) -> Agent:
    return Agent(
        model,
        max_concurrency=1,
        output_type=PromptedOutput(MajorCategoryDecision),
        instructions=_MAJOR_CATEGORY_SYSTEM_PROMPT,
        retries=2,
    )


_log = logging.getLogger(__name__)

PROMPT_VERSION = 1


@lru_cache(maxsize=1)
def _get_canonical_majors() -> list[str]:
    from src.pages.prediction.app_data import load_raw_cases_data

    cases = load_raw_cases_data()
    if cases is None or cases.empty or "background_major" not in cases.columns:
        _log.warning("无法加载 background_major 候选集")
        return []
    series = cases["background_major"].dropna().astype(str).map(str.strip)
    values = [x for x in series.unique().tolist() if x and x.lower() not in {"nan", "none"}]
    return sorted(values)


class BackgroundMajorAgent(BaseAgent):
    _cache = PersistentCache("major_category_cache.json")

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config, timeout=10, agent_name="专业名称标准化Agent", logger=_log)
        self._agent = _build_major_category_agent(self._model)

    def _cache_key(self, raw_major: str) -> str:
        return self._hash_cache_key(
            PROMPT_VERSION, self.model, raw_major=str(raw_major or "").strip()
        )

    def resolve_major(
        self,
        raw_major: str,
        use_persistent_cache: bool = True,
    ) -> MajorCategoryDecision:
        name = str(raw_major or "").strip()
        if not name:
            return MajorCategoryDecision(
                canonical_major="未知",
                confidence="low",
                reasoning="专业名为空",
            )

        candidates = _get_canonical_majors()
        if not candidates:
            _log.warning("候选专业集为空，无法解析")
            return MajorCategoryDecision(
                canonical_major="未知",
                confidence="low",
                reasoning="候选专业集不可用",
            )

        if name in candidates:
            return MajorCategoryDecision(
                canonical_major=name,
                confidence="high",
                reasoning="精确匹配",
            )

        cache_key = self._cache_key(name)
        if use_persistent_cache:
            cached = BackgroundMajorAgent._cache.get(cache_key)
            if isinstance(cached, str) and cached in candidates:
                _log.info("专业缓存命中 | raw=%s canonical=%s", name, cached)
                return MajorCategoryDecision(
                    canonical_major=cached,
                    confidence="high",
                    reasoning=f"缓存命中: {cached}",
                )
            elif isinstance(cached, str):
                _log.warning("专业缓存值无效 | raw=%s cached=%s", name, cached)

        prompt = self._build_prompt(name, candidates)
        _log.debug("专业解析API调用 | raw=%s prompt_len=%d", name, len(prompt))

        try:
            result = self._agent.run_sync(prompt)
            output = result.output
            if not isinstance(output, MajorCategoryDecision):
                _log.warning("专业解析返回非预期类型: %s", type(output))
                return MajorCategoryDecision(
                    canonical_major="未知",
                    confidence="low",
                    reasoning="LLM返回非预期类型",
                )
        except Exception:
            _log.warning("专业解析API调用失败 | raw=%s", name, exc_info=True)
            return MajorCategoryDecision(
                canonical_major="未知",
                confidence="low",
                reasoning="API调用失败",
            )

        llm_output = str(output.canonical_major or "").strip()
        confidence = output.confidence
        reasoning = output.reasoning or ""

        if confidence == "low":
            return MajorCategoryDecision(
                canonical_major="未知",
                confidence="low",
                reasoning=f"AI匹配为'{llm_output}'但置信度低: {reasoning}",
            )

        matched = self._fuzzy_validate(llm_output, candidates)
        if matched is None:
            _log.warning(
                "专业LLM输出不在候选集 | raw=%s llm_output=%s",
                name,
                llm_output,
            )
            return MajorCategoryDecision(
                canonical_major="未知",
                confidence="low",
                reasoning=f"输出'{llm_output}'不在候选集中",
            )

        if use_persistent_cache:
            BackgroundMajorAgent._cache.set(cache_key, matched)

        _log.info(
            "专业解析完成 | raw=%s → canonical=%s confidence=%s",
            name,
            matched,
            confidence,
        )
        return MajorCategoryDecision(
            canonical_major=matched,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _build_prompt(self, raw_major: str, candidates: list[str]) -> str:
        candidates_str = "\n".join(f"  - {c}" for c in candidates)
        return (
            f"原始专业名称: {raw_major}\n\n"
            f"候选规范专业名称（只能从中选择）:\n{candidates_str}\n\n"
            f"请将原始专业名称映射到最接近的候选规范专业名称。"
        )

    @staticmethod
    def _fuzzy_validate(llm_output: str, candidates: list[str]) -> str | None:
        if not llm_output:
            return None

        if llm_output in candidates:
            return llm_output

        for c in candidates:
            if c in llm_output or llm_output in c:
                return c

        return None

    def run(self, context: Any = None, **kwargs: Any) -> dict[str, Any]:
        raw_major = str(kwargs.get("raw_major", "") or "")
        if not raw_major and hasattr(context, "background_major"):
            raw_major = str(context.background_major or "")

        decision = self.resolve_major(
            raw_major=raw_major,
            use_persistent_cache=bool(kwargs.get("use_persistent_cache", True)),
        )

        return {
            "canonical_major": decision.canonical_major,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        }
