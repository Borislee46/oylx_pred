import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel

from src.agent.base_agent import BaseAgent
from src.agent.persistent_cache import PersistentCache

_SCHOOL_LEVELS = Literal[
    "985",
    "1-50",
    "51-100",
    "211",
    "101-200",
    "201-300",
    "301-500",
    "500+",
    "普通本科",
    "专接本",
    "三本/民办本科",
    "专科",
    "未知",
]


class SchoolLevelDecision(BaseModel):
    school_level: _SCHOOL_LEVELS = Field(description="推断的院校层次，从受控枚举中选择")
    confidence: Literal["high", "medium", "low"] = Field(
        description="推断置信度。high=非常确定，medium=基本确定，low=不太确定"
    )
    reasoning: str = Field(default="", description="简短推断理由（供调试和trace展示，≤80字）")


def _get_valid_levels() -> frozenset[str]:
    from src.utils.schools.constants import SCHOOL_LEVEL_PRIORITY

    return frozenset(k for k in SCHOOL_LEVEL_PRIORITY if k is not None)


VALID_SCHOOL_LEVELS: frozenset[str] = _get_valid_levels()

_SCHOOL_LEVEL_SYSTEM_PROMPT = """你是中国高校与海外院校层次识别助手。

目标：根据院校名称判断其办学层次/档次。

中国高校层次体系（从高到低）：
- **985**: 985工程高校，中国最顶尖的39所研究型大学
- **211**: 211工程高校（不含985），约75所
- **普通本科**: 非985/211的公办普通本科院校
- **专接本**: 专升本/专接本/自考本科等非全日制本科
- **三本/民办本科**: 民办本科、独立学院、三本院校
- **专科**: 专科/高职院校

海外院校层次体系（QS/THE世界排名）：
- **1-50**: 世界排名前50顶尖名校
- **51-100**: 世界排名51-100
- **101-200**: 世界排名101-200
- **201-300**: 世界排名201-300
- **301-500**: 世界排名301-500
- **500+**: 世界排名500+

如果完全无法判断，选 **未知**。

要求：
- school_level 只能从受控枚举中选择，不要自创
- confidence 诚实判断：非常确定→high，基本确定→medium，不太确定→low
- reasoning 简短说明判断依据（≤80字），用中文
- 不要解释，不要输出多余字段"""


def _build_school_level_agent(model: OpenAIChatModel) -> Agent:
    return Agent(
        model,
        max_concurrency=1,
        output_type=PromptedOutput(SchoolLevelDecision),
        instructions=_SCHOOL_LEVEL_SYSTEM_PROMPT,
        retries=2,
    )


_log = logging.getLogger(__name__)

PROMPT_VERSION = 1


class SchoolLevelAgent(BaseAgent):
    _cache = PersistentCache("school_level_cache.json")

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config, timeout=10, agent_name="院校层次推断Agent", logger=_log)
        self._agent = _build_school_level_agent(self._model)

    def _cache_key(self, school_name: str) -> str:
        return self._hash_cache_key(
            PROMPT_VERSION, self.model, school_name=str(school_name or "").strip()
        )

    def infer_school_level(
        self,
        school_name: str,
        use_persistent_cache: bool = True,
    ) -> SchoolLevelDecision:
        name = str(school_name or "").strip()
        if not name:
            _log.debug("infer_school_level 跳过（院校名为空）")
            return SchoolLevelDecision(
                school_level="未知",
                confidence="low",
                reasoning="院校名为空",
            )

        cache_key = self._cache_key(name)
        if use_persistent_cache:
            cached = SchoolLevelAgent._cache.get(cache_key)
            if isinstance(cached, str) and cached in VALID_SCHOOL_LEVELS:
                _log.info("院校层次缓存命中 | school=%s level=%s", name, cached)
                return SchoolLevelDecision(
                    school_level=cached,  # type: ignore[arg-type]
                    confidence="high",
                    reasoning=f"缓存命中: {cached}",
                )
            elif isinstance(cached, str):
                _log.warning(
                    "院校层次缓存值无效 | school=%s cached=%s",
                    name,
                    cached,
                )

        prompt = self._build_prompt(name)
        _log.debug("院校层次API调用 | school=%s prompt_len=%d", name, len(prompt))

        try:
            result = self._agent.run_sync(prompt)
            output = result.output
            if not isinstance(output, SchoolLevelDecision):
                _log.warning(
                    "院校层次返回非预期类型: %s",
                    type(output),
                )
                return SchoolLevelDecision(
                    school_level="未知",
                    confidence="low",
                    reasoning="LLM返回非预期类型",
                )
        except Exception:
            _log.warning(
                "院校层次API调用失败 | school=%s",
                name,
                exc_info=True,
            )
            return SchoolLevelDecision(
                school_level="未知",
                confidence="low",
                reasoning="API调用失败",
            )

        level = output.school_level
        confidence = output.confidence
        reasoning = output.reasoning or ""

        if level not in VALID_SCHOOL_LEVELS:
            _log.warning(
                "院校层次不在受控枚举 | school=%s level=%s → 回退为'未知'",
                name,
                level,
            )
            return SchoolLevelDecision(
                school_level="未知",
                confidence="low",
                reasoning=f"输出层次'{level}'不在受控枚举中",
            )

        if confidence == "low":
            _log.info(
                "院校层次置信度低 | school=%s level=%s → 回退为'未知'",
                name,
                level,
            )
            return SchoolLevelDecision(
                school_level="未知",
                confidence="low",
                reasoning=f"AI推断为'{level}'但置信度低: {reasoning}",
            )

        if use_persistent_cache:
            SchoolLevelAgent._cache.set(cache_key, level)

        _log.info(
            "院校层次推断完成 | school=%s level=%s confidence=%s",
            name,
            level,
            confidence,
        )
        return output

    def _build_prompt(self, school_name: str) -> str:
        return f"请判断以下院校的办学层次：\n\n院校名称: {school_name}"

    def run(self, context: Any = None, **kwargs: Any) -> dict[str, Any]:
        school_name = str(kwargs.get("school_name", "") or "")
        if not school_name and hasattr(context, "school_name"):
            school_name = str(context.school_name or "")

        decision = self.infer_school_level(
            school_name=school_name,
            use_persistent_cache=bool(kwargs.get("use_persistent_cache", True)),
        )

        return {
            "school_level": decision.school_level,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        }
