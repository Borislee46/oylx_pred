import logging
import threading
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel

from src.agent.prompts.text_preprocessing import (
    build_batch_validation_prompt,
    build_field_validation_prompt,
    build_quality_verification_prompt_cached,
)
from src.agent.runtime.model_factory import build_model_with_fallback


class BatchValidationResult(BaseModel):
    research_details: bool = False
    internship_details: bool = False
    paper_details: bool = False
    award_details: bool = False


class DowngradedTag(BaseModel):
    tag: str
    reason: str


class FieldQualityResult(BaseModel):
    is_valid: bool
    verified_tags: list[str] = Field(default_factory=list)
    downgraded_tags: list[DowngradedTag] = Field(default_factory=list)
    missed_signals: list[str] = Field(default_factory=list)
    quality_level: str = "medium"
    concern: str | None = None


class QualityVerificationResult(BaseModel):
    research_details: FieldQualityResult | None = None
    internship_details: FieldQualityResult | None = None
    paper_details: FieldQualityResult | None = None
    award_details: FieldQualityResult | None = None


_BOOL_AGENT_SYSTEM_PROMPT = """你是一个文本内容验证助手。
你的任务是判断给定的文本内容是否与指定字段类型相关。
回答 true 表示相关/有效，false 表示不相关/无效/空内容。
只返回 true 或 false，不要添加解释。"""

_BATCH_AGENT_SYSTEM_PROMPT = """你是一个文本内容验证助手。
你的任务是判断多个经历字段是否包含实质性信息（非占位符/无效内容）。
对每个字段返回 true（有实质内容）或 false（无效/占位符/空）。
只返回 JSON，不要添加解释。"""

_QUALITY_AGENT_SYSTEM_PROMPT = """你是一位学术背景审核专家。
你的任务是对经历文本做含金量校验——判断关键词匹配是否准确，
是否有遗漏的高含金量信号，以及是否存在虚假或夸大内容。

评估维度（per-field）：
- is_valid: 文本是否确实属于该字段类型
- verified_tags: 从词典标签中确认可靠的
- downgraded_tags: 应降级/删除的词典标签及原因
- missed_signals: 词典漏掉但文本中确实存在的含金量信号
- quality_level: "high" / "medium" / "low" / "invalid"
- concern: 不一致或可疑之处，简要说明；无则 null

判断原则：
- 顶刊(Nature/Science/Cell)/顶会/大厂核心岗/国奖 = high
- 有实质但非顶级 = medium
- 空洞/边缘岗位/疑似注水 = low
- 完全无关/明显虚假 = invalid
- 词典标签 ≠ 含金量（如 Nature出版社实习的Nature标签应降级）
- 背景不一致标记但不直接判否（留给人工复核）
- missed_signals 用简洁短语描述，每条≤20字

只返回 JSON，不要添加解释。"""


def _build_text_preprocessing_bool_agent(model: OpenAIChatModel) -> Agent:
    return Agent(
        model,
        max_concurrency=1,
        output_type=bool,
        instructions=_BOOL_AGENT_SYSTEM_PROMPT,
        retries=2,
    )


def _build_text_preprocessing_batch_agent(model: OpenAIChatModel) -> Agent:
    return Agent(
        model,
        max_concurrency=1,
        output_type=PromptedOutput(BatchValidationResult),
        instructions=_BATCH_AGENT_SYSTEM_PROMPT,
        retries=2,
    )


def _build_text_preprocessing_quality_agent(model: OpenAIChatModel) -> Agent:
    return Agent(
        model,
        max_concurrency=1,
        output_type=PromptedOutput(QualityVerificationResult),
        instructions=_QUALITY_AGENT_SYSTEM_PROMPT,
        retries=2,
    )


_log = logging.getLogger(__name__)

PROMPT_VERSION = 1


class TextPreprocessingAgent:
    def __init__(self, config: dict[str, Any] | None = None):
        _ = config
        timeout = 10
        quality_timeout = 30

        self._model = build_model_with_fallback(timeout=timeout)
        self._agent_bool = _build_text_preprocessing_bool_agent(self._model)
        self._agent_batch = _build_text_preprocessing_batch_agent(self._model)

        self._quality_model = build_model_with_fallback(timeout=quality_timeout)
        self._agent_quality = _build_text_preprocessing_quality_agent(self._quality_model)

        self.agent_name = "文本预处理Agent"
        self.logger = _log
        self._batch_cache: dict[str, dict[str, bool]] = {}
        self._cache_lock = threading.Lock()

    def validate_field(self, field_type: str, content: str, default_on_error: bool = False) -> bool:
        if not content or not content.strip():
            _log.debug("字段校验跳过(空内容) | field=%s", field_type)
            return False

        _log.info("单字段校验开始 | field=%s content_len=%d", field_type, len(content))
        prompt = build_field_validation_prompt(field_type, content)

        try:
            result = self._agent_bool.run_sync(prompt)
            is_valid = bool(result.output)
        except Exception:
            _log.warning(
                "单字段校验失败 | field=%s default=%s",
                field_type,
                default_on_error,
                exc_info=True,
            )
            return default_on_error

        _log.info("单字段校验完成 | field=%s is_valid=%s", field_type, is_valid)
        return is_valid

    _BATCH_CACHE_MAX = 256

    def validate_fields_batch(self, fields: dict[str, str]) -> dict[str, bool]:
        valid: dict[str, bool] = dict.fromkeys(fields, False)
        has_content = {k: v for k, v in fields.items() if v and v.strip()}
        if not has_content:
            _log.debug("批量校验跳过(无内容)")
            return valid

        cache_key = "|".join(f"{k}={v}" for k, v in sorted(has_content.items()))
        with self._cache_lock:
            cached = self._batch_cache.get(cache_key)
        if cached is not None:
            _log.info("批量校验缓存命中 | fields=%d", len(has_content))
            return cached

        _log.info("批量校验开始 | fields=%d non_empty=%d", len(fields), len(has_content))
        prompt = build_batch_validation_prompt(has_content)
        if not prompt:
            _log.warning("批量校验prompt为空")
            return valid

        _log.debug(
            "批量校验API调用 | fields=%s prompt_len=%d",
            list(has_content.keys()),
            len(prompt),
        )
        cacheable = False
        try:
            result = self._agent_batch.run_sync(prompt)
            output = result.output
            if isinstance(output, BatchValidationResult):
                valid = {
                    "research_details": output.research_details,
                    "award_details": output.award_details,
                    "internship_details": output.internship_details,
                    "paper_details": output.paper_details,
                }
                cacheable = True
            else:
                _log.warning("批量校验返回非预期类型: %s", type(output))
        except Exception:
            _log.warning("批量校验API调用失败", exc_info=True)

        if cacheable:
            with self._cache_lock:
                if len(self._batch_cache) >= self._BATCH_CACHE_MAX:
                    oldest = next(iter(self._batch_cache))
                    del self._batch_cache[oldest]
                self._batch_cache[cache_key] = valid

        valid_count = sum(1 for v in valid.values() if v)
        _log.info(
            "批量校验完成 | total=%d valid=%d invalid=%d",
            len(has_content),
            valid_count,
            len(has_content) - valid_count,
        )
        return valid

    def validate_quality_batch(
        self,
        fields: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        has_content = {
            k: v
            for k, v in fields.items()
            if isinstance(v, dict) and str(v.get("text", "") or "").strip()
        }
        if not has_content:
            _log.debug("含金量校验跳过(无内容)")
            return {}

        _log.info("含金量校验开始 | fields=%d", len(has_content))
        prompt = build_quality_verification_prompt_cached(has_content)
        if not prompt:
            _log.warning("含金量校验prompt为空")
            return {}

        _log.debug("含金量校验API调用 | prompt_len=%d", len(prompt))
        try:
            result = self._agent_quality.run_sync(prompt)
            output = result.output
            if not isinstance(output, QualityVerificationResult):
                _log.warning("含金量校验返回非预期类型: %s", type(output))
                return {}

            out: dict[str, dict[str, object]] = {}
            for key in (
                "research_details",
                "award_details",
                "internship_details",
                "paper_details",
            ):
                if key not in has_content:
                    continue
                field_result = getattr(output, key, None)
                if isinstance(field_result, FieldQualityResult):
                    out[key] = field_result.model_dump()
        except Exception:
            _log.warning("含金量校验API调用失败", exc_info=True)
            return {}

        verified_count = sum(
            1
            for v in out.values()
            if isinstance(v.get("verified_tags"), list) and v["verified_tags"]
        )
        downgraded_count = sum(
            1
            for v in out.values()
            if isinstance(v.get("downgraded_tags"), list) and v["downgraded_tags"]
        )
        _log.info(
            "含金量校验完成 | fields=%d verified=%d downgraded=%d",
            len(out),
            verified_count,
            downgraded_count,
        )
        return out

    def run(self, context: Any = None, **kwargs: Any) -> dict[str, Any]:
        from src.agent.context import StudentContext

        field_type = str(kwargs.get("field_type", ""))
        content = kwargs.get("content", "")
        if not content and isinstance(context, StudentContext):
            exp = context.experience_details or {}
            content = exp.get(field_type, "")
        if not content and isinstance(context, StudentContext):
            content = context.extracted_background.get(field_type, "")
        content = str(content or "")

        is_valid = self.validate_field(
            field_type=field_type,
            content=content,
            default_on_error=bool(kwargs.get("default_on_error", False)),
        )
        result: dict[str, Any] = {"field_type": field_type, "is_valid": is_valid}
        if not is_valid and content:
            result["_error"] = "api_failed_or_invalid"
        return result
