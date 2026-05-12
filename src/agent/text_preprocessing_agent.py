from typing import Any

from src.agent.base_agent import BaseAgent
from src.agent.text_preprocessing_prompts import (
    build_batch_validation_prompt,
    build_field_validation_prompt,
)
from src.agent.utils import parse_bool


class TextPreprocessingAgent(BaseAgent):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=10, agent_name="文本预处理Agent")

    def validate_field(self, field_type: str, content: str, default_on_error: bool = False) -> bool:
        if not content or not content.strip():
            return False

        prompt = build_field_validation_prompt(field_type, content)
        content_response = self._call_api(prompt, cache_prefix="field_validation", use_cache=True)

        if content_response is None:
            return default_on_error

        is_valid = parse_bool(content_response, default=default_on_error)

        self.logger.info(
            f"[{self.agent_name}] 字段验证完成 - 字段类型: {field_type}, 验证结果: {is_valid}"
        )
        return is_valid

    def validate_fields_batch(self, fields: dict[str, str]) -> dict[str, bool]:
        """Validate multiple experience fields in one API call.

        Returns a dict mapping each field key to its boolean validity.
        Missing/empty fields default to False.
        """
        valid: dict[str, bool] = dict.fromkeys(fields, False)
        has_content = {k: v for k, v in fields.items() if v and v.strip()}
        if not has_content:
            return valid

        prompt = build_batch_validation_prompt(has_content)
        if not prompt:
            return valid

        raw = self._call_api(prompt, cache_prefix="field_validation_batch", use_cache=True)
        if raw is None:
            return valid

        result = self._parse_json_response(raw, schema_hint='{"<key>": true/false}')
        if not isinstance(result, dict):
            return valid

        for key in fields:
            val = result.get(key)
            if isinstance(val, bool):
                valid[key] = val
            elif isinstance(val, str):
                valid[key] = parse_bool(val, default=False)

        self.logger.info(
            f"[{self.agent_name}] 批量字段验证完成 - 字段数: {len(has_content)}, "
            f"有效: {sum(1 for v in valid.values() if v)}"
        )
        return valid

    def run(self, context: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Orchestrator-compatible entry point.

        Expects kwargs:
            field_type: str  ("research" | "internship" | "paper" | "award")
            content: str
        Or reads from context.experience_details by field_type.
        """
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
