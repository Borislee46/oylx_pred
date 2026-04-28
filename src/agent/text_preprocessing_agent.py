from typing import Any

from src.agent.base_agent import BaseAgent
from src.agent.text_preprocessing_prompts import build_field_validation_prompt
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
        return {"field_type": field_type, "is_valid": is_valid}
