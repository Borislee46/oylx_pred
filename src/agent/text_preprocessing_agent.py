from typing import Any

from src.agent.base_agent import BaseAgent
from src.agent.text_preprocessing_prompts import build_field_validation_prompt


class TextPreprocessingAgent(BaseAgent):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=10, agent_name="文本预处理Agent")

    def validate_field(self, field_type: str, content: str) -> bool:
        if not content or not content.strip():
            self.logger.debug(
                f"[{self.agent_name}] 字段验证 - 字段类型: {field_type}, 内容为空，返回 False"
            )
            return False

        prompt = build_field_validation_prompt(field_type, content)

        self.logger.debug(
            f"[{self.agent_name}] 发送字段验证请求 - 字段类型: {field_type}, "
            f"内容长度: {len(content)}"
        )

        content_response = self._call_api(prompt, cache_prefix="field_validation", use_cache=True)
        if content_response is None:
            return False

        s = str(content_response).strip()
        if not s:
            return False
        first = s[0]
        if first == "是":
            is_valid = True
        elif first == "否":
            is_valid = False
        else:
            is_valid = s.startswith("是")

        self.logger.info(
            f"[{self.agent_name}] 字段验证完成 - 字段类型: {field_type}, 验证结果: {is_valid}"
        )
        return is_valid
