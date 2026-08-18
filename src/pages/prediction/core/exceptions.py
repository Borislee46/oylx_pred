import logging
from typing import Any

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        logger.debug(
            "%s | 详情: %s",
            self.__class__.__name__,
            self.details,
        )

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | 详情: {self.details}"
        return self.message


class InputError(PredictionError):
    """用户输入层异常的基类——表单校验、字段缺失、格式错误。"""


class MissingInputError(InputError):
    def __init__(self, fields: list[str]):
        message = f"缺少必要输入: {', '.join(fields)}"
        super().__init__(message, {"missing_fields": fields})
        logger.warning("缺少必要输入 | fields=%s", fields)
