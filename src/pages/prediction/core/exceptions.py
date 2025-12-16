from typing import Any


class PredictionError(Exception):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | 详情: {self.details}"
        return self.message


class ModelError(PredictionError):
    pass


class ModelLoadError(ModelError):
    def __init__(self, model_name: str, reason: str = ""):
        message = f"模型加载失败: {model_name}"
        super().__init__(message, {"model_name": model_name, "reason": reason})


class ModelPredictionError(ModelError):
    def __init__(self, message: str = "模型预测失败", batch_size: int | None = None):
        details = {}
        if batch_size is not None:
            details["batch_size"] = batch_size
        super().__init__(message, details)


class InputError(PredictionError):
    pass


class InvalidInputError(InputError):
    def __init__(self, field: str, value: Any = None, expected: str = ""):
        message = f"无效输入: {field}"
        details = {"field": field}
        if value is not None:
            details["value"] = str(value)[:100]
        if expected:
            details["expected"] = expected
        super().__init__(message, details)


class MissingInputError(InputError):
    def __init__(self, fields: list[str]):
        message = f"缺少必要输入: {', '.join(fields)}"
        super().__init__(message, {"missing_fields": fields})


class DataError(PredictionError):
    pass


class DataValidationError(DataError):
    def __init__(self, message: str = "数据验证失败", validation_errors: list[str] | None = None):
        details = {}
        if validation_errors:
            details["errors"] = validation_errors
        super().__init__(message, details)


class DataLoadError(DataError):
    def __init__(self, data_source: str, reason: str = ""):
        message = f"数据加载失败: {data_source}"
        super().__init__(message, {"data_source": data_source, "reason": reason})


class CacheError(DataError):
    def __init__(self, operation: str, cache_key: str = ""):
        message = f"缓存{operation}失败"
        super().__init__(message, {"operation": operation, "cache_key": cache_key})


class ProcessingError(PredictionError):
    pass


class PredictionExecutionError(ProcessingError):
    def __init__(self, stage: str, reason: str = ""):
        message = f"预测执行失败: {stage}"
        super().__init__(message, {"stage": stage, "reason": reason})


class AdjustmentError(ProcessingError):
    def __init__(self, adjuster_name: str, reason: str = ""):
        message = f"概率调整失败: {adjuster_name}"
        super().__init__(message, {"adjuster": adjuster_name, "reason": reason})


class FilterError(ProcessingError):
    def __init__(self, filter_name: str, reason: str = ""):
        message = f"过滤操作失败: {filter_name}"
        super().__init__(message, {"filter": filter_name, "reason": reason})


class ConfigError(PredictionError):
    pass


class ConfigLoadError(ConfigError):
    def __init__(self, config_path: str, reason: str = ""):
        message = f"配置加载失败: {config_path}"
        super().__init__(message, {"config_path": config_path, "reason": reason})


def wrap_exception(
    original: Exception, wrapper_class: type[PredictionError], context: str = ""
) -> PredictionError:
    message = f"{context}: {str(original)}" if context else str(original)
    return wrapper_class(message, {"original_type": type(original).__name__})


def is_recoverable(error: PredictionError) -> bool:
    recoverable_types = (CacheError, DataValidationError)
    return isinstance(error, recoverable_types)
