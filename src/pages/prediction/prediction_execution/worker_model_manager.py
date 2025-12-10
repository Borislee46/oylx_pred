from threading import local

from src.pages.prediction.prediction_model import PredictionModel
from src.utils.logger import setup_logger

prediction_runner_logger = setup_logger("page3", "prediction")

_thread_local = local()


def get_worker_model() -> PredictionModel | None:
    if not hasattr(_thread_local, "model"):
        return None
    return _thread_local.model


def set_worker_model(model: PredictionModel | None) -> None:
    _thread_local.model = model


def init_worker_process(model_type: str) -> None:
    try:
        from src.pages.prediction.page_data_loader import cached_get_prediction_model

        model = cached_get_prediction_model(model_type)
        set_worker_model(model)

        if model is None:
            prediction_runner_logger.error(f"子进程无法加载模型: {model_type}")
    except Exception as e:
        prediction_runner_logger.error(f"初始化子进程失败: {e}", exc_info=True)
        set_worker_model(None)
