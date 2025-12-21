import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, List, Dict, Tuple, Optional

from src.pages.prediction.modeling.model import PredictionModel
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

_WORKER_MODEL: Optional[PredictionModel] = None

def _init_worker_process(model_type: str) -> None:
    global _WORKER_MODEL
    try:
        from src.pages.prediction.page_data_loader import cached_get_prediction_model
        _WORKER_MODEL = cached_get_prediction_model(model_type)
    except Exception as e:
        logger.error(f"子进程初始化失败: {e}")
        _WORKER_MODEL = None

def _run_chunk_in_worker(
    model_input: Dict[str, Any],
    chunk: List[Tuple[str, str]],
    features: List[str]
) -> List[Dict[str, Any]]:
    if _WORKER_MODEL is None:
        raise RuntimeError("Worker model not initialized")
    return _WORKER_MODEL.predict_batch(model_input, chunk, features)


class PredictionExecutor:
    
    def __init__(self, total_tasks: int):
        self.total_tasks = max(0, int(total_tasks))
        self.cpu_count = os.cpu_count() or 2
        
        self.single_threshold = int(os.getenv("PREDICTION_SINGLE_THREAD_THRESHOLD", "256"))
        self.min_chunk_size = int(os.getenv("PREDICTION_MIN_CHUNK_SIZE", "64"))
        self.use_process_pool = os.getenv("PREDICTION_USE_PROCESS_POOL", "0").lower() in ("1", "true")
        self.timeout = int(os.getenv("PREDICTION_OVERALL_TIMEOUT_SEC", "300"))

    def execute_parallel(
        self,
        model: PredictionModel,
        combinations: List[Tuple[str, str]],
        model_input: Dict[str, Any],
        features: List[str],
    ) -> List[Dict[str, Any]]:
        if not combinations:
            return []

        if self.total_tasks <= self.single_threshold:
            return model.predict_batch(model_input, combinations, features)

        num_workers = min(4, self.cpu_count)
        chunk_size = max(self.min_chunk_size, (len(combinations) + num_workers - 1) // num_workers)
        chunks = [combinations[i : i + chunk_size] for i in range(0, len(combinations), chunk_size)]

        if self.use_process_pool:
            return self._execute_with_pool(ProcessPoolExecutor, chunks, model, model_input, features, num_workers) or \
                   self._execute_with_pool(ThreadPoolExecutor, chunks, model, model_input, features, num_workers)
        
        return self._execute_with_pool(ThreadPoolExecutor, chunks, model, model_input, features, num_workers)

    def _execute_with_pool(
        self,
        pool_class: Any,
        chunks: List[List[Tuple[str, str]]],
        model: PredictionModel,
        model_input: Dict[str, Any],
        features: List[str],
        num_workers: int
    ) -> Optional[List[Dict[str, Any]]]:
        results = []
        is_process = pool_class == ProcessPoolExecutor
        
        try:
            pool_kwargs = {"max_workers": num_workers}
            if is_process:
                pool_kwargs.update({
                    "initializer": _init_worker_process,
                    "initargs": (model.model_type,)
                })

            with pool_class(**pool_kwargs) as executor:
                if is_process:
                    futures = [executor.submit(_run_chunk_in_worker, model_input, c, features) for c in chunks]
                else:
                    futures = [executor.submit(model.predict_batch, model_input, c, features) for c in chunks]

                for future in as_completed(futures, timeout=self.timeout):
                    chunk_res = future.result()
                    if chunk_res:
                        results.extend(chunk_res)
            
            return results
        except Exception as e:
            logger.error(f"{pool_class.__name__} 执行出错: {e}", exc_info=True)
            return None

