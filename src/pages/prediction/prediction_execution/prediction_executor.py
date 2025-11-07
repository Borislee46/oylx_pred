import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from src.pages.prediction.prediction_execution.prediction_chunk_executor import (
    run_prediction_chunk,
    run_prediction_chunk_in_process,
)
from src.pages.prediction.prediction_execution.worker_model_manager import (
    init_worker_process,
)
from src.pages.prediction.prediction_model import PredictionModel
from src.utils.logger import setup_logger

prediction_runner_logger = setup_logger("page3", "prediction")


class PredictionExecutor:
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.cpu_count = os.cpu_count() or 2

    def get_execution_strategy(self) -> Tuple[Optional[type], int, int]:
        if self.total_tasks < 128:
            return None, 1, self.total_tasks

        if self.total_tasks < 512:
            return None, 1, self.total_tasks

        use_process_pool = os.getenv("PREDICTION_USE_PROCESS_POOL", "0") == "1"
        max_workers_env = int(os.getenv("PREDICTION_MAX_WORKERS", "0"))

        if self.cpu_count <= 2:
            num_workers = 2
        else:
            num_workers = min(4, self.cpu_count)

        executor_class = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor

        if max_workers_env > 0:
            num_workers = min(num_workers, max_workers_env)

        chunk_size = max(64, (self.total_tasks + num_workers - 1) // num_workers)

        return executor_class, num_workers, chunk_size

    def execute_parallel(
        self,
        prediction_model: PredictionModel,
        combinations: list[tuple[str, str]],
        model_input_features: dict[str, float | int | str],
        expected_features: list[str],
    ) -> List[dict[str, float | str]]:
        executor_class, num_workers, chunk_size = self.get_execution_strategy()

        chunks = [combinations[i : i + chunk_size] for i in range(0, len(combinations), chunk_size)]

        if executor_class is None:
            return self._execute_single_threaded(
                prediction_model, chunks, model_input_features, expected_features
            )

        if executor_class == ProcessPoolExecutor:
            result = self._execute_with_process_pool(
                prediction_model, chunks, model_input_features, expected_features
            )
            if result is not None:
                return result
            prediction_runner_logger.warning("进程池执行失败，回退到线程池")

        return self._execute_with_thread_pool(
            prediction_model,
            chunks,
            model_input_features,
            expected_features,
            num_workers,
        )

    def _execute_single_threaded(
        self,
        prediction_model: PredictionModel,
        chunks: List[list[tuple[str, str]]],
        model_input_features: dict[str, float | int | str],
        expected_features: list[str],
    ) -> List[dict[str, float | str]]:
        results = []
        for chunk in chunks:
            chunk_results = run_prediction_chunk(
                prediction_model, model_input_features, chunk, expected_features
            )
            if chunk_results:
                results.extend(chunk_results)
        return results

    def _execute_with_process_pool(
        self,
        prediction_model: PredictionModel,
        chunks: List[list[tuple[str, str]]],
        model_input_features: dict[str, float | int | str],
        expected_features: list[str],
    ) -> Optional[List[dict[str, float | str]]]:
        try:
            with ProcessPoolExecutor(
                max_workers=len(chunks),
                initializer=init_worker_process,
                initargs=(prediction_model.model_type,),
            ) as executor:
                return self._collect_results(
                    executor, chunks, model_input_features, expected_features
                )
        except Exception as e:
            prediction_runner_logger.error(f"进程池执行失败: {e}", exc_info=True)
            return None

    def _execute_with_thread_pool(
        self,
        prediction_model: PredictionModel,
        chunks: List[list[tuple[str, str]]],
        model_input_features: dict[str, float | int | str],
        expected_features: list[str],
        num_workers: int,
    ) -> List[dict[str, float | str]]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return self._collect_results(
                executor,
                chunks,
                model_input_features,
                expected_features,
                prediction_model,
            )

    def _collect_results(
        self,
        executor: ProcessPoolExecutor | ThreadPoolExecutor,
        chunks: List[list[tuple[str, str]]],
        model_input_features: dict[str, float | int | str],
        expected_features: list[str],
        prediction_model: Optional[PredictionModel] = None,
    ) -> List[dict[str, float | str]]:
        results = []

        if prediction_model:
            futures = [
                executor.submit(
                    run_prediction_chunk,
                    prediction_model,
                    model_input_features,
                    chunk,
                    expected_features,
                )
                for chunk in chunks
                if chunk
            ]
        else:
            futures = [
                executor.submit(
                    run_prediction_chunk_in_process,
                    model_input_features,
                    chunk,
                    expected_features,
                )
                for chunk in chunks
                if chunk
            ]

        for future in as_completed(futures):
            try:
                chunk_results = future.result(timeout=120)
                if chunk_results:
                    results.extend(chunk_results)
            except Exception as e:
                prediction_runner_logger.error(f"子任务执行失败: {e}", exc_info=True)

        return results
