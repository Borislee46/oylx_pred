import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError

from src.pages.prediction.modeling.model import PredictionModel
from src.pages.prediction.prediction_execution.prediction_chunk_executor import (
    run_prediction_chunk,
    run_prediction_chunk_in_process,
)
from src.pages.prediction.prediction_execution.worker_model_manager import (
    init_worker_process,
)
from src.utils.logger import setup_logger

prediction_runner_logger = setup_logger("page3", "prediction")


class PredictionExecutor:
    def __init__(self, total_tasks: int):
        self.total_tasks = max(0, int(total_tasks))
        self.cpu_count = os.cpu_count() or 2

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        v = os.getenv(name)
        if v is None:
            return default
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        v = os.getenv(name)
        if v is None:
            return default
        try:
            return int(v)
        except ValueError:
            return default

    def get_execution_strategy(self) -> tuple[type | None, int, int]:
        single_threshold = self._env_int("PREDICTION_SINGLE_THREAD_THRESHOLD", 256)
        if self.total_tasks <= single_threshold:
            return None, 1, self.total_tasks

        use_process_pool = self._env_bool("PREDICTION_USE_PROCESS_POOL", False)
        max_workers_env = self._env_int("PREDICTION_MAX_WORKERS", 0)

        if self.cpu_count <= 2:
            num_workers = 2
        else:
            num_workers = min(4, self.cpu_count)

        executor_class = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor

        if max_workers_env > 0:
            num_workers = min(num_workers, max_workers_env)

        min_chunk_size = self._env_int("PREDICTION_MIN_CHUNK_SIZE", 64)
        chunk_size = max(min_chunk_size, (self.total_tasks + num_workers - 1) // num_workers)

        return executor_class, num_workers, chunk_size

    def execute_parallel(
        self,
        prediction_model: PredictionModel,
        combinations: list[tuple[str, str]],
        model_input_features: dict[str, float | int | str],
        expected_features: list[str],
    ) -> list[dict[str, float | str]]:
        executor_class, num_workers, chunk_size = self.get_execution_strategy()

        chunks = [combinations[i : i + chunk_size] for i in range(0, len(combinations), chunk_size)]

        if executor_class is None:
            return self._execute_single_threaded(
                prediction_model, chunks, model_input_features, expected_features
            )

        if executor_class == ProcessPoolExecutor:
            result = self._execute_with_process_pool(
                prediction_model.model_type,
                chunks,
                model_input_features,
                expected_features,
                num_workers,
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
        chunks: list[list[tuple[str, str]]],
        model_input_features: dict[str, float | int | str],
        expected_features: list[str],
    ) -> list[dict[str, float | str]]:
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
        model_type: str,
        chunks: list[list[tuple[str, str]]],
        model_input_features: dict[str, float | int | str],
        expected_features: list[str],
        num_workers: int,
    ) -> list[dict[str, float | str]] | None:
        try:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=init_worker_process,
                initargs=(model_type,),
            ) as executor:
                return self._collect_results(
                    executor,
                    chunks,
                    model_input_features,
                    expected_features,
                    total_tasks=self.total_tasks,
                )
        except Exception as e:
            prediction_runner_logger.error(f"进程池执行失败: {e}", exc_info=True)
            return None

    def _execute_with_thread_pool(
        self,
        prediction_model: PredictionModel,
        chunks: list[list[tuple[str, str]]],
        model_input_features: dict[str, float | int | str],
        expected_features: list[str],
        num_workers: int,
    ) -> list[dict[str, float | str]]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return self._collect_results(
                executor,
                chunks,
                model_input_features,
                expected_features,
                prediction_model,
                total_tasks=self.total_tasks,
            )

    def _collect_results(
        self,
        executor: ProcessPoolExecutor | ThreadPoolExecutor,
        chunks: list[list[tuple[str, str]]],
        model_input_features: dict[str, float | int | str],
        expected_features: list[str],
        prediction_model: PredictionModel | None = None,
        total_tasks: int = 0,
    ) -> list[dict[str, float | str]]:
        results = []
        submitted = 0
        failed = 0
        ok = 0

        fail_fast = self._env_bool("PREDICTION_FAIL_FAST", False)
        overall_timeout_sec = self._env_int("PREDICTION_OVERALL_TIMEOUT_SEC", 300)
        overall_timeout_sec = overall_timeout_sec if overall_timeout_sec > 0 else 0
        started_at = time.monotonic()

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

        submitted = len(futures)
        if submitted == 0:
            return []

        try:
            iterator = (
                as_completed(futures, timeout=overall_timeout_sec)
                if overall_timeout_sec
                else as_completed(futures)
            )
            for future in iterator:
                try:
                    chunk_results = future.result()
                    if chunk_results:
                        results.extend(chunk_results)
                    ok += 1
                except Exception as e:
                    failed += 1
                    prediction_runner_logger.error(f"子任务执行失败: {e}", exc_info=True)
                    if fail_fast:
                        raise
        except FuturesTimeoutError:
            elapsed = time.monotonic() - started_at
            cancelled = 0
            for f in futures:
                if not f.done() and f.cancel():
                    cancelled += 1
            prediction_runner_logger.error(
                f"并行预测超时: elapsed={elapsed:.2f}s, submitted={submitted}, ok={ok}, failed={failed}, cancelled={cancelled}"
            )
            if fail_fast:
                raise

        if total_tasks and len(results) < total_tasks:
            prediction_runner_logger.warning(
                f"预测结果数量不足: expected~={total_tasks}, got={len(results)}, submitted={submitted}, ok={ok}, failed={failed}"
            )
        return results
