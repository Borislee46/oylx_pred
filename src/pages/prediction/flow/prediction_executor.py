import os
import time
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any

from src.pages.prediction.modeling import PredictionModel
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

_WORKER_MODEL: PredictionModel | None = None


def _init_worker_process(model_type: str) -> None:
    global _WORKER_MODEL
    from src.pages.prediction.page_data_loader import get_prediction_model

    _WORKER_MODEL = get_prediction_model(model_type)


def _run_chunk_in_worker(
    model_input: dict[str, Any], chunk: list[tuple[str, str]], features: list[str]
) -> list[dict[str, Any]]:
    if _WORKER_MODEL is None:
        raise RuntimeError("Worker model not initialized")
    return _WORKER_MODEL.predict_batch(model_input, chunk, features)


class PredictionExecutor:
    def __init__(self, total_tasks: int):
        self.total_tasks = max(0, int(total_tasks))
        self.cpu_count = os.cpu_count() or 2

        self.single_threshold = int(os.getenv("PREDICTION_SINGLE_THREAD_THRESHOLD", "2048"))
        self.min_chunk_size = int(os.getenv("PREDICTION_MIN_CHUNK_SIZE", "256"))
        self.use_process_pool = os.getenv("PREDICTION_USE_PROCESS_POOL", "0").lower() in (
            "1",
            "true",
        )
        self.timeout = int(os.getenv("PREDICTION_OVERALL_TIMEOUT_SEC", "300"))

    def execute_parallel(
        self,
        model: PredictionModel,
        combinations: list[tuple[str, str]],
        model_input: dict[str, Any],
        features: list[str],
    ) -> list[dict[str, Any]]:
        if not combinations:
            return []

        n_combos = len(combinations)
        t0 = time.monotonic()

        if self.use_process_pool and n_combos > self.single_threshold:
            num_workers = min(4, self.cpu_count)
            chunk_size = max(self.min_chunk_size, (n_combos + num_workers - 1) // num_workers)
            chunks = [combinations[i : i + chunk_size] for i in range(0, n_combos, chunk_size)]

            logger.info(
                "ProcessPool inference started | combinations=%d workers=%d chunks=%d chunk_size=%d",
                n_combos,
                num_workers,
                len(chunks),
                chunk_size,
            )
            try:
                result = self._execute_with_pool(
                    ProcessPoolExecutor, chunks, model, model_input, features, num_workers
                )
                if result:
                    logger.info(
                        "ProcessPool inference completed | combinations=%d results=%d elapsed=%.3fs",
                        n_combos,
                        len(result),
                        time.monotonic() - t0,
                    )
                    return result
            except Exception as e:
                logger.warning(f"ProcessPool execution failed, falling back to single-thread: {e}")

        result = model.predict_batch(model_input, combinations, features)
        logger.info(
            "Single-thread inference completed | combinations=%d elapsed=%.3fs",
            n_combos,
            time.monotonic() - t0,
        )
        return result

    def _execute_with_pool(
        self,
        pool_class: Any,
        chunks: list[list[tuple[str, str]]],
        model: PredictionModel,
        model_input: dict[str, Any],
        features: list[str],
        num_workers: int,
    ) -> list[dict[str, Any]] | None:
        results = []
        is_process = pool_class == ProcessPoolExecutor
        t0 = time.monotonic()
        chunk_sizes = [len(c) for c in chunks]

        pool_kwargs = {"max_workers": num_workers}
        if is_process:
            pool_kwargs.update(
                {"initializer": _init_worker_process, "initargs": (model.model_type,)}
            )

        executor = pool_class(**pool_kwargs)
        futures = []
        try:
            if is_process:
                futures = [
                    executor.submit(_run_chunk_in_worker, model_input, c, features) for c in chunks
                ]
            else:
                futures = [
                    executor.submit(model.predict_batch, model_input, c, features) for c in chunks
                ]

            completed = 0
            pending = set(futures)
            deadline = time.monotonic() + self.timeout
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Prediction pool timed out after {self.timeout}s "
                        f"({len(pending)}/{len(futures)} chunks unfinished)"
                    )
                done, pending = wait(pending, timeout=min(remaining, 5.0))
                for future in done:
                    chunk_res = future.result()
                    completed += 1
                    if chunk_res:
                        results.extend(chunk_res)
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        elapsed = time.monotonic() - t0
        logger.debug(
            "Pool execution details | pool=%s chunks=%s completed=%d/%d elapsed=%.3fs "
            "results_per_sec=%.0f",
            pool_class.__name__,
            chunk_sizes,
            completed,
            len(chunks),
            elapsed,
            len(results) / elapsed if elapsed > 0 else 0,
        )

        return results
