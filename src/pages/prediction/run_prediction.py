import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Tuple

import pandas as pd

from src.pages.prediction.prediction_processor import (
    generate_prediction_combinations,
    process_prediction_results,
)
from src.pages.prediction.prediction_types import PredictionInput
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

prediction_runner_logger = setup_logger("page3", "prediction")

_process_local_model = None


def _init_worker_process(model_type: str) -> None:
    global _process_local_model

    try:
        from src.pages.prediction.page_data_loader import cached_get_prediction_model

        _process_local_model = cached_get_prediction_model(model_type)

        if _process_local_model is None:
            prediction_runner_logger.error(f"子进程无法加载模型: {model_type}")
    except Exception as e:
        prediction_runner_logger.error(f"初始化子进程失败: {e}", exc_info=True)


def _run_prediction_chunk(
    prediction_model: Any,
    model_input_features: dict,
    combinations_chunk: list,
    expected_features: list,
) -> Optional[list]:
    if not combinations_chunk or prediction_model is None:
        return None

    try:
        return prediction_model.predict_batch(
            model_input_features, combinations_chunk, expected_features
        )
    except Exception as e:
        prediction_runner_logger.error(f"预测失败: {e}", exc_info=True)
        return None


def _run_prediction_chunk_in_process(
    model_input_features: dict,
    combinations_chunk: list,
    expected_features: list,
) -> Optional[list]:
    global _process_local_model
    return _run_prediction_chunk(
        _process_local_model,
        model_input_features,
        combinations_chunk,
        expected_features,
    )


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
        prediction_model: Any,
        combinations: list,
        model_input_features: dict,
        expected_features: list,
    ) -> List[Any]:
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
        prediction_model: Any,
        chunks: List[list],
        model_input_features: dict,
        expected_features: list,
    ) -> List[Any]:
        results = []
        for chunk in chunks:
            chunk_results = _run_prediction_chunk(
                prediction_model, model_input_features, chunk, expected_features
            )
            if chunk_results:
                results.extend(chunk_results)
        return results

    def _execute_with_process_pool(
        self,
        prediction_model: Any,
        chunks: List[list],
        model_input_features: dict,
        expected_features: list,
    ) -> Optional[List[Any]]:
        try:
            with ProcessPoolExecutor(
                max_workers=len(chunks),
                initializer=_init_worker_process,
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
        prediction_model: Any,
        chunks: List[list],
        model_input_features: dict,
        expected_features: list,
        num_workers: int,
    ) -> List[Any]:
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
        executor: Any,
        chunks: List[list],
        model_input_features: dict,
        expected_features: list,
        prediction_model: Any = None,
    ) -> List[Any]:
        results = []

        if prediction_model:
            futures = [
                executor.submit(
                    _run_prediction_chunk,
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
                    _run_prediction_chunk_in_process,
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


def _prepare_model_inputs(current_input_data: dict, expected_features: list) -> Tuple[dict, list]:
    base_expected_features = [
        f for f in expected_features if f not in ["target_university", "target_major"]
    ]

    model_input_features = {
        feature: current_input_data[feature]
        for feature in base_expected_features
        if feature in current_input_data
    }

    missing_inputs = [f for f in base_expected_features if f not in model_input_features]
    if missing_inputs:
        prediction_runner_logger.error(f"缺少必要的输入特征: {missing_inputs}")

    return model_input_features, missing_inputs


def _get_user_specified_combinations(
    current_input_data: dict,
    all_universities_target: list,
    session_manager: SessionManager,
) -> Optional[list]:
    has_user_specification = session_manager.get("selected_target_majors") or session_manager.get(
        "selected_major_categories"
    )

    if not has_user_specification:
        return None

    target_unis = current_input_data.get("target_universities")
    target_majors = current_input_data.get("target_majors")

    if not target_majors:
        return None

    unis_to_use = target_unis if target_unis else all_universities_target
    return [(uni, major) for uni in unis_to_use for major in target_majors]


def run_single_prediction(
    current_input_data: dict,
    prediction_model: Any,
    cases_df: pd.DataFrame,
    bg_target_similarity_cache: Any,
    expected_features: list,
    all_universities_target: list,
    all_majors_target: list,
    num_target_universities: int,
) -> Tuple[list, list, Optional[list], None]:
    session_manager = SessionManager()

    try:
        combinations, meta = generate_prediction_combinations(
            PredictionInput(**current_input_data), all_universities_target, all_majors_target
        )

        session_manager.set(**meta)

        if not combinations:
            prediction_runner_logger.warning("有效组合为空：请检查候选池或筛选条件。")
            return [], [], None, None

        model_input_features, missing_inputs = _prepare_model_inputs(
            current_input_data, expected_features
        )
        if missing_inputs or prediction_model is None:
            return [], [], None, None

        executor = PredictionExecutor(len(combinations))
        all_prediction_outputs = executor.execute_parallel(
            prediction_model, combinations, model_input_features, expected_features
        )

        all_prediction_outputs.sort(
            key=lambda x: (
                -float(x.get("probability", 0.0) or 0.0),
                str(x.get("university", "")),
                str(x.get("major", "")),
            )
        )

        user_specified_combinations = _get_user_specified_combinations(
            current_input_data, all_universities_target, session_manager
        )

        cross_faculty_confirmed = session_manager.get("cross_faculty_confirmed", False)
        background_faculty = None if cross_faculty_confirmed else current_input_data.get("faculty")

        results = process_prediction_results(
            results=all_prediction_outputs,
            background_major=current_input_data.get("background_major", ""),
            bg_target_similarity_cache=bg_target_similarity_cache,
            num_target_universities=num_target_universities,
            cases_df=cases_df if cases_df is not None else pd.DataFrame(),
            user_specified_combinations=user_specified_combinations,
            background_faculty=background_faculty,
        )

        return (*results, None)

    except Exception as e:
        prediction_runner_logger.error(f"执行单个预测时发生错误: {e}", exc_info=True)
        return [], [], None, None
