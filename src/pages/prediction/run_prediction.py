import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import local
from typing import Any, List, Optional, Tuple, cast

import pandas as pd

from src.pages.prediction.prediction_model import PredictionModel
from src.pages.prediction.prediction_processor import (
    generate_prediction_combinations,
    process_prediction_results,
)
from src.pages.prediction.prediction_types import PredictionInput
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

prediction_runner_logger = setup_logger("page3", "prediction")

_thread_local = local()


def _get_worker_model() -> Optional[PredictionModel]:
    if not hasattr(_thread_local, "model"):
        return None
    return _thread_local.model


def _set_worker_model(model: Optional[PredictionModel]) -> None:
    _thread_local.model = model


def _init_worker_process(model_type: str) -> None:
    try:
        from src.pages.prediction.page_data_loader import cached_get_prediction_model

        model = cached_get_prediction_model(model_type)
        _set_worker_model(model)

        if model is None:
            prediction_runner_logger.error(f"子进程无法加载模型: {model_type}")
    except Exception as e:
        prediction_runner_logger.error(f"初始化子进程失败: {e}", exc_info=True)
        _set_worker_model(None)


def _run_prediction_chunk(
    prediction_model: Optional[PredictionModel],
    model_input_features: dict[str, float | int | str],
    combinations_chunk: list[tuple[str, str]],
    expected_features: list[str],
) -> Optional[list[dict[str, float | str]]]:
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
    model_input_features: dict[str, float | int | str],
    combinations_chunk: list[tuple[str, str]],
    expected_features: list[str],
) -> Optional[list[dict[str, float | str]]]:
    model = _get_worker_model()
    return _run_prediction_chunk(
        model,
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
            chunk_results = _run_prediction_chunk(
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


def _prepare_model_inputs(
    current_input_data: dict[str, Any],
    expected_features: list[str],
) -> Tuple[dict[str, float | int | str], list[str]]:
    base_expected_features = [
        f for f in expected_features if f not in ["target_university", "target_major"]
    ]

    model_input_features: dict[str, float | int | str] = {}
    for feature in base_expected_features:
        if feature in current_input_data:
            value = current_input_data[feature]
            if isinstance(value, (float, int, str)):
                model_input_features[feature] = value

    missing_inputs = [f for f in base_expected_features if f not in model_input_features]
    if missing_inputs:
        prediction_runner_logger.error(f"缺少必要的输入特征: {missing_inputs}")

    return model_input_features, missing_inputs


def _get_user_specified_combinations(
    current_input_data: dict[str, Any],
    all_universities_target: list[str],
    session_manager: SessionManager,
) -> Optional[list[tuple[str, str]]]:
    has_user_specification = session_manager.get("selected_target_majors") or session_manager.get(
        "selected_major_categories"
    )

    if not has_user_specification:
        return None

    target_unis = current_input_data.get("target_universities")
    target_majors = current_input_data.get("target_majors")

    if not target_majors or not isinstance(target_majors, list):
        return None

    if target_unis and isinstance(target_unis, list):
        unis_to_use = target_unis
    else:
        unis_to_use = all_universities_target

    return [(uni, major) for uni in unis_to_use for major in target_majors]


def run_single_prediction(
    current_input_data: dict[str, Any],
    prediction_model: PredictionModel,
    cases_df: pd.DataFrame,
    bg_target_similarity_cache: dict[str, float],
    expected_features: list[str],
    all_universities_target: list[str],
    all_majors_target: list[str],
    num_target_universities: int,
) -> Tuple[
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    Optional[list[dict[str, float | str]]],
    None,
]:
    session_manager = SessionManager()

    try:
        prediction_input: PredictionInput = {
            "background_university": str(current_input_data.get("background_university", "")),
            "background_major": str(current_input_data.get("background_major", "")),
            "target_universities": (
                current_input_data["target_universities"]
                if isinstance(current_input_data.get("target_universities"), list)
                else []
            ),
            "target_majors": (
                current_input_data["target_majors"]
                if isinstance(current_input_data.get("target_majors"), list)
                else []
            ),
        }

        if "gpa" in current_input_data and isinstance(current_input_data["gpa"], (int, float)):
            prediction_input["gpa"] = float(current_input_data["gpa"])
        if "language_score" in current_input_data and isinstance(
            current_input_data["language_score"], (int, float)
        ):
            prediction_input["language_score"] = float(current_input_data["language_score"])
        if "internship_count" in current_input_data and isinstance(
            current_input_data["internship_count"], (int, float)
        ):
            prediction_input["internship_count"] = int(current_input_data["internship_count"])
        if "research_count" in current_input_data and isinstance(
            current_input_data["research_count"], (int, float)
        ):
            prediction_input["research_count"] = int(current_input_data["research_count"])
        if "award_count" in current_input_data and isinstance(
            current_input_data["award_count"], (int, float)
        ):
            prediction_input["award_count"] = int(current_input_data["award_count"])
        if "paper_count" in current_input_data and isinstance(
            current_input_data["paper_count"], (int, float)
        ):
            prediction_input["paper_count"] = int(current_input_data["paper_count"])
        if "school_level" in current_input_data and isinstance(
            current_input_data["school_level"], (int, float)
        ):
            prediction_input["school_level"] = int(current_input_data["school_level"])
        if "experience_details" in current_input_data and isinstance(
            current_input_data["experience_details"], dict
        ):
            prediction_input["experience_details"] = cast(
                dict[str, str], current_input_data["experience_details"]
            )

        combinations, meta = generate_prediction_combinations(
            prediction_input, all_universities_target, all_majors_target
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
        faculty_value = current_input_data.get("faculty")
        background_faculty = (
            None
            if cross_faculty_confirmed
            else (faculty_value if isinstance(faculty_value, str) else None)
        )

        background_major_value = current_input_data.get("background_major", "")
        background_major = background_major_value if isinstance(background_major_value, str) else ""

        results = process_prediction_results(
            results=all_prediction_outputs,
            background_major=background_major,
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
