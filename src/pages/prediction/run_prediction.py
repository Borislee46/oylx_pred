import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from src.pages.prediction.prediction_processor import (
    generate_prediction_combinations,
    process_prediction_results,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

prediction_runner_logger = setup_logger("page3", "prediction")

_process_local_model = None


def _init_worker_process(model_type: str):
    global _process_local_model

    try:
        from src.pages.prediction.page_data_loader import cached_get_prediction_model

        _process_local_model = cached_get_prediction_model(model_type)

        if _process_local_model is None:
            prediction_runner_logger.error(f"子进程无法加载模型: {model_type}")
    except Exception as e:
        prediction_runner_logger.error(f"初始化子进程失败: {e}", exc_info=True)


def _run_prediction_chunk_in_process(
    model_input_features: dict,
    combinations_chunk: list,
    expected_features: list,
):
    global _process_local_model

    if not combinations_chunk:
        return None

    if _process_local_model is None:
        prediction_runner_logger.error("子进程中模型未初始化")
        return None

    try:
        return _process_local_model.predict_batch(
            model_input_features, combinations_chunk, expected_features
        )
    except Exception as e:
        prediction_runner_logger.error(f"子进程预测失败: {e}", exc_info=True)
        return None


def run_prediction_chunk(
    prediction_model, model_input_features, combinations_chunk, expected_features
):
    if not combinations_chunk:
        return None
    return prediction_model.predict_batch(
        model_input_features, combinations_chunk, expected_features
    )


def run_single_prediction(
    current_input_data,
    prediction_model,
    cases_df,
    bg_target_similarity_cache,
    expected_features,
    all_universities_target,
    all_majors_target,
    num_target_universities,
):
    session_manager = SessionManager()

    try:
        is_truly_specified = session_manager.get("selected_target_majors") or session_manager.get(
            "selected_major_categories"
        )

        combinations, meta = generate_prediction_combinations(
            current_input_data, all_universities_target, all_majors_target
        )
        try:
            session_manager.set(**meta)
        except Exception:
            pass

        if not combinations:
            try:
                prediction_runner_logger.warning("有效组合为空：请检查候选池或筛选条件。")
            except Exception:
                pass
            return [], [], None, None

        base_expected_features = [
            f for f in expected_features if f not in ["target_university", "target_major"]
        ]
        model_input_features = {
            feature: current_input_data[feature]
            for feature in base_expected_features
            if feature in current_input_data
        }

        missing_model_inputs = [f for f in base_expected_features if f not in model_input_features]
        if missing_model_inputs:
            prediction_runner_logger.error(f"模型预测时缺少必要的输入特征: {missing_model_inputs}")
            return [], [], None, None

        if prediction_model is None:
            prediction_runner_logger.error("模型对象为 None，无法执行预测。")
            return [], [], None, None

        total = len(combinations)
        cpu_cnt = os.cpu_count() or 2
        try:
            max_workers_env = int(os.getenv("PREDICTION_MAX_WORKERS", "0"))
            max_workers_env = max(0, max_workers_env)
        except Exception:
            max_workers_env = 0

        use_process_pool = os.getenv("PREDICTION_USE_PROCESS_POOL", "0") == "1"

        if total < 64:
            executor_class = None
            num_workers = 1
        elif total < 256:
            executor_class = ThreadPoolExecutor
            num_workers = min(2 if cpu_cnt <= 2 else 4, cpu_cnt)
        else:
            if cpu_cnt <= 2:
                num_workers = 2
            elif cpu_cnt <= 4:
                num_workers = 3
            else:
                num_workers = min(8, cpu_cnt - 1)
            if use_process_pool:
                executor_class = ProcessPoolExecutor
            else:
                executor_class = ThreadPoolExecutor

        if max_workers_env > 0:
            num_workers = max(1, min(num_workers, min(max_workers_env, cpu_cnt)))

        low_core_min_chunk = 64 if cpu_cnt <= 2 else 32
        chunk_size = max(
            low_core_min_chunk, int(np.ceil(total / num_workers)) if num_workers > 0 else total
        )
        chunks = [combinations[i : i + chunk_size] for i in range(0, len(combinations), chunk_size)]

        all_prediction_outputs = []

        if executor_class is None:
            for chunk in chunks:
                outputs = run_prediction_chunk(
                    prediction_model, model_input_features, chunk, expected_features
                )
                if outputs:
                    all_prediction_outputs.extend(outputs)

        elif executor_class == ProcessPoolExecutor:
            try:
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_init_worker_process,
                    initargs=(prediction_model.model_type,),
                ) as executor:
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
                            prediction_outputs = future.result(timeout=120)
                        except Exception as e:
                            prediction_runner_logger.error(f"进程池子任务失败: {e}", exc_info=True)
                            continue
                        if prediction_outputs is not None:
                            all_prediction_outputs.extend(prediction_outputs)
            except Exception as e:
                prediction_runner_logger.error(f"进程池执行失败，回退到线程池: {e}", exc_info=True)
                executor_class = ThreadPoolExecutor

        if executor_class == ThreadPoolExecutor:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
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
                for future in as_completed(futures):
                    try:
                        prediction_outputs = future.result(timeout=120)
                    except Exception as e:
                        prediction_runner_logger.error(f"线程池子任务失败: {e}", exc_info=True)
                        continue
                    if prediction_outputs is not None:
                        all_prediction_outputs.extend(prediction_outputs)

        results = all_prediction_outputs
        try:
            results.sort(
                key=lambda x: (
                    -float(x.get("probability", 0.0) or 0.0),
                    str(x.get("university", "")),
                    str(x.get("major", "")),
                )
            )
        except Exception:
            pass

        user_specified_combinations_param = None

        if is_truly_specified:
            target_unis = current_input_data.get("target_universities")
            target_majors = current_input_data.get("target_majors")

            if target_majors:
                unis_to_use = target_unis if target_unis else all_universities_target

                user_specified_combinations_param = [
                    (uni, major) for uni in unis_to_use for major in target_majors
                ]
        else:
            pass

        (
            processed_similarity_results,
            processed_cross_major_results,
            processed_user_specified_results,
        ) = process_prediction_results(
            results=results,
            background_major=current_input_data.get("background_major", ""),
            bg_target_similarity_cache=bg_target_similarity_cache,
            num_target_universities=num_target_universities,
            cases_df=cases_df if cases_df is not None else pd.DataFrame(),
            user_specified_combinations=user_specified_combinations_param,
        )

        return (
            processed_similarity_results,
            processed_cross_major_results,
            processed_user_specified_results,
            None,
        )
    except Exception as e:
        prediction_runner_logger.error(f"执行单个预测时发生错误: {e}", exc_info=True)
        return [], [], None, None
