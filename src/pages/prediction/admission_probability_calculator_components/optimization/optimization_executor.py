import concurrent.futures
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.admission_probability_calculator_components.data_processor import (
    DataProcessor,
)
from src.pages.prediction.page_components.pdf_generation.generators.pdf_data_extractor import (
    PDFDataExtractor,
)
from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    DEFAULT_N_GENERATIONS,
    DEFAULT_POPULATION_SIZE,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer import (
    SchoolSelectionOptimizer,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

from .cache_builder import CacheBuilder
from .pdf_generator import PDFGenerator
from .result_handler import ResultHandler
from .ui_animation import OptimizationUIAnimation

logger = setup_logger("page3", "prediction")


class OptimizationExecutor:

    def __init__(
        self,
        session_manager: SessionManager,
        correlation_matrix,
        data_processor: DataProcessor,
        thread_lock: threading.Lock,
        max_optimization_time: float = 30.0,
    ):
        self.session_manager = session_manager
        self.correlation_matrix = correlation_matrix
        self.data_processor = data_processor
        self._thread_lock = thread_lock
        self.max_optimization_time = max_optimization_time
        self.cache_builder = CacheBuilder()
        self.result_handler = ResultHandler(session_manager)
        self.pdf_generator = PDFGenerator(thread_lock)
        self.ui_animation = OptimizationUIAnimation()

    def run_optimization_with_timeout(
        self,
        optimizer,
        all_schools_data,
        input_data,
        major_category_cache,
        bg_target_similarity_cache,
    ):
        use_process_pool = bool(os.getenv("OPTIMIZER_USE_PROCESS_POOL", "0") == "1")
        executor_class = (
            ProcessPoolExecutor if use_process_pool else concurrent.futures.ThreadPoolExecutor
        )
        max_workers = 1 if use_process_pool else None

        def run_core_optimization():
            logger.info(
                f"调用optimizer.optimize，参数: "
                f"all_schools_data数量={len(all_schools_data)}, "
                f"background_major={input_data.get('background_major', '')}, "
                f"background_faculty={input_data.get('faculty')}, "
                f"school_level={input_data.get('school_level')}, "
                f"gpa={input_data.get('gpa')}"
            )
            result = optimizer.optimize(
                all_schools_data=all_schools_data,
                background_major=input_data.get("background_major", ""),
                background_faculty=input_data.get("faculty"),
                school_level=input_data.get("school_level"),
                gpa=input_data.get("gpa"),
                major_category_cache=major_category_cache,
                bg_target_similarity_cache=bg_target_similarity_cache,
            )
            logger.info(f"optimizer.optimize返回结果: 推荐数量={len(result[0]) if result else 0}")
            return result

        with executor_class(max_workers=max_workers) as executor:
            future = executor.submit(run_core_optimization)

            try:
                return future.result(timeout=self.max_optimization_time)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError("优化运行超时，请稍后重试或缩小候选集")

    def run_optimization_in_thread(
        self,
        optimizer,
        all_schools_data,
        input_data,
        major_category_cache,
        bg_target_similarity_cache,
        result_container: dict,
    ):
        try:
            logger.info(f"开始执行优化，学校数据数量: {len(all_schools_data)}")
            recommendations, adaptive_thresholds = self.run_optimization_with_timeout(
                optimizer,
                all_schools_data,
                input_data,
                major_category_cache,
                bg_target_similarity_cache,
            )
            logger.info(f"优化完成，推荐数量: {len(recommendations)}")
            with self._thread_lock:
                result_container["recommendations"] = recommendations
                result_container["adaptive_thresholds"] = adaptive_thresholds
                result_container["success"] = True
        except TimeoutError as e:
            logger.error(f"优化超时: {str(e)}")
            with self._thread_lock:
                result_container["error"] = f"优化超时: {str(e)}"
                result_container["success"] = False
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"优化数据错误: {type(e).__name__}: {str(e)}", exc_info=True)
            with self._thread_lock:
                result_container["error"] = f"数据错误: {str(e)}"
                result_container["success"] = False
        except Exception as e:
            logger.error(f"优化失败: {type(e).__name__}: {str(e)}", exc_info=True)
            with self._thread_lock:
                result_container["error"] = f"未知错误: {str(e)}"
                result_container["success"] = False

    def execute_optimization(
        self, df: pd.DataFrame, gpa: float = None, language_score: float = None
    ):
        status_container = st.empty()

        try:
            logger.info(f"开始执行优化，输入数据框大小: {len(df)}")
            logger.info(f"输入数据框列: {list(df.columns)}")

            with status_container.status("解析中...", expanded=True) as status:
                step_placeholder = st.empty()
                animate_ui = len(df) >= 20

                self.ui_animation.update_step(
                    step_placeholder, "解析用户背景与申请偏好", animate_ui, 0.7
                )
                all_schools_data = self.data_processor.prepare_optimizer_input(df)
                logger.info(f"准备优化器输入数据完成，学校数据数量: {len(all_schools_data)}")

                if not all_schools_data:
                    logger.warning("优化器输入数据为空，无法继续优化")
                    st.error("优化器输入数据为空，无法执行优化。")
                    status.update(label="优化失败", state="error")
                    self.session_manager.set(processing_lock=False, lock_start_time=0)
                    return

                input_data = self.session_manager.get("input_data")
                if not input_data:
                    logger.error("无法获取用户输入信息")
                    st.error("无法获取用户输入信息，无法执行优化。")
                    status.update(label="优化失败", state="error")
                    self.session_manager.set(processing_lock=False, lock_start_time=0)
                    return

                logger.info(
                    f"用户输入数据: background_major={input_data.get('background_major')}, "
                    f"faculty={input_data.get('faculty')}, "
                    f"school_level={input_data.get('school_level')}, "
                    f"gpa={input_data.get('gpa')}"
                )

                major_category_cache = self.cache_builder.build_major_category_cache(
                    all_schools_data
                )
                logger.info(f"专业类别缓存大小: {len(major_category_cache)}")

                bg_target_similarity_cache = self.cache_builder.get_bg_target_similarity_cache()
                logger.info(f"背景目标相似度缓存大小: {len(bg_target_similarity_cache)}")

                self.ui_animation.update_step(
                    step_placeholder, "调用 NSGA-III 算法", animate_ui, 0.8
                )
                optimizer = SchoolSelectionOptimizer(
                    population_size=DEFAULT_POPULATION_SIZE,
                    n_generations=DEFAULT_N_GENERATIONS,
                    correlation_matrix=self.correlation_matrix,
                )
                logger.info("优化器初始化完成")

                if animate_ui:
                    if self.session_manager.get("optimization_thread_running", False):
                        logger.warning("检测到已有优化线程在运行，跳过本次请求")
                        st.warning("优化任务正在进行中，请勿重复点击")
                        status.update(label="优化进行中", state="running")
                        self.session_manager.set(processing_lock=False, lock_start_time=0)
                        return

                    result_container: dict[str, Any] = {}
                    optimization_thread = threading.Thread(
                        target=self.run_optimization_in_thread,
                        args=(
                            optimizer,
                            all_schools_data,
                            input_data,
                            major_category_cache,
                            bg_target_similarity_cache,
                            result_container,
                        ),
                        name="OptimizationThread",
                    )
                    optimization_thread.daemon = True
                    with self._thread_lock:
                        self.session_manager.set(optimization_thread_running=True)
                    optimization_thread.start()

                    interval = 0.3
                    cycle_count = 0
                    base_text = "执行核心算法迭代，构建 pareto 最优前沿解集中"

                    while optimization_thread.is_alive():
                        dots = "." * ((cycle_count % 3) + 1)
                        step_placeholder.markdown(
                            f"<p style='font-size: 14px; font-weight: normal;'>{base_text}{dots}</p>",
                            unsafe_allow_html=True,
                        )
                        time.sleep(interval)
                        cycle_count += 1

                    optimization_thread.join(timeout=2.0)

                    with self._thread_lock:
                        self.session_manager.set(optimization_thread_running=False)

                    if not result_container.get("success", False):
                        error_msg = result_container.get("error", "未知错误")
                        logger.error(f"优化过程失败: {error_msg}")
                        raise Exception(error_msg)

                    recommendations = result_container["recommendations"]
                    adaptive_thresholds = result_container["adaptive_thresholds"]
                    logger.info(
                        f"优化完成，推荐结果数量: {len(recommendations)}, "
                        f"自适应阈值: {adaptive_thresholds}"
                    )

                    step_placeholder.markdown(
                        f"<p style='font-size: 14px; font-weight: normal;'>{base_text}...</p>",
                        unsafe_allow_html=True,
                    )
                else:
                    step_placeholder.markdown(
                        "<p style='font-size: 14px; font-weight: normal;'>执行核心算法迭代，构建 pareto 最优前沿解集中...</p>",
                        unsafe_allow_html=True,
                    )
                    logger.info("开始同步执行优化（数据量小于20）")
                    recommendations, adaptive_thresholds = self.run_optimization_with_timeout(
                        optimizer,
                        all_schools_data,
                        input_data,
                        major_category_cache,
                        bg_target_similarity_cache,
                    )
                    logger.info(
                        f"同步优化完成，推荐结果数量: {len(recommendations)}, "
                        f"自适应阈值: {adaptive_thresholds}"
                    )

                self.ui_animation.update_step(
                    step_placeholder,
                    "运行 Quasi-Monte Carlo 仿真模拟中",
                    animate_ui,
                    1.2,
                )

                logger.info(f"准备保存优化结果，推荐数量: {len(recommendations)}")
                self.result_handler.save_optimization_results(
                    df, recommendations, adaptive_thresholds
                )

                extractor = PDFDataExtractor(self.session_manager)
                pdf_data_bundle = extractor.validate_data_for_pdf_generation()

                if pdf_data_bundle["is_valid"]:
                    user_nickname = (
                        pdf_data_bundle.get("user_nickname")
                        or self.session_manager.get("user_nickname")
                        or "用户"
                    )

                    pdf_result_container: dict[str, Any] = {}
                    pdf_timeout_event = threading.Event()
                    pdf_thread = threading.Thread(
                        target=self.pdf_generator.generate_pdf_in_thread_with_timeout,
                        args=(
                            pdf_data_bundle["user_data"],
                            pdf_data_bundle["prediction_results"],
                            pdf_data_bundle["optimization_results"],
                            pdf_data_bundle["cases_df"],
                            user_nickname,
                            pdf_result_container,
                            pdf_timeout_event,
                        ),
                        name="PDFGenerationThread",
                    )
                    pdf_thread.daemon = True
                    pdf_thread.start()

                    interval = 0.3
                    cycle_count = 0
                    base_text = "生成专属PDF报告中"

                    while pdf_thread.is_alive():
                        dots = "." * ((cycle_count % 3) + 1)
                        step_placeholder.markdown(
                            f"<p style='font-size: 14px; font-weight: normal;'>{base_text}{dots}</p>",
                            unsafe_allow_html=True,
                        )
                        time.sleep(interval)
                        cycle_count += 1

                    pdf_thread.join(timeout=35.0)

                    if pdf_result_container.get("success", False):
                        current_time = datetime.now()
                        self.session_manager.set(
                            pdf_generated=True,
                            pdf_data=pdf_result_container["pdf_data"],
                            pdf_filename=pdf_result_container["filename"],
                            pdf_generation_time=current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            pdf_generation_started=False,
                        )
                    else:
                        error_msg = pdf_result_container.get("error", "未知错误")
                        logger.warning(f"PDF生成失败: {error_msg}")
                        self.session_manager.set(pdf_generation_error=True)
                else:
                    error_msg = pdf_data_bundle.get("error_message", "PDF数据准备失败")
                    logger.warning(f"PDF数据准备失败: {error_msg}")
                    self.session_manager.set(pdf_generation_error=True)

                step_placeholder.empty()
                status.update(label="智能选校优化完成", state="complete", expanded=False)
                time.sleep(0.5)

            status_container.empty()

        except TimeoutError as e:
            logger.error(f"[优化] 超时: {str(e)}", exc_info=True)
            self.result_handler.handle_optimization_error(f"优化超时: {str(e)}", status)
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"[优化] 数据错误: {type(e).__name__}: {str(e)}", exc_info=True)
            self.result_handler.handle_optimization_error(f"数据错误: {str(e)}", status)
        except Exception as e:
            logger.error(f"[优化] 未知错误: {type(e).__name__}: {str(e)}", exc_info=True)
            self.result_handler.handle_optimization_error(f"优化过程中出错: {str(e)}", status)
        finally:
            with self._thread_lock:
                self.session_manager.set(optimization_thread_running=False)

