import concurrent.futures
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import streamlit as st
from pandas.util import hash_pandas_object

from src.pages.prediction.admission_probability_calculator_components.data_processor import (
    DataProcessor,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer import (
    SchoolSelectionOptimizer,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

logger = setup_logger("page3", "prediction")


class OptimizationUI:
    def __init__(self, session_manager: SessionManager, correlation_matrix=None):
        self.session_manager = session_manager
        self.correlation_matrix = correlation_matrix
        self.data_processor = DataProcessor()

    def _check_and_reset_stuck_lock(self):
        current_time = time.time()
        lock_start_time = self.session_manager.get("lock_start_time", 0)
        processing_lock = self.session_manager.get("processing_lock", False)

        if processing_lock and (current_time - lock_start_time) > 30:
            logger.warning("检测到长时间锁定状态，自动解锁")
            self.session_manager.set(
                processing_lock=False, run_optimization=False, lock_start_time=0
            )
            st.warning("检测到优化任务异常中断，已自动重置状态。你现在可以重新开始优化。")
            return True
        return False

    def display_optimization_tab(
        self,
        df: pd.DataFrame,
        gpa: float = None,
        language_score: float = None,
        disabled_status: bool = False,
    ) -> bool:
        self._check_and_reset_stuck_lock()

        def on_optimize_click():
            current_time = time.time()
            last_click_time = self.session_manager.get("last_optimize_click_time", 0)
            processing_lock = self.session_manager.get("processing_lock", False)
            min_interval = 10.0

            if processing_lock:
                st.toast("优化正在进行中，请等待完成")
                return

            if current_time - last_click_time < min_interval:
                remaining_time = min_interval - (current_time - last_click_time)
                st.toast(f"请等待 {remaining_time:.1f} 秒后再次点击")
                return

            self.session_manager.set(
                last_optimize_click_time=current_time,
                processing_lock=True,
                run_optimization=True,
                lock_start_time=current_time,
            )

        processing_lock = self.session_manager.get("processing_lock", False)
        optimization_performed = self.session_manager.get("optimization_performed", False)

        help_text_optimize = None
        if optimization_performed:
            help_text_optimize = (
                "同一用户背景下的最优组合已展示，如需重新优化，请更改表单输入后重新预测。"
            )
        elif processing_lock:
            help_text_optimize = "优化正在进行中，请等待完成..."

        session_id = self.session_manager.get("session_id", "default")
        button_key = f"optimize_button_{session_id}"

        optimize_btn = st.button(
            "开始智能优化选校",
            type="primary",
            on_click=on_optimize_click,
            disabled=(disabled_status or processing_lock or optimization_performed),
            help=help_text_optimize,
            key=button_key,
        )

        optimization_started_this_run = False

        if self.session_manager.get("run_optimization", False):
            self.session_manager.set(run_optimization=False)

            optimization_started_this_run = True

            logger.info(f"优化开始，候选学校数量: {len(df)}")

            if len(df) < 2:
                st.warning("候选学校数量不足，至少需要2所学校才能进行优化")
                self.session_manager.set(processing_lock=False, lock_start_time=0)
            else:
                try:
                    self.optimize_school_selection(df, gpa, language_score)

                except Exception as e:
                    logger.error(f"优化过程中发生错误: {e}")
                    st.error(f"优化过程中出错: {str(e)}")

                finally:
                    self.session_manager.set(processing_lock=False, lock_start_time=0)

        if self.session_manager.get("optimization_performed", False):
            recommendations = self.session_manager.get("optimization_recommendations", [])
            adaptive_thresholds = self.session_manager.get("adaptive_thresholds", None)
            if recommendations:
                optimizer_for_viz = SchoolSelectionOptimizer()
                optimizer_for_viz.visualize_recommendations(recommendations, adaptive_thresholds)

        return optimization_started_this_run

    def optimize_school_selection(
        self, df: pd.DataFrame, gpa: float = None, language_score: float = None
    ) -> None:
        def _animate_step(placeholder, base_text, duration):
            interval = 0.3
            start_time = time.time()
            cycle_count = 0
            while time.time() - start_time < duration:
                dots = "." * ((cycle_count % 3) + 1)
                placeholder.markdown(
                    f"<p style='font-size: 14px; font-weight: normal;'>{base_text}{dots}</p>",
                    unsafe_allow_html=True,
                )
                time.sleep(interval)
                cycle_count += 1
            placeholder.markdown(
                f"<p style='font-size: 14px; font-weight: normal;'>{base_text}...</p>",
                unsafe_allow_html=True,
            )

        status_container = st.empty()
        try:
            with status_container.status("解析中...", expanded=True) as status:
                step_placeholder = st.empty()

                animate_ui = len(df) >= 20
                if animate_ui:
                    _animate_step(step_placeholder, "解析用户背景与申请偏好", 0.7)
                else:
                    step_placeholder.markdown(
                        "<p style='font-size: 14px; font-weight: normal;'>解析用户背景与申请偏好...</p>",
                        unsafe_allow_html=True,
                    )

                all_schools_data = self.data_processor.prepare_optimizer_input(df)

                input_data = self.session_manager.get("input_data")
                if not input_data:
                    st.error("无法获取用户输入信息，无法执行优化。")
                    status.update(label="优化失败", state="error")
                    return

                background_major = input_data.get("background_major", "")
                background_faculty = input_data.get("faculty")
                school_level = input_data.get("school_level")
                gpa_value = input_data.get("gpa")

                if animate_ui:
                    _animate_step(step_placeholder, "调用 NSGA-III 算法", 0.8)
                else:
                    step_placeholder.markdown(
                        "<p style='font-size: 14px; font-weight: normal;'>调用 NSGA-III 算法...</p>",
                        unsafe_allow_html=True,
                    )

                population_size = 48
                n_generations = 60
                max_runtime_seconds = 25.0
                optimizer = SchoolSelectionOptimizer(
                    population_size=population_size,
                    n_generations=n_generations,
                    correlation_matrix=self.correlation_matrix,
                )

                def run_core_optimization():
                    return optimizer.optimize(
                        all_schools_data=all_schools_data,
                        background_major=background_major,
                        background_faculty=background_faculty,
                        school_level=school_level,
                        gpa=gpa_value,
                    )

                use_process_pool = bool(os.getenv("OPTIMIZER_USE_PROCESS_POOL", "0") == "1")

                if use_process_pool:
                    with ProcessPoolExecutor(max_workers=1) as process_executor:
                        future = process_executor.submit(run_core_optimization)
                        base_text = "执行核心算法迭代，构建 pareto 最优前沿解集中"
                        cycle_count = 0
                        start_time = time.time()
                        while animate_ui and not future.done():
                            if time.time() - start_time > max_runtime_seconds:
                                try:
                                    future.cancel()
                                except Exception:
                                    pass
                                raise TimeoutError("优化运行超时，请稍后重试或缩小候选集")
                            dots = "." * ((cycle_count % 3) + 1)
                            step_placeholder.markdown(
                                f"<p style='font-size: 14px; font-weight: normal;'>{base_text}{dots}</p>",
                                unsafe_allow_html=True,
                            )
                            time.sleep(0.3)
                            cycle_count += 1
                else:
                    with concurrent.futures.ThreadPoolExecutor() as thread_executor:
                        future = thread_executor.submit(run_core_optimization)

                        base_text = "执行核心算法迭代，构建 pareto 最优前沿解集中"
                        cycle_count = 0
                        start_time = time.time()
                        while animate_ui and not future.done():
                            if time.time() - start_time > max_runtime_seconds:
                                try:
                                    future.cancel()
                                except Exception:
                                    pass
                                raise TimeoutError("优化运行超时，请稍后重试或缩小候选集")
                            dots = "." * ((cycle_count % 3) + 1)
                            step_placeholder.markdown(
                                f"<p style='font-size: 14px; font-weight: normal;'>{base_text}{dots}</p>",
                                unsafe_allow_html=True,
                            )
                            time.sleep(0.3)
                            cycle_count += 1

                step_placeholder.markdown(
                    f"<p style='font-size: 14px; font-weight: normal;'>{base_text}...</p>",
                    unsafe_allow_html=True,
                )

                recommendations, adaptive_thresholds = future.result()

                if animate_ui:
                    _animate_step(step_placeholder, "运行 Quasi-Monte Carlo 仿真模拟中", 1.2)
                else:
                    step_placeholder.markdown(
                        "<p style='font-size: 14px; font-weight: normal;'>运行 Quasi-Monte Carlo 仿真模拟中...</p>",
                        unsafe_allow_html=True,
                    )

                self.session_manager.set(
                    optimization_performed=True,
                    optimization_recommendations=recommendations,
                    adaptive_thresholds=adaptive_thresholds,
                )

                if not df.empty:
                    try:
                        required_cols = ["目标院校", "原始专业名称", "调整后概率"]
                        safe_df = df[required_cols].copy()
                        safe_df["调整后概率"] = pd.to_numeric(
                            safe_df["调整后概率"], errors="coerce"
                        ).fillna(0.0)
                        self.session_manager.set(
                            optimization_input_hash=hash_pandas_object(safe_df).sum()
                        )
                    except Exception:
                        self.session_manager.set(optimization_input_hash=int(time.time()))

                step_placeholder.empty()

                status.update(label="智能选校优化完成", state="complete", expanded=False)
                time.sleep(0.5)

            status_container.empty()

        except Exception as e:
            logger.error(f"优化过程中出错: {str(e)}")
            st.error(f"优化过程中出错，请稍后再试: {str(e)}")
            try:
                status.update(label="优化失败", state="error")
            except Exception:
                pass
