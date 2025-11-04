import concurrent.futures
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from pandas.util import hash_pandas_object

from src.pages.prediction.admission_probability_calculator_components.data_processor import (
    DataProcessor,
)
from src.pages.prediction.page_components.pdf_generation.generators.pdf_data_extractor import (
    PDFDataExtractor,
)
from src.pages.prediction.page_components.pdf_generation.pdf_download_section import (
    generate_pdf_without_session,
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
        self.max_optimization_time = 25.0

    def _reset_stuck_lock(self):
        current_time = time.time()
        lock_start_time = self.session_manager.get("lock_start_time", 0)
        processing_lock = self.session_manager.get("processing_lock", False)

        if processing_lock and (current_time - lock_start_time) > 30:
            logger.warning("检测到长时间锁定状态，自动解锁")
            self.session_manager.set(
                processing_lock=False, run_optimization=False, lock_start_time=0
            )
            st.warning("检测到优化任务异常中断，已自动重置状态。你可以重新开始优化。")
            return True
        return False

    def _should_enable_optimization(self, disabled_status: bool) -> bool:
        processing_lock = self.session_manager.get("processing_lock", False)
        optimization_performed = self.session_manager.get("optimization_performed", False)

        return not (disabled_status or processing_lock or optimization_performed)

    def _get_help_text(self) -> str | None:
        processing_lock = self.session_manager.get("processing_lock", False)
        optimization_performed = self.session_manager.get("optimization_performed", False)

        if optimization_performed:
            return "同一用户背景下的最优组合已展示，如需重新优化，请更改表单输入后重新预测。"
        elif processing_lock:
            return "优化正在进行中，请等待完成..."
        return None

    def _handle_optimize_click(self):
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

    def display_optimization_tab(
        self,
        df: pd.DataFrame,
        gpa: float = None,
        language_score: float = None,
        disabled_status: bool = False,
    ) -> bool:
        self._reset_stuck_lock()

        session_id = self.session_manager.get("session_id", "default")
        optimize_btn = st.button(
            "开始智能优化选校",
            type="primary",
            on_click=self._handle_optimize_click,
            disabled=not self._should_enable_optimization(disabled_status),
            help=self._get_help_text(),
            key=f"optimize_button_{session_id}",
        )
        optimization_started_this_run = False
        if self.session_manager.get("run_optimization", False):
            self.session_manager.set(run_optimization=False)
            optimization_started_this_run = True
            logger.info(f"开始智能优化选校，输入数据框大小: {len(df)}")

            if len(df) < 2:
                logger.warning(f"候选学校数量不足: {len(df)} < 2")
                st.warning("候选学校数量不足，至少需要2所学校才能进行优化")
                self.session_manager.set(processing_lock=False, lock_start_time=0)
            else:
                logger.info(f"满足优化条件，开始执行优化，数据框大小: {len(df)}")
                self._execute_optimization(df, gpa, language_score)

        if self.session_manager.get("optimization_performed", False):
            self._display_optimization_results()

        return optimization_started_this_run

    def _animate_step(self, placeholder, base_text: str, duration: float):
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

    def _run_optimization_in_thread(
        self,
        optimizer,
        all_schools_data,
        input_data,
        major_category_cache,
        bg_target_similarity_cache,
        result_container: dict,
    ):
        try:
            logger.info(f"线程中开始执行优化，学校数据数量: {len(all_schools_data)}")
            recommendations, adaptive_thresholds = self._run_optimization_with_timeout(
                optimizer,
                all_schools_data,
                input_data,
                major_category_cache,
                bg_target_similarity_cache,
            )
            logger.info(f"线程中优化完成，推荐数量: {len(recommendations)}")
            result_container["recommendations"] = recommendations
            result_container["adaptive_thresholds"] = adaptive_thresholds
            result_container["success"] = True
        except Exception as e:
            logger.error(f"线程中优化失败: {str(e)}", exc_info=True)
            result_container["error"] = str(e)
            result_container["success"] = False

    def _generate_pdf_in_thread(
        self,
        user_data: dict,
        prediction_results: Any,
        optimization_results: dict,
        cases_df: pd.DataFrame,
        user_nickname: str,
        result_container: dict,
    ):
        try:
            pdf_data, filename, error_msg = generate_pdf_without_session(
                user_data=user_data,
                prediction_results=prediction_results,
                optimization_results=optimization_results,
                cases_df=cases_df,
                user_nickname=user_nickname,
            )
            if pdf_data is not None:
                result_container["pdf_data"] = pdf_data
                result_container["filename"] = filename
                result_container["success"] = True
            else:
                result_container["error"] = error_msg or "PDF生成失败"
                result_container["success"] = False
        except Exception as e:
            result_container["error"] = str(e)
            result_container["success"] = False

    def _update_step(self, step_placeholder, text: str, animate: bool, duration: float = 0.7):
        if animate:
            self._animate_step(step_placeholder, text, duration)
        else:
            step_placeholder.markdown(
                f"<p style='font-size: 14px; font-weight: normal;'>{text}...</p>",
                unsafe_allow_html=True,
            )

    def _build_major_category_cache(self, all_schools_data: list) -> dict:
        major_category_cache = {}
        for school in all_schools_data:
            uni = school.get("university", "")
            major = school.get("major", "")
            faculty = school.get("faculty", "")
            if uni and major and faculty:
                key = f"{uni}|{major}"
                major_category_cache[key] = faculty
        return major_category_cache

    def _get_bg_target_similarity_cache(self) -> dict:
        try:
            from src.pages.prediction.page_data_loader import (
                cached_load_bg_target_similarity_cache,
            )

            return cached_load_bg_target_similarity_cache()
        except Exception as e:
            logger.warning(f"无法加载bg_target_similarity_cache: {e}")
            return {}

    def _run_optimization_with_timeout(
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

    def _execute_optimization(
        self, df: pd.DataFrame, gpa: float = None, language_score: float = None
    ):
        status_container = st.empty()

        try:
            logger.info(f"开始执行优化，输入数据框大小: {len(df)}")
            logger.info(f"输入数据框列: {list(df.columns)}")

            with status_container.status("解析中...", expanded=True) as status:
                step_placeholder = st.empty()
                animate_ui = len(df) >= 20

                self._update_step(step_placeholder, "解析用户背景与申请偏好", animate_ui, 0.7)
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

                major_category_cache = self._build_major_category_cache(all_schools_data)
                logger.info(f"专业类别缓存大小: {len(major_category_cache)}")

                bg_target_similarity_cache = self._get_bg_target_similarity_cache()
                logger.info(f"背景目标相似度缓存大小: {len(bg_target_similarity_cache)}")

                self._update_step(step_placeholder, "调用 NSGA-III 算法", animate_ui, 0.8)
                optimizer = SchoolSelectionOptimizer(
                    population_size=48,
                    n_generations=60,
                    correlation_matrix=self.correlation_matrix,
                )
                logger.info("优化器初始化完成")

                if animate_ui:
                    result_container: dict[str, Any] = {}
                    optimization_thread = threading.Thread(
                        target=self._run_optimization_in_thread,
                        args=(
                            optimizer,
                            all_schools_data,
                            input_data,
                            major_category_cache,
                            bg_target_similarity_cache,
                            result_container,
                        ),
                    )
                    optimization_thread.daemon = True
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
                    recommendations, adaptive_thresholds = self._run_optimization_with_timeout(
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

                self._update_step(
                    step_placeholder,
                    "运行 Quasi-Monte Carlo 仿真模拟中",
                    animate_ui,
                    1.2,
                )

                logger.info(f"准备保存优化结果，推荐数量: {len(recommendations)}")
                self._save_optimization_results(df, recommendations, adaptive_thresholds)
                logger.info("优化结果保存完成")

                user_nickname = self.session_manager.get("user_nickname", "用户")

                extractor = PDFDataExtractor(self.session_manager)
                pdf_data_bundle = extractor.validate_data_for_pdf_generation()

                if pdf_data_bundle["is_valid"]:
                    pdf_result_container: dict[str, Any] = {}
                    pdf_thread = threading.Thread(
                        target=self._generate_pdf_in_thread,
                        args=(
                            pdf_data_bundle["user_data"],
                            pdf_data_bundle["prediction_results"],
                            pdf_data_bundle["optimization_results"],
                            pdf_data_bundle["cases_df"],
                            user_nickname,
                            pdf_result_container,
                        ),
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

                    pdf_thread.join(timeout=15.0)

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
            logger.error(f"优化超时: {str(e)}", exc_info=True)
            self._handle_optimization_error(f"优化超时: {str(e)}", status)
        except Exception as e:
            logger.error(f"优化过程中出错: {str(e)}", exc_info=True)
            self._handle_optimization_error(f"优化过程中出错: {str(e)}", status)

    def _save_optimization_results(self, df: pd.DataFrame, recommendations, adaptive_thresholds):
        logger.info(
            f"保存优化结果: recommendations数量={len(recommendations) if recommendations else 0}, "
            f"adaptive_thresholds={adaptive_thresholds}"
        )
        if recommendations:
            for i, rec in enumerate(recommendations):
                logger.debug(
                    f"推荐结果{i + 1}: type={rec.get('type')}, "
                    f"schools数量={len(rec.get('schools', []))}"
                )

        self.session_manager.set(
            optimization_performed=True,
            optimization_recommendations=recommendations,
            adaptive_thresholds=adaptive_thresholds,
            processing_lock=False,
            lock_start_time=0,
        )

        if not df.empty:
            required_cols = ["目标院校", "原始专业名称", "调整后概率"]
            safe_df = df[required_cols].copy()
            safe_df["调整后概率"] = pd.to_numeric(safe_df["调整后概率"], errors="coerce").fillna(
                0.0
            )
            self.session_manager.set(optimization_input_hash=hash_pandas_object(safe_df).sum())
            self.session_manager.set(optimization_input_hash=int(time.time()))

    def _handle_optimization_error(self, error_message: str, status):
        logger.error(error_message)
        st.error(f"优化过程中出错，请稍后再试: {error_message}")
        status.update(label="优化失败", state="error")
        self.session_manager.set(processing_lock=False, lock_start_time=0)

    def _display_optimization_results(self):
        recommendations = self.session_manager.get("optimization_recommendations", [])
        adaptive_thresholds = self.session_manager.get("adaptive_thresholds", None)

        logger.info(
            f"显示优化结果: recommendations数量={len(recommendations) if recommendations else 0}, "
            f"adaptive_thresholds是否存在={adaptive_thresholds is not None}"
        )

        if recommendations:
            logger.info(f"开始可视化推荐结果，数量: {len(recommendations)}")
            optimizer_for_viz = SchoolSelectionOptimizer()
            optimizer_for_viz.visualize_recommendations(recommendations, adaptive_thresholds)
            logger.info("可视化完成")
        else:
            logger.warning("优化结果为空，无法显示")
