# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
import threading

import pandas as pd
import streamlit as st

from src.pages.prediction.admission_probability_calculator_components.data_processor import (
    DataProcessor,
)
from src.pages.prediction.admission_probability_calculator_components.optimization.optimization_executor import (
    OptimizationExecutor,
)
from src.pages.prediction.admission_probability_calculator_components.optimization.result_handler import (
    ResultHandler,
)
from src.pages.prediction.admission_probability_calculator_components.optimization.ui_controls import (
    OptimizationUIControls,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

logger = setup_logger("page3", "prediction")


class OptimizationUI:
    def __init__(self, session_manager: SessionManager, correlation_matrix=None):
        self.session_manager = session_manager
        self.correlation_matrix = correlation_matrix
        self.data_processor = DataProcessor()
        self.max_optimization_time = 30.0
        self._thread_lock = threading.Lock()
        self._setup_thread_exception_handler()

        self.ui_controls = OptimizationUIControls(session_manager)
        self.result_handler = ResultHandler(session_manager)
        self.optimization_executor = OptimizationExecutor(
            session_manager=session_manager,
            correlation_matrix=correlation_matrix,
            data_processor=self.data_processor,
            thread_lock=self._thread_lock,
            max_optimization_time=self.max_optimization_time,
        )

    def _setup_thread_exception_handler(self):
        def thread_exception_handler(args):
            logger.error(
                f"线程中未捕获的异常: {args.exc_type.__name__}: {args.exc_value}",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = thread_exception_handler

    def display_optimization_tab(
        self,
        df: pd.DataFrame,
        gpa: float = None,
        language_score: float = None,
        disabled_status: bool = False,
    ) -> bool:
        self.ui_controls.reset_stuck_lock()

        session_id = self.session_manager.get("session_id", "default")
        st.button(
            "开始智能优化选校",
            type="primary",
            on_click=self.ui_controls.handle_optimize_click,
            disabled=not self.ui_controls.should_enable_optimization(disabled_status),
            help=self.ui_controls.get_help_text(),
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
                self.optimization_executor.execute_optimization(df, gpa, language_score)

        if self.session_manager.get("optimization_performed", False):
            self.result_handler.display_optimization_results()

        return optimization_started_this_run
