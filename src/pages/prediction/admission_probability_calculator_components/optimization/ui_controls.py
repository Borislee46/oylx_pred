import time
from typing import Optional

import streamlit as st

from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    OPTIMIZATION_BUTTON_MIN_INTERVAL,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

logger = setup_logger("page3", "prediction")


class OptimizationUIControls:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def reset_stuck_lock(self) -> bool:
        current_time = time.time()
        lock_start_time = self.session_manager.get("lock_start_time", 0)
        processing_lock = self.session_manager.get("processing_lock", False)

        if processing_lock and (current_time - lock_start_time) > 30:
            logger.warning("检测到长时间锁定状态，自动重置")
            self.session_manager.set(
                processing_lock=False, run_optimization=False, lock_start_time=0
            )
            st.warning("检测到优化任务异常中断，已自动重置状态。请重新开始优化。")
            return True
        return False

    def should_enable_optimization(self, disabled_status: bool) -> bool:
        processing_lock = self.session_manager.get("processing_lock", False)
        optimization_performed = self.session_manager.get("optimization_performed", False)

        return not (disabled_status or processing_lock or optimization_performed)

    def get_help_text(self) -> Optional[str]:
        processing_lock = self.session_manager.get("processing_lock", False)
        optimization_performed = self.session_manager.get("optimization_performed", False)

        if optimization_performed:
            return "同一用户背景下的最优组合已展示，如需重新优化，请更改表单输入后重新预测。"
        elif processing_lock:
            return "优化正在进行中，请等待完成..."
        return None

    def handle_optimize_click(self):
        current_time = time.time()
        last_click_time = self.session_manager.get("last_optimize_click_time", 0)
        processing_lock = self.session_manager.get("processing_lock", False)
        min_interval = OPTIMIZATION_BUTTON_MIN_INTERVAL

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
