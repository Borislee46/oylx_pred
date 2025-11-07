import pandas as pd
import streamlit as st
from pandas.util import hash_pandas_object

from src.pages.prediction.school_combination_optimizer_algorithm.optimizer import (
    SchoolSelectionOptimizer,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

logger = setup_logger("page3", "prediction")


class ResultHandler:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def save_optimization_results(self, df: pd.DataFrame, recommendations, adaptive_thresholds):
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

    def handle_optimization_error(self, error_message: str, status):
        logger.error(error_message)
        st.error(f"优化过程中出错，请稍后再试: {error_message}")
        status.update(label="优化失败", state="error")
        self.session_manager.set(processing_lock=False, lock_start_time=0)

    def display_optimization_results(self):
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
        else:
            logger.warning("优化结果为空，无法可视化")
