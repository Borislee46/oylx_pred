from typing import Any

import pandas as pd
import streamlit as st
from pandas.util import hash_pandas_object

from src.pages.prediction.admission_probability_calculator_components.correlation_matrix_loader import (
    load_correlation_matrix,
)
from src.pages.prediction.admission_probability_calculator_components.data_processor import (
    DataProcessor,
)
from src.pages.prediction.admission_probability_calculator_components.optimization_ui import (
    OptimizationUI,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

logger = setup_logger("page3", "prediction")


class AdmissionProbabilityCalculator:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.data_processor = DataProcessor()
        self.correlation_matrix = None
        self.optimization_ui = None
        self.available_targets_in_corr: set[str] = set()
        self.min_optimization_threshold = 20

    def _ensure_correlation_and_ui_initialized(self):
        if self.correlation_matrix is None:
            self.correlation_matrix = load_correlation_matrix()
            if self.correlation_matrix is not None:
                self.available_targets_in_corr = set(self.correlation_matrix.index)
            else:
                self.available_targets_in_corr = set()
                logger.warning("由于相关系数矩阵未能加载，选校组合分析功能将不可用。")

        if self.optimization_ui is None:
            self.optimization_ui = OptimizationUI(self.session_manager, self.correlation_matrix)

    def prepare_selected_schools_data(
        self,
        similarity_results: list[dict[str, Any]],
        cross_major_results: list[dict[str, Any]],
        user_specified_results: list[dict[str, Any]],
    ) -> pd.DataFrame:
        return self.data_processor.prepare_selected_schools_data(
            similarity_results, cross_major_results, user_specified_results
        )

    def _clear_session_state_on_data_change(self, df_hash: int):
        """当数据发生变化时清理session状态"""
        if self.session_manager.get("school_list_hash") != df_hash:
            keys_to_clear = [
                "school_selections",
                "manual_selection_applied",
                "manual_selection_hash",
                "optimization_performed",
                "optimization_recommendations",
                "adaptive_thresholds",
                "optimization_input_hash",
            ]
            for key in keys_to_clear:
                self.session_manager.delete(key)
            self.session_manager.set(school_list_hash=df_hash)

    def _get_selected_schools_data(
        self, df: pd.DataFrame
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """获取选中的学校数据"""
        school_selections = self.session_manager.get("school_selections", {})
        school_keys = list(zip(df["目标院校"], df["原始专业名称"], strict=False))

        selected_indices = [
            i for i, key in enumerate(school_keys) if school_selections.get(key, False)
        ]

        if not selected_indices:
            return [], []

        selected_rows = df.iloc[selected_indices]
        selected_probabilities = selected_rows["调整后概率"].tolist()

        renamed_columns = {
            "目标院校": "university",
            "原始专业名称": "major",
            "调整后概率": "probability",
        }
        columns_to_select = ["university", "major", "probability"]

        if "类型" in selected_rows.columns:
            renamed_columns["类型"] = "type"
            columns_to_select.append("type")

        selected_results = selected_rows.rename(columns=renamed_columns)[columns_to_select].to_dict(
            "records"
        )

        return selected_results, selected_probabilities

    def display_school_selection(
        self,
        similarity_results: list[dict[str, Any]],
        cross_major_results: list[dict[str, Any]],
        user_specified_results: list[dict[str, Any]],
        gpa: float = None,
        language_score: float = None,
        disabled_status: bool = False,
    ) -> tuple[list[dict[str, Any]], list[float]]:
        # 初始化优化运行状态
        if self.session_manager.get("run_optimization") is None:
            self.session_manager.set(run_optimization=False)

        # 准备数据
        df = self.prepare_selected_schools_data(
            similarity_results=similarity_results,
            cross_major_results=cross_major_results,
            user_specified_results=user_specified_results,
        )

        # 处理空数据情况
        if df.empty:
            st.info("当前没有可供选择的推荐学校。")
            self.session_manager.delete("school_list_hash")
            return [], []

        # 检查数据变化并清理session状态
        if not df.empty:
            df_hash = hash_pandas_object(df[["目标院校", "原始专业名称", "调整后概率"]]).sum()
            self._clear_session_state_on_data_change(df_hash)

        # 检查是否满足优化条件
        if len(df) < self.min_optimization_threshold:
            return [], []

        # 显示优化界面
        self._ensure_correlation_and_ui_initialized()
        if self.optimization_ui:
            self.optimization_ui.display_optimization_tab(df, gpa, language_score, disabled_status)

        # 获取选中的学校数据
        selected_results, selected_probabilities = self._get_selected_schools_data(df)

        # 保存到session
        self.session_manager.set(
            selected_school_results=selected_results,
            selected_school_probabilities=selected_probabilities,
        )

        return selected_results, selected_probabilities

    def optimize_school_selection(
        self, df: pd.DataFrame, gpa: float = None, language_score: float = None
    ) -> None:
        if self.optimization_ui:
            return self.optimization_ui.optimize_school_selection(df, gpa, language_score)
        return None
