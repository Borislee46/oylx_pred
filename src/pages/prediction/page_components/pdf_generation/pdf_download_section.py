from datetime import datetime
from typing import Any, Optional, Tuple

import pandas as pd
import streamlit as st

from src.pages.prediction.page_components.pdf_generation.generators.pdf_report_generator import (
    PDFReportGenerator,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import PredictionResultModel, SessionManager

logger = setup_logger("page3", "prediction")

# PDF缓存相关的session key列表
PDF_CACHE_KEYS = [
    "pdf_generated",
    "pdf_data",
    "pdf_filename",
    "pdf_generation_time",
    "pdf_generation_started",
    "pdf_generation_error",
]


def display_pdf_download_section(session_manager: SessionManager, user_nickname: str) -> None:
    """显示PDF下载部分"""
    if not session_manager.get("optimization_performed", False):
        return

    download_label = "下载包含个人评估与智能申请策略的完整报告"
    pdf_generated = session_manager.get("pdf_generated", False)

    if pdf_generated:
        pdf_data = session_manager.get("pdf_data")
        pdf_filename = session_manager.get("pdf_filename", f"EasyApply申请方案_{user_nickname}.pdf")
        if pdf_data:
            st.download_button(
                label=download_label,
                data=pdf_data,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True,
                type="secondary",
            )
    else:
        st.button(
            label=download_label,
            use_container_width=True,
            disabled=True,
            help="报告生成失败或未生成，请重新进行智能优化",
        )


def generate_pdf_without_session(
    user_data: dict[str, Any],
    prediction_results: PredictionResultModel,
    optimization_results: dict[str, Any],
    cases_df: pd.DataFrame,
    user_nickname: str,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """在不使用session的情况下生成PDF报告"""
    generator = PDFReportGenerator()

    pdf_data = generator.generate_report(
        user_data=user_data,
        prediction_results=prediction_results,
        optimization_results=optimization_results,
        cases_df=cases_df,
        user_nickname=user_nickname,
    )

    current_time = datetime.now()
    safe_nickname = user_nickname or "用户"
    filename = f"EasyApply申请方案_{safe_nickname}_{current_time.strftime('%Y%m%d')}.pdf"

    return pdf_data, filename, None


def clear_pdf_cache(session_manager: SessionManager) -> None:
    """清除PDF相关的缓存数据"""
    for key in PDF_CACHE_KEYS:
        session_manager.delete(key)
