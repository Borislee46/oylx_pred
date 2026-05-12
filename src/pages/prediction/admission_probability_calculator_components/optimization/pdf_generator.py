# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
import threading
from typing import Any

import pandas as pd

from src.pages.prediction.page_components.pdf_generation.pdf_download_section import (
    generate_pdf_without_session,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class PDFGenerator:
    def __init__(self, thread_lock: threading.Lock):
        self._thread_lock = thread_lock

    def generate_pdf_in_thread(
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
                with self._thread_lock:
                    result_container["pdf_data"] = pdf_data
                    result_container["filename"] = filename
                    result_container["success"] = True
            else:
                with self._thread_lock:
                    result_container["error"] = error_msg or "PDF生成失败"
                    result_container["success"] = False
        except Exception as e:
            logger.error(f"PDF生成失败: {type(e).__name__}: {str(e)}", exc_info=True)
            with self._thread_lock:
                result_container["error"] = str(e)
                result_container["success"] = False

    def generate_pdf_in_thread_with_timeout(
        self,
        user_data: dict,
        prediction_results: Any,
        optimization_results: dict,
        cases_df: pd.DataFrame,
        user_nickname: str,
        result_container: dict,
        timeout_event: threading.Event,
    ):
        pdf_timeout_seconds = 30.0

        def pdf_generation():
            self.generate_pdf_in_thread(
                user_data,
                prediction_results,
                optimization_results,
                cases_df,
                user_nickname,
                result_container,
            )
            timeout_event.set()

        pdf_thread = threading.Thread(target=pdf_generation, name="PDFGenerationWorker")
        pdf_thread.daemon = True
        pdf_thread.start()

        pdf_thread.join(timeout=pdf_timeout_seconds)

        if pdf_thread.is_alive():
            logger.warning(f"PDF生成超时（>{pdf_timeout_seconds}秒），终止线程")
            with self._thread_lock:
                result_container["error"] = f"PDF生成超时（>{pdf_timeout_seconds}秒）"
                result_container["success"] = False
        elif not timeout_event.is_set():
            logger.warning("PDF生成线程异常终止")
            with self._thread_lock:
                if "error" not in result_container:
                    result_container["error"] = "PDF生成线程异常终止"
                    result_container["success"] = False
