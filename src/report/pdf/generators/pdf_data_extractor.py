from typing import Any

import pandas as pd

from src.report.pdf.utils import DataNormalizer
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

logger = setup_logger("page3", "prediction")


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


class PDFDataExtractor:
    _REQUIRED_DEFAULTS = {
        "background_university": "未填写",
        "background_major": "未填写",
        "background_major_original": "未填写",
        "background_major_2": "",
        "is_dual_degree": False,
        "degree_type": "",
        "gpa_score": "未填写",
        "gpa_raw": "未填写",
        "gpa_scale": "未知",
        "language_type": "language",
        "language_score": "未填写",
        "language_score_raw": "未填写",
        "research_count": 0,
        "award_count": 0,
        "internship_count": 0,
        "paper_count": 0,
        "target_universities": [],
        "target_majors": [],
    }

    def __init__(self, session_manager: SessionManager):
        self.session = session_manager
        self.norm = DataNormalizer()

    def _get(self, key: str, default: Any = None) -> Any:
        val = self.session.get(key, default)
        return default if _is_empty(val) else val

    def extract_user_data(self) -> dict[str, Any]:
        data = {**self._get("input_data", {}), **self._get("original_form_data", {})}

        if "gpa" in data and data["gpa"] is not None:
            data["gpa_score"] = data["gpa"]
        data.setdefault("gpa_raw", self._get("gpa_raw_input"))
        data.setdefault("language_score_raw", self._get("language_score_input"))

        for field, default in self._REQUIRED_DEFAULTS.items():
            val = data.get(field)
            if _is_empty(val):
                data[field] = default

        data["major_categories"] = self._get("selected_major_categories", [])
        return data

    def extract_prediction_results(self) -> Any | None:
        return self._get("prediction_results")

    def extract_optimization_results(self) -> dict[str, Any]:
        recs = self._get("optimization_recommendations", [])
        formatted = [
            {
                "strategy_name": r.get("type", f"申请策略{i + 1}"),
                "schools": r.get("schools", []),
                "metrics": r.get("metrics", {}),
            }
            for i, r in enumerate(recs)
            if r.get("schools")
        ]
        return {
            "recommendations": formatted,
            "adaptive_thresholds": self._get("adaptive_thresholds", {}),
        }

    def extract_cases_data(self) -> pd.DataFrame:
        from src.pages.prediction.page_data_loader import cached_load_cases_data

        return cached_load_cases_data()

    def get_user_nickname(self) -> str:
        nickname = self._get("user_nickname")
        if not nickname:
            import streamlit as st

            nickname = st.session_state.get("e2_user_nickname") or st.session_state.get(
                "user_nickname"
            )
        return self.norm.get_field_value({"nickname": nickname or "用户"}, ["nickname"], "用户")

    def get_user_email(self) -> str | None:
        email = self.session.get("user_info", {}).get("email") or self._get("user_email")
        return email.strip() if email and isinstance(email, str) else None
