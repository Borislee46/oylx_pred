from typing import Any, Dict, Optional

import pandas as pd

from src.pages.prediction.page_components.pdf_generation.utils import DataNormalizer
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

logger = setup_logger("page3", "prediction")


class PDFDataExtractor:
    _REQUIRED_DEFAULTS = {
        "background_university": "未填写",
        "background_major": "未填写",
        "background_major_original": "未填写",
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

    _DEFAULT_SCHOOL = {
        "total_applications": 0,
        "acceptance_rate": 0,
        "average_gpa": 0,
        "popular_majors": {},
    }

    def __init__(self, session_manager: SessionManager):
        self.session = session_manager
        self.norm = DataNormalizer()

    def _get(self, key: str, default: Any = None) -> Any:
        val = self.session.get(key, default)
        return default if val in (None, "") else val

    def extract_user_data(self) -> Dict[str, Any]:
        data = {**self._get("input_data", {}), **self._get("original_form_data", {})}

        if "gpa" in data and data["gpa"] is not None:
            data["gpa_score"] = data["gpa"]
        data.setdefault("gpa_raw", self._get("gpa_raw_input"))
        data.setdefault("language_score_raw", self._get("language_score_input"))

        for field, default in self._REQUIRED_DEFAULTS.items():
            val = data.get(field)
            if val in (None, "") or (isinstance(default, str) and val == ""):
                data[field] = default

        data["major_categories"] = self._get("selected_major_categories", [])
        return data

    def extract_prediction_results(self) -> Optional[Any]:
        return self._get("prediction_results")

    def extract_optimization_results(self) -> Dict[str, Any]:
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

    def extract_school_details(
        self, university_name: str, cases_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        df = cases_df if cases_df is not None else self.extract_cases_data()
        if df.empty:
            return {**self._DEFAULT_SCHOOL, "name": university_name}

        school = df[df["target_university"] == university_name]
        if school.empty:
            return {**self._DEFAULT_SCHOOL, "name": university_name}

        return {
            "name": university_name,
            "total_applications": len(school),
            "acceptance_rate": (
                school["admission_result"].mean() if "admission_result" in school.columns else 0
            ),
            "average_gpa": school["gpa_score"].mean() if "gpa_score" in school.columns else 0,
            "popular_majors": (
                school["target_major"].value_counts().head(5).to_dict()
                if "target_major" in school.columns
                else {}
            ),
        }

    def get_user_nickname(self) -> str:
        nickname = self._get("user_nickname")
        if not nickname:
            import streamlit as st
            nickname = st.session_state.get("e2_user_nickname") or st.session_state.get("user_nickname")
        return self.norm.get_field_value(
            {"nickname": nickname or "用户"}, ["nickname"], "用户"
        )

    def get_user_email(self) -> Optional[str]:
        email = self.session.get("user_info", {}).get("email") or self._get("user_email")
        return email.strip() if email and isinstance(email, str) else None

    def validate_data_for_pdf_generation(self) -> Dict[str, Any]:
        user_data = self.extract_user_data()
        email = self.get_user_email()
        if email:
            user_data["user_email"] = email

        return {
            "user_data": user_data,
            "prediction_results": self.extract_prediction_results(),
            "optimization_results": self.extract_optimization_results(),
            "cases_df": self.extract_cases_data(),
            "user_nickname": self.get_user_nickname(),
            "user_email": email,
            "is_valid": True,
        }
