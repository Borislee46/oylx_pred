from typing import Any, Dict, Optional

import pandas as pd

from src.pages.prediction.page_components.pdf_generation.utils import DataNormalizer
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

logger = setup_logger("page3", "prediction")


class PDFDataExtractor:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.normalizer = DataNormalizer()

    def extract_user_data(self) -> Dict[str, Any]:
        input_data = self.session_manager.get("input_data", {})
        if input_data is None:
            input_data = {}

        original_form_data = self.session_manager.get("original_form_data", {})
        if original_form_data is None:
            original_form_data = {}

        user_data = {**input_data, **original_form_data}

        if "gpa_raw" not in user_data or not user_data["gpa_raw"]:
            user_data["gpa_raw"] = self.session_manager.get("gpa_raw_input")

        if "language_score_raw" not in user_data or not user_data["language_score_raw"]:
            user_data["language_score_raw"] = self.session_manager.get("language_score_input")

        required_fields = {
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

        for field, default_value in required_fields.items():
            if field not in user_data or user_data[field] is None:
                user_data[field] = default_value

        selected_major_categories = self.session_manager.get("selected_major_categories", [])
        user_data["major_categories"] = selected_major_categories

        return user_data

    def extract_prediction_results(self) -> Optional[Any]:
        prediction_results = self.session_manager.get("prediction_results")
        return prediction_results

    def extract_optimization_results(self) -> Optional[Dict]:
        recommendations = self.session_manager.get("optimization_recommendations", [])
        adaptive_thresholds = self.session_manager.get("adaptive_thresholds", {})

        formatted_recommendations = []
        for i, rec in enumerate(recommendations):
            strategy_name = rec.get("type", f"申请策略{i + 1}")
            schools = rec.get("schools", [])

            if schools:
                formatted_recommendations.append(
                    {
                        "strategy_name": strategy_name,
                        "schools": schools,
                        "metrics": rec.get("metrics", {}),
                    }
                )

        return {
            "recommendations": formatted_recommendations,
            "adaptive_thresholds": adaptive_thresholds,
        }

    def extract_cases_data(self) -> pd.DataFrame:
        from src.pages.prediction.page_data_loader import cached_load_cases_data

        cases_df = cached_load_cases_data()

        if cases_df is not None and not cases_df.empty:
            return cases_df
        else:
            logger.warning("案例数据为空")
            return pd.DataFrame()

    def extract_school_details(
        self, university_name: str, cases_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        try:
            if cases_df is None:
                cases_df = self.extract_cases_data()

            if cases_df.empty:
                return self._get_default_school_details(university_name)

            school_data = cases_df[cases_df["target_university"] == university_name]

            if school_data.empty:
                return self._get_default_school_details(university_name)

            school_details = {
                "name": university_name,
                "total_applications": len(school_data),
                "acceptance_rate": (
                    school_data["admission_result"].mean()
                    if "admission_result" in school_data.columns
                    else 0
                ),
                "average_gpa": (
                    school_data["gpa_score"].mean() if "gpa_score" in school_data.columns else 0
                ),
                "popular_majors": (
                    school_data["target_major"].value_counts().head(5).to_dict()
                    if "target_major" in school_data.columns
                    else {}
                ),
            }

            return school_details

        except Exception as e:
            logger.error(f"提取学校详细信息失败 {university_name}: {str(e)}", exc_info=True)
            return self._get_default_school_details(university_name)

    def get_user_nickname(self) -> str:
        nickname = self.session_manager.get("user_nickname", "用户")
        return self.normalizer.get_field_value({"nickname": nickname}, ["nickname"], "用户")

    def get_user_email(self) -> Optional[str]:
        user_info = self.session_manager.get("user_info", {})
        email = user_info.get("email")

        if not email:
            email = self.session_manager.get("user_email")

        return email if email and email.strip() else None

    def validate_data_for_pdf_generation(self) -> Dict[str, Any]:
        try:
            user_data = self.extract_user_data()
            prediction_results = self.extract_prediction_results()
            optimization_results = self.extract_optimization_results()
            cases_df = self.extract_cases_data()
            user_nickname = self.get_user_nickname()
            user_email = self.get_user_email()

            if user_email:
                user_data["user_email"] = user_email

            return {
                "user_data": user_data,
                "prediction_results": prediction_results,
                "optimization_results": optimization_results,
                "cases_df": cases_df,
                "user_nickname": user_nickname,
                "user_email": user_email,
                "is_valid": True,
            }

        except Exception as e:
            logger.error(f"PDF数据验证失败: {str(e)}", exc_info=True)
            return {
                "user_data": self._get_default_user_data(),
                "prediction_results": None,
                "optimization_results": None,
                "cases_df": pd.DataFrame(),
                "user_nickname": "用户",
                "user_email": None,
                "is_valid": False,
                "error_message": str(e),
            }

    def _get_default_user_data(self) -> Dict[str, Any]:
        return {
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
            "major_categories": [],
        }

    def _get_default_school_details(self, university_name: str) -> Dict[str, Any]:
        return {
            "name": university_name,
            "total_applications": 0,
            "acceptance_rate": 0,
            "average_gpa": 0,
            "popular_majors": {},
        }
