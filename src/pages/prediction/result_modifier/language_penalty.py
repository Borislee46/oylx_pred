import math
import re
from typing import Any

import pandas as pd

from src.pages.prediction.core.utils import _data_manager
from src.pages.prediction.input_form_components.language_score_converter import (
    LanguageScoreConverter,
)
from src.pages.prediction.result_modifier.config import (
    LANGUAGE_REQUIREMENT_PENALTY_MIDPOINT,
    LANGUAGE_REQUIREMENT_PENALTY_STEEPNESS,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class LanguageRequirementPenalty:
    IELTS_PATTERN = re.compile(r"IELTS\s*[:：\s]*(\d(?:\.\d)?)", re.IGNORECASE)
    TOEFL_PATTERN = re.compile(r"TOEFL\s*[:：\s]*(\d{2,3})", re.IGNORECASE)

    @staticmethod
    def extract_requirement(row: pd.Series) -> dict[str, float]:
        requirements = {}

        ielts_val = row.get("IELTS")
        if pd.notna(ielts_val):
            match = LanguageRequirementPenalty.IELTS_PATTERN.search(str(ielts_val))
            if match:
                requirements["IELTS"] = float(match.group(1))
            else:
                try:
                    requirements["IELTS"] = float(ielts_val)
                except (ValueError, TypeError):
                    pass

        toefl_val = row.get("TOEFL")
        if pd.notna(toefl_val):
            match = LanguageRequirementPenalty.TOEFL_PATTERN.search(str(toefl_val))
            if match:
                requirements["TOEFL"] = float(match.group(1))
            else:
                try:
                    requirements["TOEFL"] = float(toefl_val)
                except (ValueError, TypeError):
                    pass

        if not requirements.get("IELTS") or not requirements.get("TOEFL"):
            admission_req = row.get("录取要求", "")
            if pd.notna(admission_req) and isinstance(admission_req, str):
                if not requirements.get("IELTS"):
                    match = LanguageRequirementPenalty.IELTS_PATTERN.search(admission_req)
                    if match:
                        requirements["IELTS"] = float(match.group(1))
                if not requirements.get("TOEFL"):
                    match = LanguageRequirementPenalty.TOEFL_PATTERN.search(admission_req)
                    if match:
                        requirements["TOEFL"] = float(match.group(1))

        return requirements

    @staticmethod
    def calculate_penalty(
        user_score: float, user_lang_type: str, requirements: dict[str, float]
    ) -> float:
        if not requirements:
            return 1.0

        target_ielts = requirements.get("IELTS")
        target_toefl = requirements.get("TOEFL")

        if target_ielts is None and target_toefl is None:
            return 1.0

        try:
            if user_lang_type == "托福":
                user_toefl = user_score
                user_ielts = LanguageScoreConverter.toefl_to_ielts(user_score)
            else:
                user_ielts = user_score
                user_toefl = LanguageScoreConverter.ielts_to_toefl(user_score)
        except Exception:
            return 1.0

        gaps = []
        if target_ielts is not None and user_ielts is not None:
            gaps.append(target_ielts - user_ielts)

        if target_toefl is not None and user_toefl is not None:
            gaps.append((target_toefl - user_toefl) / 10.0)

        if not gaps:
            return 1.0

        gap = min(gaps)
        if gap <= 0:
            return 1.0

        try:
            penalty = 1.0 / (
                1.0
                + math.exp(
                    LANGUAGE_REQUIREMENT_PENALTY_STEEPNESS
                    * (gap - LANGUAGE_REQUIREMENT_PENALTY_MIDPOINT)
                )
            )
            return max(0.01, min(1.0, penalty))
        except OverflowError:
            return 0.01

    @classmethod
    def apply_penalty_to_results(
        cls, results: list[dict[str, Any]], user_lang_score: float, user_lang_type: str
    ) -> list[dict[str, Any]]:
        if not results or user_lang_score is None:
            return results

        for res in results:
            univ = res.get("university")
            major = res.get("major")
            if not univ or not major:
                continue

            row = _data_manager.get_row(univ, major)
            if row is None:
                continue

            reqs = cls.extract_requirement(row)
            penalty = cls.calculate_penalty(user_lang_score, user_lang_type, reqs)

            if penalty < 1.0:
                original_prob = res.get("probability", 0.0)
                res["probability"] = original_prob * penalty
                res["language_penalty_applied"] = True

        return results
