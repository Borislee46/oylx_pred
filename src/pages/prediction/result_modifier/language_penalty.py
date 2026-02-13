import math
import re
from functools import lru_cache

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
    @lru_cache(maxsize=1024)
    def _extract_from_text(text: str) -> dict[str, float]:
        if not text:
            return {}
        res = {}
        i_m = LanguageRequirementPenalty.IELTS_PATTERN.search(text)
        if i_m:
            res["IELTS"] = float(i_m.group(1))
        t_m = LanguageRequirementPenalty.TOEFL_PATTERN.search(text)
        if t_m:
            res["TOEFL"] = float(t_m.group(1))
        return res

    @staticmethod
    def extract_requirement(row: dict) -> dict[str, float]:
        requirements = {}

        ielts_val = row.get("IELTS")
        if ielts_val is not None and not (isinstance(ielts_val, float) and math.isnan(ielts_val)):
            try:
                requirements["IELTS"] = float(ielts_val)
            except (ValueError, TypeError):
                text_res = LanguageRequirementPenalty._extract_from_text(str(ielts_val))
                if "IELTS" in text_res:
                    requirements["IELTS"] = text_res["IELTS"]

        toefl_val = row.get("TOEFL")
        if toefl_val is not None and not (isinstance(toefl_val, float) and math.isnan(toefl_val)):
            try:
                requirements["TOEFL"] = float(toefl_val)
            except (ValueError, TypeError):
                text_res = LanguageRequirementPenalty._extract_from_text(str(toefl_val))
                if "TOEFL" in text_res:
                    requirements["TOEFL"] = text_res["TOEFL"]

        if "IELTS" not in requirements or "TOEFL" not in requirements:
            admission_req = row.get("录取要求")
            if admission_req and isinstance(admission_req, str):
                text_res = LanguageRequirementPenalty._extract_from_text(admission_req)
                if "IELTS" not in requirements and "IELTS" in text_res:
                    requirements["IELTS"] = text_res["IELTS"]
                if "TOEFL" not in requirements and "TOEFL" in text_res:
                    requirements["TOEFL"] = text_res["TOEFL"]

        return requirements

    @staticmethod
    def calculate_penalty(
        user_score: float,
        user_lang_type: str,
        requirements: dict[str, float],
        pre_converted_ielts: float | None = None,
        pre_converted_toefl: float | None = None,
    ) -> float:
        if not requirements:
            return 1.0

        target_ielts = requirements.get("IELTS")
        target_toefl = requirements.get("TOEFL")

        if target_ielts is None and target_toefl is None:
            return 1.0

        if pre_converted_ielts is not None and pre_converted_toefl is not None:
            user_ielts = pre_converted_ielts
            user_toefl = pre_converted_toefl
        else:
            if user_lang_type == "托福":
                user_toefl = user_score
                user_ielts = LanguageScoreConverter.toefl_to_ielts(user_score)
            else:
                user_ielts = user_score
                user_toefl = LanguageScoreConverter.ielts_to_toefl(user_score)

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

        penalty = 1.0 / (
            1.0
            + math.exp(
                LANGUAGE_REQUIREMENT_PENALTY_STEEPNESS
                * (gap - LANGUAGE_REQUIREMENT_PENALTY_MIDPOINT)
            )
        )
        return max(0.01, min(1.0, penalty))
