import math
import re
from functools import lru_cache

from src.adjustment.config import (
    LANGUAGE_REQUIREMENT_PENALTY_MIDPOINT,
    LANGUAGE_REQUIREMENT_PENALTY_STEEPNESS,
)
from src.pages.prediction.input_form_components.language_score_converter import (
    LanguageScoreConverter,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_scalar, sigmoid_k

logger = setup_logger("page3", "prediction")

_seen_penalties: set[tuple[float, float, str]] = set()


def reset_penalty_tracker() -> None:
    _seen_penalties.clear()


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
            logger.debug(
                "语言要求惩罚: 达标 | user_ielts=%.1f target_ielts=%s user_toefl=%s target_toefl=%s",
                user_ielts,
                target_ielts,
                user_toefl,
                target_toefl,
            )
            return 1.0

        # 语义：返回概率乘数（1.0 = 无惩罚）。sigmoid_k 随 gap 递增，
        # 因此乘数应为 1 - sigmoid(gap)：gap 越大（差得越多）乘数越小、惩罚越重。
        # （2026-08-10 修复：此前直接返回 sigmoid(gap)，导致 gap 越大惩罚越轻，
        #  低分区出现概率随语言分提高反而下降的非单调行为。）
        raw_penalty = sigmoid_k(
            gap,
            LANGUAGE_REQUIREMENT_PENALTY_STEEPNESS,
            LANGUAGE_REQUIREMENT_PENALTY_MIDPOINT,
        )
        result = clip_scalar(1.0 - raw_penalty, 0.01, 1.0)
        key = (round(gap, 3), round(user_ielts, 1) if user_ielts else 0, str(target_ielts))
        if key in _seen_penalties:
            logger.debug(
                "语言要求惩罚(重复) | gap=%.2f penalty=%.4f | user_ielts=%.1f target_ielts=%s",
                gap,
                result,
                user_ielts,
                target_ielts,
            )
        else:
            _seen_penalties.add(key)
            logger.info(
                "语言要求惩罚 | gap=%.2f penalty=%.4f | user_ielts=%.1f target_ielts=%s",
                gap,
                result,
                user_ielts,
                target_ielts,
            )
        return result
