FLOAT_EPSILON = 1e-9

from typing import Optional, Tuple, Union, cast

from src.pages.prediction.input_form_components.form_config import (
    LANGUAGE_SCORE_RANGES,
)


class LanguageScoreValidator:
    @staticmethod
    def validate_ielts_step(score: float) -> bool:
        return abs(score * 2 - round(score * 2)) <= FLOAT_EPSILON

    @staticmethod
    def validate_score_range(score: float, language_type: str) -> Tuple[bool, Optional[str]]:
        if language_type not in LANGUAGE_SCORE_RANGES:
            return False, f"未知的语言类型: {language_type}"

        score_config = LANGUAGE_SCORE_RANGES[language_type]
        min_score = float(cast(Union[int, float], score_config["min"]))
        max_score = float(cast(Union[int, float], score_config["max"]))

        if score < min_score or score > max_score:
            return (
                False,
                f"{language_type}成绩必须在 {min_score} 到 {max_score} 之间",
            )

        if language_type == "雅思" and not LanguageScoreValidator.validate_ielts_step(score):
            return False, "雅思成绩必须是0.5的倍数"

        return True, None

    @staticmethod
    def validate_and_parse_score(
        score_text: str, language_type: str
    ) -> Tuple[Optional[float], Optional[str], bool]:
        if not score_text or not score_text.strip():
            return None, None, False

        try:
            score_value = float(score_text.strip())
        except ValueError:
            return None, f"请输入有效的{language_type}成绩", True

        is_valid, error_msg = LanguageScoreValidator.validate_score_range(
            score_value, language_type
        )

        if not is_valid:
            return None, error_msg, True

        return score_value, None, False
