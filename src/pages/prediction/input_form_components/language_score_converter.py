class LanguageScoreConverter:
    TOEFL_TO_IELTS_MAP = {
        (118, 120): 9.0,
        (115, 117): 8.5,
        (110, 114): 8.0,
        (102, 109): 7.5,
        (94, 101): 7.0,
        (79, 93): 6.5,
        (60, 78): 6.0,
        (46, 59): 5.5,
        (35, 45): 5.0,
        (32, 34): 4.5,
        (0, 31): 4.0,
    }

    IELTS_TO_TOEFL_MAP = {
        9.0: 119,
        8.5: 116,
        8.0: 112,
        7.5: 106,
        7.0: 98,
        6.5: 86,
        6.0: 69,
        5.5: 53,
        5.0: 40,
        4.5: 33,
        4.0: 16,
    }

    _SORTED_IELTS_SCORES = sorted(IELTS_TO_TOEFL_MAP.keys())

    @staticmethod
    def toefl_to_ielts(toefl_score):
        if toefl_score is None:
            return None
        for (
            min_score,
            max_score,
        ), ielts in LanguageScoreConverter.TOEFL_TO_IELTS_MAP.items():
            if min_score <= toefl_score <= max_score:
                return ielts
        return None

    @staticmethod
    def ielts_to_toefl(ielts_score):
        if ielts_score is None:
            return None

        closest_ielts = min(
            LanguageScoreConverter._SORTED_IELTS_SCORES,
            key=lambda x: abs(x - ielts_score),
        )
        return LanguageScoreConverter.IELTS_TO_TOEFL_MAP.get(closest_ielts)
