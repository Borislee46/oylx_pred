from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from src.pages.prediction.core.utils import get_cached_major_similarity
from src.pages.prediction.input_form_components.language_score_converter import (
    LanguageScoreConverter,
)
from src.pages.prediction.result_modifier.language_penalty import LanguageRequirementPenalty
from src.pages.prediction.result_modifier.similarity_adjuster import (
    adjust_similarity_score,
    get_applicable_similarity_rules,
)


class SingleResultProcessor:
    def __init__(
        self,
        data_manager: Any,
        bg_major: str,
        bg_major_orig: str,
        bg_orig_lower: str,
        raw_lang: float | None,
        lang_type: str,
        bg_target_similarity_cache: dict,
        allowed_faculties: set[str],
        background_faculty: str | None,
    ):
        self.data_manager = data_manager
        self.bg_major = bg_major
        self.bg_major_orig = bg_major_orig
        self.effective_bg_major = bg_major_orig or bg_major
        self.bg_orig_lower = bg_orig_lower
        self.raw_lang = raw_lang
        self.lang_type = lang_type
        self.bg_target_similarity_cache = bg_target_similarity_cache
        self.allowed_faculties = allowed_faculties
        self.background_faculty = background_faculty

        self.user_ielts = None
        self.user_toefl = None
        if raw_lang is not None:
            if lang_type == "托福":
                self.user_toefl = raw_lang
                self.user_ielts = LanguageScoreConverter.toefl_to_ielts(raw_lang)
            else:
                self.user_ielts = raw_lang
                self.user_toefl = LanguageScoreConverter.ielts_to_toefl(raw_lang)

        self.applicable_rules = get_applicable_similarity_rules(self.effective_bg_major)

        details_df = data_manager.details_df
        self.mode_col = (
            next((c for c in details_df.columns if "学习模式" in c or "ѧϰ" in c), None)
            if details_df is not None
            else None
        )

    def process(self, result: dict[str, Any]) -> dict[str, Any] | None:
        u, m = result.get("university"), result.get("major")
        m_lower = str(m or "").lower()

        if "part" in m_lower and "time" in m_lower and "full" not in m_lower:
            return None

        row = self.data_manager.get_row(u, m)
        if row is not None and self.mode_col:
            if (
                "part" in str(row.get(self.mode_col, "")).lower()
                and "time" in str(row.get(self.mode_col, "")).lower()
                and "full" not in str(row.get(self.mode_col, "")).lower()
            ):
                return None

        res = result.copy()
        if self.raw_lang is not None and row is not None:
            reqs = row.get("_lang_reqs", {})
            penalty = LanguageRequirementPenalty.calculate_penalty(
                self.raw_lang,
                self.lang_type,
                reqs,
                pre_converted_ielts=self.user_ielts,
                pre_converted_toefl=self.user_toefl,
            )
            if penalty < 1.0:
                res["probability"] = res.get("probability", 0.0) * penalty
                res["language_penalty_applied"] = True

        res["faculty"] = (
            str(row.get("专业大类", ""))
            if row is not None and pd.notna(row.get("专业大类"))
            else ""
        )
        res["major_cn"] = (
            str(row.get("专业中文名称", ""))
            if row is not None and pd.notna(row.get("专业中文名称"))
            else ""
        )
        res["_is_in_faculty_scope"] = (
            not (self.background_faculty and self.allowed_faculties)
            or res["faculty"] in self.allowed_faculties
        )

        if self.bg_orig_lower:
            score_en = fuzz.token_sort_ratio(self.bg_orig_lower, m_lower)
            score_cn = (
                fuzz.token_sort_ratio(self.bg_orig_lower, res["major_cn"].lower())
                if res["major_cn"]
                else 0
            )
            res["_strong_match_score"] = max(score_en, score_cn)
        else:
            res["_strong_match_score"] = 0

        raw_sim = 0.0
        if self.bg_major:
            raw_sim = get_cached_major_similarity(m, self.bg_major, self.bg_target_similarity_cache)
        if self.bg_major_orig and self.bg_major_orig != self.bg_major:
            raw_sim = max(
                raw_sim,
                get_cached_major_similarity(m, self.bg_major_orig, self.bg_target_similarity_cache),
            )

        res["similarity"] = (
            adjust_similarity_score(
                background_major=self.effective_bg_major,
                target_major=str(m),
                similarity=raw_sim,
                target_major_cn=res["major_cn"],
                fuzzy_score=res["_strong_match_score"],
                applicable_rules=self.applicable_rules,
            )
            if self.effective_bg_major
            else 0.0
        )

        return res
