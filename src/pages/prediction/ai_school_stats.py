from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PERCENTILE_POINTS = [5, 10, 25, 50, 75, 90, 95]

NUMERIC_FEATURES = [
    "gpa",
    "language_score",
    "research_count",
    "internship_count",
    "award_count",
    "paper_count",
]

FEATURE_LABELS: dict[str, str] = {
    "gpa": "GPA",
    "language_score": "语言成绩",
    "research_count": "科研经历",
    "internship_count": "实习经历",
    "award_count": "获奖经历",
    "paper_count": "论文发表",
}


class SchoolFeatureStats:
    def __init__(self, cases_df: pd.DataFrame | None):
        self._percentiles: dict[str, dict[str, np.ndarray]] = {}
        self._global_percentiles: dict[str, np.ndarray] = {}
        self._university_counts: dict[str, int] = {}
        if cases_df is not None and not cases_df.empty:
            self._compute(cases_df)

    def _compute(self, df: pd.DataFrame) -> None:
        if "target_university" not in df.columns:
            return

        for uni in df["target_university"].dropna().unique():
            uni = str(uni)
            uni_df = df[df["target_university"] == uni]
            n = len(uni_df)
            if n < 10:
                continue
            self._university_counts[uni] = n
            self._percentiles[uni] = {}
            for feat in NUMERIC_FEATURES:
                if feat not in uni_df.columns:
                    continue
                vals = pd.to_numeric(uni_df[feat], errors="coerce").dropna()
                if len(vals) >= 5:
                    self._percentiles[uni][feat] = np.percentile(vals, PERCENTILE_POINTS)

        # Global (all-applicant) percentiles for radar chart
        for feat in NUMERIC_FEATURES:
            if feat not in df.columns:
                continue
            vals = pd.to_numeric(df[feat], errors="coerce").dropna()
            if len(vals) >= 10:
                self._global_percentiles[feat] = np.percentile(vals, PERCENTILE_POINTS)

    def get_percentile(self, university: str, feature: str, value: float) -> float | None:
        uni_pct = self._percentiles.get(university)
        if uni_pct is None:
            return None
        feat_pct = uni_pct.get(feature)
        if feat_pct is None or len(feat_pct) == 0:
            return None
        if value <= feat_pct[0]:
            return float(PERCENTILE_POINTS[0])
        if value >= feat_pct[-1]:
            return float(PERCENTILE_POINTS[-1])
        return float(np.interp(value, feat_pct, PERCENTILE_POINTS))

    def get_rank_label(self, university: str, feature: str, value: float) -> str | None:
        pct = self.get_percentile(university, feature, value)
        if pct is None:
            return None
        if pct <= 10:
            return "明显偏低"
        elif pct <= 25:
            return "偏低"
        elif pct <= 40:
            return "略低于平均"
        elif pct <= 60:
            return "中等"
        elif pct <= 75:
            return "略高于平均"
        elif pct <= 90:
            return "较高"
        else:
            return "显著优势"

    def get_global_percentile(self, feature: str, value: float) -> float | None:
        """Return student's percentile vs ALL applicants (not per-school)."""
        feat_pct = self._global_percentiles.get(feature)
        if feat_pct is None or len(feat_pct) == 0:
            return None
        if value <= feat_pct[0]:
            return float(PERCENTILE_POINTS[0])
        if value >= feat_pct[-1]:
            return float(PERCENTILE_POINTS[-1])
        return float(np.interp(value, feat_pct, PERCENTILE_POINTS))

    def get_school_feature_summary(
        self,
        student_features: dict[str, float],
        top_results: list[dict[str, Any]],
        top_n: int = 5,
    ) -> dict[str, dict[str, Any]]:
        """Return structured percentile data for unified school cards.

        Returns dict keyed by "university|major":
            percentiles: raw percentile (0-100) per feature
            values: student's actual value per feature
            labels: human-readable rank label per feature
            has_data: True if any feature has valid percentile data
            university_count: number of historical cases
        """
        result: dict[str, dict[str, Any]] = {}
        for r in top_results[:top_n]:
            uni = str(r.get("university", ""))
            major = str(r.get("major", ""))
            key = f"{uni}|{major}"
            count = self._university_counts.get(uni, 0)
            percentiles: dict[str, float] = {}
            values: dict[str, float] = {}
            labels: dict[str, str] = {}
            for feat in NUMERIC_FEATURES:
                val = student_features.get(feat)
                if val is None or val <= 0:
                    continue
                pct = self.get_percentile(uni, feat, val)
                if pct is None:
                    continue
                rank = self.get_rank_label(uni, feat, val)
                percentiles[feat] = round(pct, 1)
                values[feat] = val
                labels[feat] = rank or ""
            result[key] = {
                "percentiles": percentiles,
                "values": values,
                "labels": labels,
                "has_data": len(percentiles) > 0,
                "university_count": count,
            }
        return result
