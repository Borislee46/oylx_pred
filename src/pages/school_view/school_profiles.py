import numpy as np
import pandas as pd

from .data_loader import FEATURE_LABELS, PROFILE_FEATURES

MIN_SAMPLES_FOR_PROFILE = 10
MIN_SAMPLES_FOR_PERCENTILE = 5


class SchoolProfileCalculator:
    def __init__(self, cases_df: pd.DataFrame):
        self.df = cases_df

    def get_target_schools(self) -> list[str]:
        schools = self.df["target_university"].dropna().astype(str).unique().tolist()
        return sorted(schools)

    def compute_school_profile(self, university: str) -> dict | None:
        """P50/P25/P75 per feature for admitted students of a school."""
        mask = (self.df["target_university"] == university) & (self.df["admitted"] == 1)
        cohort = self.df[mask]
        n = len(cohort)
        if n < MIN_SAMPLES_FOR_PROFILE:
            return {"university": university, "n_admitted": n, "insufficient": True}

        profile = {"university": university, "n_admitted": n, "insufficient": False, "features": {}}
        for feat in PROFILE_FEATURES:
            vals = cohort[feat].dropna()
            if len(vals) < MIN_SAMPLES_FOR_PERCENTILE:
                profile["features"][feat] = {"p50": None, "p25": None, "p75": None, "n_valid": len(vals)}
            else:
                profile["features"][feat] = {
                    "p50": float(np.percentile(vals, 50)),
                    "p25": float(np.percentile(vals, 25)),
                    "p75": float(np.percentile(vals, 75)),
                    "n_valid": len(vals),
                }
        return profile

    def get_student_percentile(self, university: str, feature: str, value: float) -> float | None:
        """Student's percentile rank within a school's admitted cohort for a feature."""
        mask = (self.df["target_university"] == university) & (self.df["admitted"] == 1)
        vals = self.df.loc[mask, feature].dropna()
        if len(vals) < MIN_SAMPLES_FOR_PERCENTILE or value is None:
            return None
        return float((vals <= value).mean() * 100)

    def get_gap_analysis(self, university: str, student_values: dict) -> list[dict]:
        """Return gaps sorted by severity (largest negative gap first)."""
        profile = self.compute_school_profile(university)
        if profile is None or profile.get("insufficient"):
            return []

        gaps = []
        for feat in PROFILE_FEATURES:
            feat_info = profile["features"].get(feat, {})
            p50 = feat_info.get("p50")
            student_val = student_values.get(feat)
            if p50 is None or student_val is None:
                continue

            gap = student_val - p50
            percentile = self.get_student_percentile(university, feat, student_val)
            gaps.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "student_value": student_val,
                "p50": p50,
                "gap": round(gap, 3),
                "percentile": round(percentile) if percentile is not None else None,
            })

        gaps.sort(key=lambda g: g["gap"])
        return gaps

    def get_top_majors(self, university: str, top_n: int = 10) -> list[dict]:
        """Most common majors for a school in the training data."""
        mask = self.df["target_university"] == university
        counts = self.df.loc[mask, "target_major"].value_counts().head(top_n)
        return [{"major": k, "count": int(v)} for k, v in counts.items()]
