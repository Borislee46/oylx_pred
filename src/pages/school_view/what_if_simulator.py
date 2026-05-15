"""Counterfactual simulator: perturb features → observe probability deltas via XGBoost."""

from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.modeling.model import PredictionModel


SCENARIOS = [
    {"key": "baseline", "label": "当前", "feature": None, "delta": 0},
    {"key": "gpa_02", "label": "GPA +0.2", "feature": "gpa", "delta": 0.2},
    {"key": "gpa_04", "label": "GPA +0.4", "feature": "gpa", "delta": 0.4},
    {"key": "lang_005", "label": "语言 +0.5 (约IELTS+0.5)", "feature": "language_score", "delta": 0.05},
    {"key": "research_1", "label": "科研 +1段", "feature": "research_count", "delta": 1},
    {"key": "internship_1", "label": "实习 +1段", "feature": "internship_count", "delta": 1},
]

TOP_MAJORS_PER_SCHOOL = 5


class WhatIfSimulator:
    def __init__(self):
        from src.pages.prediction.page_data_loader import machine_learning_model

        self.mlm = machine_learning_model.resource_loader()
        self.model: PredictionModel = self.mlm.prediction_model
        self.feature_names: list[str] = (
            self.mlm.loaded_feature_names if self.mlm.loaded_feature_names else []
        )
        self.cases_df: pd.DataFrame = self.mlm.cases_df

    def _get_top_combinations(
        self, schools: list[str]
    ) -> list[tuple[str, str]]:
        """Return top (university, major) combos per selected school from training data."""
        combos = []
        df = self.cases_df
        for school in schools:
            mask = df["target_university"] == school
            top = df.loc[mask, "target_major"].value_counts().head(TOP_MAJORS_PER_SCHOOL)
            for major in top.index:
                combos.append((school, major))
        return combos

    def _build_input_data(self, form_data: dict) -> dict[str, Any]:
        return {
            "background_university": form_data.get("background_university", ""),
            "background_major": form_data.get("background_major", ""),
            "gpa": form_data.get("gpa") or 0,
            "research_count": float(form_data.get("research_count", 0) or 0),
            "paper_count": float(form_data.get("paper_count", 0) or 0),
            "internship_count": float(form_data.get("internship_count", 0) or 0),
            "award_count": float(form_data.get("award_count", 0) or 0),
            "language_score": form_data.get("language_score") or 0,
        }

    def simulate(
        self, form_data: dict, schools: list[str]
    ) -> dict[str, Any]:
        """Run all scenarios and return per-school probability deltas."""
        if not schools:
            return {}

        combinations = self._get_top_combinations(schools)
        if not combinations:
            return {}

        base_input = self._build_input_data(form_data)

        # Run baseline + all scenarios
        all_probas: dict[str, list[dict]] = {}
        for scenario in SCENARIOS:
            modified_input = dict(base_input)
            feat = scenario["feature"]
            if feat and feat in modified_input:
                modified_input[feat] = base_input[feat] + scenario["delta"]
            try:
                results = self.model.predict_batch(modified_input, combinations, self.feature_names)
                all_probas[scenario["key"]] = results
            except Exception:
                all_probas[scenario["key"]] = []

        # Aggregate per school (average across top majors)
        school_baseline = self._aggregate_by_school(all_probas.get("baseline", []))
        school_scenarios: dict[str, dict[str, float]] = {}
        for key, results in all_probas.items():
            school_scenarios[key] = self._aggregate_by_school(results)

        return {
            "combinations": combinations,
            "school_baseline": school_baseline,
            "school_scenarios": school_scenarios,
        }

    def _aggregate_by_school(self, results: list[dict]) -> dict[str, float]:
        """Average probability per school from a list of (university, major, probability) results."""
        school_probs: dict[str, list[float]] = {}
        for r in results:
            u = r.get("university", "")
            p = r.get("probability", 0) or 0
            school_probs.setdefault(u, []).append(float(p))
        return {u: sum(ps) / len(ps) for u, ps in school_probs.items()}

    def compute_roi_table(
        self, simulate_result: dict
    ) -> pd.DataFrame:
        """Build a per-school ROI table showing probability deltas for each scenario."""
        baseline = simulate_result.get("school_baseline", {})
        scenarios = simulate_result.get("school_scenarios", {})
        schools = list(baseline.keys())

        rows = []
        for school in schools:
            base_p = baseline.get(school, 0)
            row = {"学校": school, "当前概率": f"{base_p:.1%}"}
            for sc in SCENARIOS:
                if sc["key"] == "baseline":
                    continue
                sc_prob = scenarios.get(sc["key"], {}).get(school, base_p)
                delta = sc_prob - base_p
                row[sc["label"]] = f"{sc_prob:.1%} ({delta:+.1%})"
            rows.append(row)

        return pd.DataFrame(rows)
