"""Admission competitiveness positioning panel.

Renders a difficulty bar with clickable school circles. All school-switching
is client-side JS — zero Python reruns on selection change.
Inserted after hero_summary in the prediction results page.
"""

from __future__ import annotations

import numpy as np

from src.agent.schemas import compute_tiers
from src.pages.prediction.result_display._comp_core import (
    MIN_SAMPLES_FOR_PROFILE,
    PROFILE_FEATURES,
    compute_school_difficulty,
    get_tier_thresholds,
    safe_float,
)
from src.pages.prediction.result_display.competitiveness_html import render_card


def render_competitiveness_panel(
    candidates: list[dict],
    input_data: dict,
    cases_df,
) -> None:
    if not candidates or len(candidates) < 2:
        return

    schools = _dedupe_by_university(candidates)
    if len(schools) < 2:
        return

    student = _extract_student(input_data)
    profiles = _compute_profiles(schools, cases_df)
    if len(profiles) < 2:
        return

    # Compute tiers via difficulty-weighted thresholds
    difficulty_map = compute_school_difficulty(cases_df)
    probs = [safe_float(s.get("probability")) for s in schools]
    diffs = {i: difficulty_map.get(s["university"], 0.5) for i, s in enumerate(schools)}
    tier_labels = compute_tiers(probs, difficulties=diffs)
    tier_map = {s["university"]: tier_labels[i] for i, s in enumerate(schools)}

    # Per-school (safety, target) thresholds for dynamic bar zones
    threshold_positions: dict[str, tuple[float, float]] = {}
    for i, s in enumerate(schools):
        d = diffs.get(i, 0.5)
        threshold_positions[s["university"]] = get_tier_thresholds(d)

    selected = schools[-1]["university"]

    render_card(schools, profiles, student, selected, tier_map, threshold_positions)


def _dedupe_by_university(candidates: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for c in candidates:
        u = c.get("university") or ""
        if not u:
            continue
        p = safe_float(c.get("probability"))
        if u not in best or p > safe_float(best[u].get("probability")):
            best[u] = c
    return sorted(best.values(), key=lambda r: safe_float(r.get("probability")), reverse=True)


def _extract_student(input_data: dict) -> dict:
    return {
        "gpa": input_data.get("gpa"),
        "language_score": input_data.get("language_score"),
        "research_count": safe_float(input_data.get("research_count")),
        "internship_count": safe_float(input_data.get("internship_count")),
    }


def _compute_profiles(schools: list[dict], df) -> dict:
    profiles = {}
    for s in schools:
        u = s["university"]
        mask = (df["target_university"] == u) & (df["admitted"] == 1)
        cohort = df[mask]
        n = len(cohort)
        if n < MIN_SAMPLES_FOR_PROFILE:
            continue
        feats = {}
        for feat in PROFILE_FEATURES:
            if feat not in cohort.columns:
                feats[feat] = {"p50": None, "n_valid": 0}
                continue
            vals = cohort[feat].dropna()
            if len(vals) >= MIN_SAMPLES_FOR_PROFILE:
                feats[feat] = {"p50": float(np.percentile(vals, 50)), "n_valid": len(vals)}
            else:
                feats[feat] = {"p50": None, "n_valid": len(vals)}
        profiles[u] = {"n_admitted": n, "features": feats}
    return profiles
