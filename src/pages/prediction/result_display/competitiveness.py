from __future__ import annotations

import time

import numpy as np

from src.agent.schemas import compute_tiers
from src.pages.prediction.result_display._comp_core import (
    MIN_SAMPLES_FOR_PROFILE,
    PROFILE_FEATURES,
    compute_school_difficulty,
    get_tier_thresholds,
)
from src.pages.prediction.result_display.competitiveness_html import render_card
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce, float_or_none, safe_float

_logger = setup_logger("page3", "prediction")


def render_competitiveness_panel(
    candidates: list[dict],
    input_data: dict,
    cases_df,
) -> None:
    t0 = time.perf_counter()
    if not candidates or len(candidates) < 2:
        _logger.info("render_competitiveness_panel: skipped (candidates=%d)", len(candidates or []))
        return

    schools = _dedupe_by_university(candidates)
    if len(schools) < 2:
        _logger.info("render_competitiveness_panel: skipped (unique_schools=%d)", len(schools))
        return

    student = _extract_student(input_data)

    t1 = time.perf_counter()
    profiles = _compute_profiles(schools, cases_df)
    if len(profiles) < 2:
        _logger.info("render_competitiveness_panel: skipped (profiles=%d, need≥2)", len(profiles))
        return

    _logger.info(
        "render_competitiveness_panel: %d schools → %d profiles (%.0fms)",
        len(schools),
        len(profiles),
        (time.perf_counter() - t1) * 1000,
    )

    probs = [clip_probability_coerce(s.get("probability")) for s in schools]
    tier_labels = compute_tiers(probs)
    tier_map = {s["university"]: tier_labels[i] for i, s in enumerate(schools)}

    difficulty_map = compute_school_difficulty(cases_df)
    threshold_positions: dict[str, tuple[float, float]] = {}
    for _i, s in enumerate(schools):
        d = difficulty_map.get(s["university"], 0.5)
        threshold_positions[s["university"]] = get_tier_thresholds(d)

    selected = schools[0]["university"]

    schools_in_bar = [s for s in schools if s["university"] in profiles]
    missing = len(schools) - len(schools_in_bar)
    if missing:
        _logger.info(
            "render_competitiveness_panel: %d/%d schools missing profile data",
            missing,
            len(schools),
        )

    render_card(schools, profiles, student, selected, tier_map, threshold_positions, missing)
    _logger.info("render_competitiveness_panel: done in %.0fms", (time.perf_counter() - t0) * 1000)


def _dedupe_by_university(candidates: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for c in candidates:
        u = c.get("university") or ""
        if not u:
            continue
        p = clip_probability_coerce(c.get("probability"))
        if u not in best or p > clip_probability_coerce(best[u].get("probability")):
            best[u] = c
    return sorted(
        best.values(), key=lambda r: clip_probability_coerce(r.get("probability")), reverse=True
    )


def _extract_student(input_data: dict) -> dict:
    return {
        "gpa": float_or_none(input_data.get("gpa")),
        "language_score": float_or_none(input_data.get("language_score")),
        "research_count": safe_float(input_data.get("research_count")),
        "paper_count": safe_float(input_data.get("paper_count")),
        "internship_count": safe_float(input_data.get("internship_count")),
        "award_count": safe_float(input_data.get("award_count")),
    }


def _compute_profiles(schools: list[dict], df) -> dict:
    if "target_university" not in df.columns or "admitted" not in df.columns:
        return {}

    universities = {s["university"] for s in schools}
    admitted = df[(df["admitted"] == 1) & (df["target_university"].isin(universities))]
    # noqa: C416 —— 不能用 dict(admitted.groupby(...))：pandas 2.3.1 下 dict() 对
    # DataFrameGroupBy 抛 TypeError('str' object is not callable)，推导式等价且安全。
    cohorts = {u: g for u, g in admitted.groupby("target_university")}  # noqa: C416

    profiles = {}
    skipped = 0
    for s in schools:
        u = s["university"]
        cohort = cohorts.get(u)
        n = len(cohort) if cohort is not None else 0
        if n < MIN_SAMPLES_FOR_PROFILE:
            skipped += 1
            continue
        feats = {}
        for feat in PROFILE_FEATURES:
            if feat not in cohort.columns:
                feats[feat] = {"p25": None, "p50": None, "p75": None, "n_valid": 0}
                continue
            vals = cohort[feat].dropna()
            if len(vals) >= MIN_SAMPLES_FOR_PROFILE:
                q25, q50, q75 = np.percentile(vals, [25, 50, 75])
                feats[feat] = {
                    "p25": float(q25),
                    "p50": float(q50),
                    "p75": float(q75),
                    "n_valid": len(vals),
                }
            else:
                feats[feat] = {"p25": None, "p50": None, "p75": None, "n_valid": len(vals)}
        profiles[u] = {"n_admitted": n, "features": feats}
    if skipped:
        _logger.info(
            "_compute_profiles: %d/%d schools skipped (<%d samples)",
            skipped,
            len(schools),
            MIN_SAMPLES_FOR_PROFILE,
        )
    return profiles
