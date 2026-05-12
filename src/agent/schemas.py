"""Shared types, tier thresholds, and tier classification logic.

- TypedDict schemas for Agent data contracts
- ``compute_tiers()`` — Jenks natural-breaks tiering (shared by AI report,
  hero summary, application prompts)
- ``TIER_THRESHOLD_*`` — fallback absolute thresholds (used when n < 3)
"""

from __future__ import annotations

from typing import TypedDict


class ExtractedBackground(TypedDict, total=False):
    """Output schema for LeadInAgent.extracted_background.

    All keys are optional — the agent fills whatever it can infer from the
    consultant's free-text input. Missing keys mean the agent hasn't
    extracted that piece of information yet.
    """

    university: str
    major: str
    gpa: float
    language_score: float
    language_type: str
    standardized_test_type: str
    standardized_test_score: float
    country: str
    target_schools: list[str]
    target_majors: list[str]
    research: str
    internship: str
    award: str
    paper: str


class LeadInResult(TypedDict, total=False):
    """Return type for LeadInAgent.run()."""

    extracted_info: ExtractedBackground
    quick_assessment: str
    suggested_questions: list[str]
    _error: str


class ExplainResult(TypedDict, total=False):
    """Output of ExplainAgent."""

    overview: str
    strengths: list[str]
    concerns: list[str]
    summary: str
    school_notes: list[dict[str, object]]
    products: list[dict[str, object]]
    _error: str
    _ts: float


TIER_THRESHOLD_SAFETY = 0.55
TIER_THRESHOLD_MATCH = 0.30


# ── Dynamic tier classification ──────────────────────────────────────────


def _jenks_breaks_1d(values: list[float], k: int = 3) -> list[int]:
    """1D Jenks natural breaks — return k-1 split indices. O(kn²)."""
    n = len(values)
    if n <= k:
        return list(range(1, n))

    pref = [0.0] * (n + 1)
    pref2 = [0.0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + values[i]
        pref2[i + 1] = pref2[i] + values[i] ** 2

    def _ssd(lo: int, hi: int) -> float:
        cnt = hi - lo
        if cnt <= 0:
            return 0.0
        s = pref[hi] - pref[lo]
        s2 = pref2[hi] - pref2[lo]
        return s2 - s * s / cnt

    best_ssd = float("inf")
    best = None
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            ssd = _ssd(0, i) + _ssd(i, j) + _ssd(j, n)
            if ssd < best_ssd:
                best_ssd = ssd
                best = (i, j)
    return list(best) if best else [n // 3, 2 * n // 3]


def compute_tiers(probs: list[float]) -> list[str]:
    """Return tier label per probability using Jenks breaks (n≥3) or absolute
    thresholds (n<3). Labels: 保底 / 适中 / 冲刺."""
    n = len(probs)
    if n < 3:
        labels = []
        for p in probs:
            if p >= TIER_THRESHOLD_SAFETY:
                labels.append("保底")
            elif p >= TIER_THRESHOLD_MATCH:
                labels.append("适中")
            else:
                labels.append("冲刺")
        return labels

    sorted_probs = sorted(probs)
    breaks = _jenks_breaks_1d(sorted_probs, k=3)
    i, j = breaks[0], breaks[1]

    abv = dict(zip(sorted_probs, [""] * n, strict=True))
    for idx in range(n):
        if idx < i:
            abv[sorted_probs[idx]] = "冲刺"
        elif idx < j:
            abv[sorted_probs[idx]] = "适中"
        else:
            abv[sorted_probs[idx]] = "保底"

    return [abv[p] for p in probs]
