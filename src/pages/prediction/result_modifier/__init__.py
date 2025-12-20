from __future__ import annotations

from src.pages.prediction.result_modifier.adjustment_pipeline import (
    AdjustmentContext,
    ProbabilityAdjustmentPipeline,
)
from src.pages.prediction.result_modifier.config import (
    HIGHER_SIMILARITY_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD,
    UNIVERSITY_COUNT_THRESHOLD,
)
from src.pages.prediction.result_modifier.probability_adjuster import (
    ProbabilityAdjuster,
    penalize_cross_major_without_cases,
)
from src.pages.prediction.result_modifier.similarity_adjuster import (
    adjust_similarity_score,
)

__all__ = [
    "ProbabilityAdjuster",
    "ProbabilityAdjustmentPipeline",
    "AdjustmentContext",
    "adjust_similarity_score",
    "MIN_SIMILARITY_THRESHOLD",
    "HIGHER_SIMILARITY_THRESHOLD",
    "UNIVERSITY_COUNT_THRESHOLD",
    "penalize_cross_major_without_cases",
]
