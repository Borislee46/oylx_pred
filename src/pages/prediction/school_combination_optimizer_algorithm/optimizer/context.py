from dataclasses import dataclass
from typing import Any, Optional

from src.pages.prediction.school_combination_optimizer_algorithm.problem import (
    SchoolSelectionProblem,
)


@dataclass
class OptimizationContext:
    all_schools_data: list[dict[str, Any]]
    background_major: str
    background_faculty: Optional[str]
    school_level: Optional[str] = None
    gpa: Optional[float] = None
    adaptive_thresholds: Optional[dict[str, float]] = None
    problem: Optional[SchoolSelectionProblem] = None
    major_category_cache: Optional[dict] = None

