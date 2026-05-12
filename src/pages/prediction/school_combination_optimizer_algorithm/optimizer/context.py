# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
from dataclasses import dataclass
from typing import Any

from src.pages.prediction.school_combination_optimizer_algorithm.problem import (
    SchoolSelectionProblem,
)


@dataclass
class OptimizationContext:
    all_schools_data: list[dict[str, Any]]
    background_major: str
    background_faculty: str | None
    school_level: str | None = None
    gpa: float | None = None
    adaptive_thresholds: dict[str, float] | None = None
    problem: SchoolSelectionProblem | None = None
    major_category_cache: dict | None = None
