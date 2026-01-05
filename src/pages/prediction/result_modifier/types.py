from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias, TypedDict, TypeGuard

CaseKey: TypeAlias = tuple[str, str]


class AdjustmentFactorType(str, Enum):
    PENALTY = "penalty"
    BOOST = "boost"


@dataclass
class AdjustmentFactor:
    name: str
    value: float
    factor_type: AdjustmentFactorType
    description: str = ""
    weight: float = 1.0


class CaseWithKey(TypedDict, total=False):
    university: str
    major: str
    similarity: float
    probability: float
    faculty: str


class AdjustmentDecision(str, Enum):
    DEFER_TO_AGENT = "defer_to_agent"
    ADJUST = "adjust"
    NO_ADJUST = "no_adjust"


def is_case_with_key(x: Any) -> TypeGuard[CaseWithKey]:
    return (
        isinstance(x, dict)
        and isinstance(x.get("university"), str)
        and isinstance(x.get("major"), str)
    )


def case_key(case: Any) -> CaseKey | None:
    return case["university"], case["major"]


def get_similarity(case: dict[str, Any]) -> float:
    v = case.get("similarity", 0.0)
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0
