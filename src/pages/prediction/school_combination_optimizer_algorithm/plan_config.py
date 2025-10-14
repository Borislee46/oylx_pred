from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class PlanConfig:
    name: str
    min_schools: int
    max_schools: int


DEFAULT_PLAN_CONFIGS: List[PlanConfig] = [
    PlanConfig(name="申请策略1", min_schools=6, max_schools=6),
    PlanConfig(name="申请策略2", min_schools=9, max_schools=9),
    PlanConfig(name="申请策略3", min_schools=10, max_schools=10),
]


def get_plan_configs(overrides: Iterable[PlanConfig] | None = None) -> List[PlanConfig]:
    if overrides:
        return list(overrides)
    return DEFAULT_PLAN_CONFIGS.copy()
