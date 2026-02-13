from __future__ import annotations

import math
from typing import Any

import numba


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


@numba.njit(cache=True)
def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


@numba.njit(cache=True)
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))
