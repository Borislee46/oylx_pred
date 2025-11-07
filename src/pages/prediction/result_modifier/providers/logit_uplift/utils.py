from __future__ import annotations

from typing import Any

import numpy as np


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-z)))
