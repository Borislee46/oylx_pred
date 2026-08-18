from __future__ import annotations

import math

from src.utils.numeric.scalars import clip_probability

_DEFAULT_Z = 1.96


def wilson_score_ci(k: float, n: float, z: float = _DEFAULT_Z) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    p = clip_probability(p)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    lo = clip_probability(center - margin)
    hi = clip_probability(center + margin)
    return (lo, hi)
