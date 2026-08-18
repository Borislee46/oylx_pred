from __future__ import annotations

import math
from collections.abc import Callable
from itertools import combinations

from src.utils.numeric import PROB_EPS, clip_scalar, logit, prob_to_pct, sigmoid


def fixed_beta(fixed_pp: float, base_prob: float) -> float:
    if fixed_pp <= 0:
        return 0.0
    return logit(clip_scalar(base_prob + fixed_pp / 100.0, PROB_EPS, 0.99)) - logit(base_prob)


def shapley_values(
    products: list[str],
    value: Callable[[frozenset[str]], float],
) -> dict[str, float]:
    n = len(products)
    phi = dict.fromkeys(products, 0.0)
    if n == 0:
        return phi
    fact = [math.factorial(i) for i in range(n + 1)]
    for p in products:
        rest = [q for q in products if q != p]
        for k in range(len(rest) + 1):
            weight = fact[k] * fact[n - k - 1] / fact[n]
            for combo in combinations(rest, k):
                s = frozenset(combo)
                phi[p] += weight * (value(s | {p}) - value(s))
    return phi


def attribute_selection(
    products: list[str],
    *,
    base_prob: float,
    pipeline_prob_of: Callable[[frozenset[str]], float],
    fixed_pp: dict[str, float],
    pipeline_names: set[str],
) -> dict[str, object]:
    betas = {
        n: fixed_beta(fixed_pp.get(n, 0.0), base_prob) for n in products if n not in pipeline_names
    }

    def value(subset: frozenset[str]) -> float:
        z = logit(pipeline_prob_of(subset))
        for n in subset:
            if n not in pipeline_names:
                z += betas.get(n, 0.0)
        return sigmoid(z)

    phi = shapley_values(products, value)
    final = value(frozenset(products))
    return {
        "base_pct": prob_to_pct(base_prob),
        "final_pct": prob_to_pct(final),
        "contributions": {n: phi[n] * 100 for n in products},
    }
