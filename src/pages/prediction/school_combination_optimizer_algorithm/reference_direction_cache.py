from functools import lru_cache

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions


@lru_cache(maxsize=32)
def _fetch_reference_directions(method: str, n_dim: int, n_points: int) -> np.ndarray:
    return get_reference_directions(method, n_dim=n_dim, n_points=n_points)


def get_cached_reference_directions(method: str, n_dim: int, n_points: int) -> np.ndarray:
    ref_dirs = _fetch_reference_directions(method, n_dim, n_points)
    return ref_dirs.copy()
