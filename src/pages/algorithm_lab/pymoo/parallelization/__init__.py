"""
Parallelization utilities for pymoo.
"""

from .dask import DaskParallelization
from .joblib import JoblibParallelization
from .ray import RayParallelization
from .starmap import StarmapParallelization

__all__ = [
    "StarmapParallelization",
    "DaskParallelization",
    "JoblibParallelization",
    "RayParallelization",
]
