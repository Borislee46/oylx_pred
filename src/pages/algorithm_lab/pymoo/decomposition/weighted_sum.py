import numpy as np

from src.pages.algorithm_lab.pymoo.core.decomposition import Decomposition


class WeightedSum(Decomposition):
    def _do(self, F, weights, **kwargs):
        return np.sum(F * weights, axis=1)
