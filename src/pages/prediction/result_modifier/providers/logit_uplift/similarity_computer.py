from __future__ import annotations

from typing import Any

import numpy as np

from src.pages.prediction.result_modifier.providers.logit_uplift.model_loader import (
    ModelLoader,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.text_processor import (
    TextProcessor,
)


class SimilarityComputer:
    def __init__(
        self,
        model_loader: ModelLoader,
        text_processor: TextProcessor,
    ) -> None:
        self._model_loader = model_loader
        self._text_processor = text_processor

    def compute_similarities(self, details: dict[str, Any]) -> dict[str, float]:
        vectorizer = self._model_loader.vectorizer
        centroids = self._model_loader.centroids
        text_keys = self._text_processor.text_keys

        texts = [self._text_processor.prep_text(details.get(k, "")) for k in text_keys]

        if all(not t for t in texts):
            return {k: 0.0 for k in text_keys}

        X = vectorizer.transform(texts)

        sims: dict[str, float] = {}
        for idx, k in enumerate(text_keys):
            row = X.getrow(idx)
            if row.nnz == 0:
                sims[k] = 0.0
                continue
            centroid = centroids.get(k)
            if centroid is None or centroid.size == 0:
                sims[k] = 0.0
                continue
            dot_val = row.dot(centroid)
            dot_scalar = float(np.asarray(dot_val).flat[0])
            sims[k] = float(np.clip(dot_scalar, 0.0, 1.0))
        return sims
