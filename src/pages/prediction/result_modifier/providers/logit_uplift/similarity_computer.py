from __future__ import annotations

from typing import Any

import numpy as np

from src.pages.prediction.result_modifier.providers.logit_uplift.model_loader import (
    ModelLoader,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.signal_scorer import (
    SignalScorer,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.text_processor import (
    TextProcessor,
)
from src.pages.prediction.result_modifier.utils import clip_basic


class SimilarityComputer:
    def __init__(
        self,
        model_loader: ModelLoader,
        text_processor: TextProcessor,
        signal_scorer: SignalScorer | None = None,
        novelty_weight: float = 0.0,
        novelty_min_chars: int = 12,
    ) -> None:
        self._model_loader = model_loader
        self._text_processor = text_processor
        self._signal_scorer = signal_scorer
        self._novelty_weight = float(novelty_weight)
        self._novelty_min_chars = int(novelty_min_chars)

    @staticmethod
    def _bounded_fuse(base: float, bonus: float) -> float:
        # 比np.clip快
        base = clip_basic(base)
        bonus = clip_basic(bonus)
        return 1.0 - (1.0 - base) * (1.0 - bonus)

    def _compute_novelty_bonus(self, text: str, row: Any) -> float:
        if self._novelty_weight <= 0:
            return 0.0
        if not isinstance(text, str) or len(text.strip()) < self._novelty_min_chars:
            return 0.0
        if row is None or getattr(row, "data", None) is None or row.data.size == 0:
            return 0.0
        max_val = float(np.max(row.data))
        raw = clip_basic((max_val - 0.18) / 0.35)
        return clip_basic(raw * self._novelty_weight)

    def compute_similarities(
        self, details: dict[str, Any]
    ) -> tuple[dict[str, float], tuple[str, ...]]:
        vectorizer = self._model_loader.vectorizer
        centroids = self._model_loader.centroids
        text_keys = self._text_processor.text_keys

        texts = [self._text_processor.prep_text(details.get(k, "")) for k in text_keys]

        if all(not t for t in texts):
            return dict.fromkeys(text_keys, 0.0), ()

        X = vectorizer.transform(texts)

        lex_bonuses: dict[str, float] = {}
        reasons: tuple[str, ...] = ()
        if self._signal_scorer is not None:
            lex_bonuses, reasons = self._signal_scorer.score(
                {k: texts[idx] for idx, k in enumerate(text_keys)}
            )

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
            s0 = clip_basic(dot_scalar)
            bonus = float(lex_bonuses.get(k, 0.0)) + self._compute_novelty_bonus(texts[idx], row)
            sims[k] = self._bounded_fuse(s0, bonus)
        return sims, reasons
