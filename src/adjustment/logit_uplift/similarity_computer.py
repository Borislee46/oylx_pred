from __future__ import annotations

from typing import Any

import numpy as np

from src.adjustment.logit_uplift.model_loader import (
    ModelLoader,
)
from src.adjustment.logit_uplift.signal_scorer import (
    SignalScorer,
)
from src.adjustment.logit_uplift.text_processor import (
    TextProcessor,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability

logger = setup_logger("page3", "prediction")


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
        base = clip_probability(base)
        bonus = clip_probability(bonus)
        return 1.0 - (1.0 - base) * (1.0 - bonus)

    def _compute_novelty_bonus(self, text: str, row: Any) -> float:
        if self._novelty_weight <= 0:
            return 0.0
        if not isinstance(text, str) or len(text.strip()) < self._novelty_min_chars:
            return 0.0
        if row is None or getattr(row, "data", None) is None or row.data.size == 0:
            return 0.0

        max_val = float(np.max(row.data))
        raw = clip_probability((max_val - 0.18) / 0.35)
        return clip_probability(raw * self._novelty_weight)

    def compute_similarities(
        self, details: dict[str, Any]
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        vectorizer = self._model_loader.vectorizer
        centroids = self._model_loader.centroids
        text_keys = self._text_processor.text_keys

        texts = [self._text_processor.prep_text(details.get(k, "")) for k in text_keys]
        if all(not t for t in texts):
            logger.debug("文本相似度: 全部为空 | text_keys=%s", text_keys)
            return dict.fromkeys(text_keys, 0.0), {}

        X = vectorizer.transform(texts)
        logger.debug(
            "文本相似度计算 | n_texts=%d n_features=%d",
            len(texts),
            X.shape[1],
        )

        lex_bonuses: dict[str, float] = {}
        lex_tags: dict[str, list[str]] = {}
        if self._signal_scorer is not None:
            lex_bonuses, lex_tags = self._signal_scorer.score(
                {k: texts[idx] for idx, k in enumerate(text_keys)}
            )

        sims: dict[str, float] = {}
        remarks: dict[str, list[str]] = {}

        for idx, k in enumerate(text_keys):
            row = X.getrow(idx)
            current_remarks = []

            if row.nnz == 0:
                sims[k] = 0.0
                continue

            centroid = centroids.get(k)
            if centroid is None or centroid.size == 0:
                sims[k] = 0.0
                continue

            dot_val = row.dot(centroid)
            dot_scalar = float(np.asarray(dot_val).flat[0])
            s0 = clip_probability(dot_scalar)

            if k in lex_tags:
                current_remarks.extend(lex_tags[k])

            novelty_bonus = self._compute_novelty_bonus(texts[idx], row)
            if novelty_bonus > 0.001:
                current_remarks.append("content_novelty")

            bonus = float(lex_bonuses.get(k, 0.0)) + novelty_bonus
            sims[k] = self._bounded_fuse(s0, bonus)

            if current_remarks:
                remarks[k] = current_remarks

        logger.debug(
            "文本相似度完成 | sims=%s n_remarks=%d",
            {k: round(v, 4) for k, v in sims.items()},
            sum(len(v) for v in remarks.values()),
        )
        return sims, remarks
