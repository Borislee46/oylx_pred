from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np

from src.pages.prediction.result_modifier.providers.logit_uplift.model_loader import (
    ModelLoader,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.similarity_computer import (
    SimilarityComputer,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.text_processor import (
    TextProcessor,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class DeltaCalculator:
    def __init__(
        self,
        model_loader: ModelLoader,
        similarity_computer: SimilarityComputer,
        text_processor: TextProcessor,
        sim_gate_sum_min: float,
        sim_gate_max_min: float,
    ) -> None:
        self._model_loader = model_loader
        self._similarity_computer = similarity_computer
        self._text_processor = text_processor
        self._sim_gate_sum_min = sim_gate_sum_min
        self._sim_gate_max_min = sim_gate_max_min
        self._get_delta_logit_cached = lru_cache(maxsize=512)(self._compute_delta_logit_raw)

    def _compute_delta_logit_raw(
        self, sig: str
    ) -> tuple[float, tuple[tuple[str, float], ...], tuple[tuple[str, tuple[str, ...]], ...]]:
        weights = self._model_loader.weights_array
        text_keys = self._text_processor.text_keys
        count_keys = self._text_processor.count_keys
        compute_sims = self._similarity_computer.compute_similarities
        sum_min = self._sim_gate_sum_min
        max_min = self._sim_gate_max_min
        _log1p = math.log1p

        details = {}
        if sig and sig.startswith("{"):
            details = json.loads(sig)

        sims, remarks = compute_sims(details)
        if not sims:
            return 0.0, (), ()

        n_text = len(text_keys)
        s_values = [0.0] * n_text
        ssum = 0.0
        smax = 0.0
        sims_get = sims.get

        for i in range(n_text):
            val = sims_get(text_keys[i], 0.0)
            if val:
                s_values[i] = val
                ssum += val
                if val > smax:
                    smax = val

        if ssum < sum_min or smax < max_min:
            return 0.0, tuple(sims.items()), tuple((k, tuple(v)) for k, v in remarks.items())

        delta = float(weights[0])
        tw_start = 1

        text_w = weights[tw_start : tw_start + n_text]
        n_counts = len(count_keys)
        has_inter = n_counts == n_text and len(weights) >= tw_start + 2 * n_text
        inter_w = weights[tw_start + n_text : tw_start + 2 * n_text] if has_inter else None
        details_get = details.get

        sims_adj = {}
        for i in range(n_text):
            s = s_values[i]
            if s <= 0:
                continue

            txt = details_get(text_keys[i], "")
            richness = _fast_entropy(txt)

            s_adj = float(s * richness)
            sims_adj[text_keys[i]] = s_adj
            delta += text_w[i] * s_adj

            if has_inter:
                v = details_get(count_keys[i])
                if v:
                    try:
                        fv = float(v)
                        if fv > 0:
                            delta += inter_w[i] * s_adj * _log1p(fv * richness)
                    except (TypeError, ValueError):
                        pass

        final_delta = delta if delta > 0.0 else 0.0

        if final_delta > 0 and remarks:
            flat_remarks = []
            for field, tags in remarks.items():
                field_cn = {
                    "research_details": "科研",
                    "award_details": "奖项",
                    "internship_details": "实习",
                    "paper_details": "论文",
                }.get(field, field)
                if tags:
                    flat_remarks.append(f"{field_cn}: {', '.join(tags)}")

            if flat_remarks:
                logger.info(
                    f"[背提文本加成算法] Logit+{final_delta:.3f}: {'; '.join(flat_remarks)}"
                )

        return (
            final_delta,
            tuple(sims_adj.items()),
            tuple((k, tuple(v)) for k, v in remarks.items()),
        )

    def cached_delta_logit(self, sig: str) -> tuple[float, dict[str, float], dict[str, list[str]]]:
        delta, sims_tuple, remarks_tuple = self._get_delta_logit_cached(sig)
        return delta, dict(sims_tuple), {k: list(v) for k, v in remarks_tuple}


def _fast_entropy(text: str) -> float:
    if not text:
        return 0.0
    try:
        b = text.encode("utf-8")
    except UnicodeEncodeError:
        return 0.0
    if len(b) < 10:
        return 0.0
    counts = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(b)
    entropy = -np.sum(probs * np.log2(probs))
    byte_rich = float(np.clip(entropy / 5.0, 0.0, 1.0))

    n = len(text)
    if n >= 12:
        span = max(12.0, float(n**0.55))
        char_f = float(np.clip(len(set(text)) / span, 0.0, 1.0))
        byte_rich *= 0.35 + 0.65 * char_f

    return float(np.clip(byte_rich, 0.0, 1.0))
