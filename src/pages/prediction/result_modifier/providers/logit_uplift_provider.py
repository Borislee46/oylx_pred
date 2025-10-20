from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np

from src.pages.prediction.result_modifier.text_boost_provider import TextBoostProvider
from src.pages.prediction.result_modifier.utils import has_valid_experience_details


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-z)))


class LogitUpliftProvider(TextBoostProvider):
    def __init__(
        self,
        vectorizer_path: str,
        centroids_path: str,
        weights_path: str,
        max_total_boost: float = 0.05,
        sim_gate_sum_min: float | None = None,
        sim_gate_max_min: float | None = None,
        smoothing: float | None = None,
        cap_min_factor: float | None = None,
        cap_quality_gamma: float | None = None,
    ) -> None:
        self._vectorizer_path = vectorizer_path
        self._centroids_path = centroids_path
        self._weights_path = weights_path
        self._max_total_boost = float(max_total_boost)
        self._sim_gate_sum_min = 0.25 if sim_gate_sum_min is None else float(sim_gate_sum_min)
        self._sim_gate_max_min = 0.22 if sim_gate_max_min is None else float(sim_gate_max_min)
        self._smoothing = 0.5 if smoothing is None else float(smoothing)
        self._cap_min_factor = 0.4 if cap_min_factor is None else float(cap_min_factor)
        self._cap_quality_gamma = 1.0 if cap_quality_gamma is None else float(cap_quality_gamma)

        self._vectorizer = None
        self._centroids: dict[str, np.ndarray] | None = None
        self._weights: dict[str, float] | None = None
        self._weights_array: np.ndarray | None = None
        self._text_keys = (
            "research_details",
            "award_details",
            "internship_details",
            "paper_details",
        )
        self._count_keys = ("research_count", "award_count", "internship_count", "paper_count")

    def _lazy_load(self) -> None:
        if self._vectorizer is None:
            self._vectorizer = joblib.load(self._vectorizer_path)
        if self._centroids is None:
            data = np.load(self._centroids_path, mmap_mode="r")
            centroids = {k: data[k] for k in data.files}
            normed: dict[str, np.ndarray] = {}
            for k, arr in centroids.items():
                v = np.asarray(arr, dtype=np.float32)
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                normed[k] = v
            self._centroids = normed
        if self._weights is None:
            with open(self._weights_path, "r", encoding="utf-8") as f:
                self._weights = json.load(f) or {}
            self._weights_array = np.array(
                [
                    _safe_float(self._weights.get("b", 0.0)),
                    _safe_float(self._weights.get("w_r", 0.0)),
                    _safe_float(self._weights.get("w_a", 0.0)),
                    _safe_float(self._weights.get("w_i", 0.0)),
                    _safe_float(self._weights.get("w_p", 0.0)),
                    _safe_float(self._weights.get("u_r", 0.0)),
                    _safe_float(self._weights.get("u_a", 0.0)),
                    _safe_float(self._weights.get("u_i", 0.0)),
                    _safe_float(self._weights.get("u_p", 0.0)),
                ],
                dtype=np.float64,
            )

    @staticmethod
    def _prep_text(s: str | None) -> str:
        if not isinstance(s, str):
            return ""
        return s.strip()

    def _make_signature(self, details: dict[str, Any]) -> str:
        obj: dict[str, Any] = {k: self._prep_text(str(details.get(k, ""))) for k in self._text_keys}
        for k in self._count_keys:
            obj[k] = int(_safe_float(details.get(k, 0), 0))
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)

    def _compute_similarities(self, details: dict[str, Any]) -> dict[str, float]:
        self._lazy_load()
        assert self._vectorizer is not None
        assert self._centroids is not None

        texts = [self._prep_text(details.get(k, "")) for k in self._text_keys]

        if all(not t for t in texts):
            return {k: 0.0 for k in self._text_keys}

        X = self._vectorizer.transform(texts)

        sims: dict[str, float] = {}
        for idx, k in enumerate(self._text_keys):
            row = X.getrow(idx)
            if row.nnz == 0:
                sims[k] = 0.0
                continue
            centroid = self._centroids.get(k)
            if centroid is None or centroid.size == 0:
                sims[k] = 0.0
                continue
            dot_val = row.dot(centroid)
            dot_scalar = float(np.asarray(dot_val).flat[0])
            sims[k] = float(np.clip(dot_scalar, 0.0, 1.0))
        return sims

    @lru_cache(maxsize=512)
    def _cached_delta_logit(self, sig: str) -> tuple[float, dict[str, float]]:
        self._lazy_load()
        assert self._weights_array is not None

        try:
            details = json.loads(sig)
        except Exception:
            details = {}

        sims = self._compute_similarities(details)

        s_values = [sims.get(k, 0.0) for k in self._text_keys]
        ssum = sum(s_values)
        smax = max(s_values) if s_values else 0.0

        if ssum < self._sim_gate_sum_min or smax < self._sim_gate_max_min:
            return 0.0, sims

        counts = np.array(
            [_safe_float(details.get(k, 0)) for k in self._count_keys], dtype=np.float64
        )
        log_counts = np.log1p(counts)

        s_arr = np.array(s_values, dtype=np.float64)

        b, wr, wa, wi, wp, ur, ua, ui, up = self._weights_array

        delta = b
        delta += np.dot(self._weights_array[1:5], s_arr)
        delta += np.dot(self._weights_array[5:9], s_arr * log_counts)

        delta = max(0.0, float(delta))

        return delta, sims

    def apply(
        self, probabilities: list[float], experience_details: dict[str, Any]
    ) -> tuple[list[float], str]:
        if not probabilities:
            return probabilities, ""
        if not has_valid_experience_details(experience_details):
            return probabilities, ""

        sig = self._make_signature(experience_details)
        try:
            delta_logit, sims = self._cached_delta_logit(sig)
        except Exception:
            return probabilities, ""

        if delta_logit <= 0:
            return probabilities, ""

        effective_delta = delta_logit * self._smoothing

        s_values = [sims.get(k, 0.0) for k in self._text_keys]
        q_raw = 0.7 * max(s_values) + 0.3 * (sum(s_values) / len(s_values))
        q_adj = q_raw ** max(1.0, self._cap_quality_gamma)
        cap_factor = min(1.0, max(self._cap_min_factor, q_adj))

        updated: list[float] = []
        boosts: list[float] = []

        for p in probabilities:
            p0 = _safe_float(p, 0.0)
            if 0.1 <= p0 <= 0.9:
                new_p = _sigmoid(_logit(p0) + effective_delta)
                scale = 1.0 - 2.0 * abs(p0 - 0.5)
                cap_boost = self._max_total_boost * cap_factor * scale
                cap = p0 * (1.0 + cap_boost)
                new_p = min(new_p, cap, 1.0)
                updated.append(new_p)
                boosts.append((new_p / p0) - 1.0)
            else:
                updated.append(p0)

        summary = ""
        if boosts:
            parts: list[str] = []
            name_map = {
                "research_details": "科研项目",
                "award_details": "获奖情况",
                "internship_details": "实习经历",
                "paper_details": "论文发表",
            }
            for k in self._text_keys:
                s = sims.get(k, 0.0)
                if s > 0.15:
                    parts.append(f"{name_map[k]}: {s:.2f}")
            avg_boost = float(np.mean(boosts))
            summary = f"+{avg_boost:.1%} ({', '.join(parts)})" if avg_boost > 0 else ""

        return updated, summary
