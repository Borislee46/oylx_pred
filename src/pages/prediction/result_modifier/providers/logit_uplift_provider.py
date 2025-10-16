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

    @staticmethod
    def _prep_text(s: str | None) -> str:
        if not isinstance(s, str):
            return ""
        return s.strip()

    def _make_signature(self, details: dict[str, Any]) -> str:
        keys_text = (
            "research_details",
            "award_details",
            "internship_details",
            "paper_details",
        )
        keys_cnt = ("research_count", "award_count", "internship_count", "paper_count")
        obj: dict[str, Any] = {k: self._prep_text(str(details.get(k, ""))) for k in keys_text}
        for k in keys_cnt:
            obj[k] = int(_safe_float(details.get(k, 0), 0))
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)

    def _compute_similarities(self, details: dict[str, Any]) -> dict[str, float]:
        self._lazy_load()
        assert self._vectorizer is not None
        assert self._centroids is not None

        texts = {
            k: self._prep_text(details.get(k, ""))
            for k in (
                "research_details",
                "award_details",
                "internship_details",
                "paper_details",
            )
        }
        keys = list(texts.keys())
        X = self._vectorizer.transform([texts[k] for k in keys])

        sims: dict[str, float] = {}
        for idx, k in enumerate(keys):
            row = X.getrow(idx)
            if row is None:
                sims[k] = 0.0
                continue
            data = getattr(row, "data", None)
            if data is None or getattr(data, "size", 0) == 0:
                sims[k] = 0.0
                continue
            centroid = self._centroids.get(k)
            if centroid is None or centroid.size == 0:
                sims[k] = 0.0
                continue
            dot_val = row.dot(centroid)
            try:
                dot_scalar = float(np.asarray(dot_val).ravel()[0])
            except Exception:
                dot_scalar = 0.0
            sims[k] = float(np.clip(dot_scalar, 0.0, 1.0))
        return sims

    @lru_cache(maxsize=512)
    def _cached_delta_logit(self, sig: str) -> tuple[float, dict[str, float]]:
        self._lazy_load()
        assert self._weights is not None

        try:
            details = json.loads(sig)
        except Exception:
            details = {}

        sims = self._compute_similarities(details)

        rc = _safe_float(details.get("research_count", 0))
        ac = _safe_float(details.get("award_count", 0))
        ic = _safe_float(details.get("internship_count", 0))
        pc = _safe_float(details.get("paper_count", 0))

        b = _safe_float(self._weights.get("b", 0.0))
        wr = _safe_float(self._weights.get("w_r", 0.0))
        wa = _safe_float(self._weights.get("w_a", 0.0))
        wi = _safe_float(self._weights.get("w_i", 0.0))
        wp = _safe_float(self._weights.get("w_p", 0.0))
        ur = _safe_float(self._weights.get("u_r", 0.0))
        ua = _safe_float(self._weights.get("u_a", 0.0))
        ui = _safe_float(self._weights.get("u_i", 0.0))
        up = _safe_float(self._weights.get("u_p", 0.0))

        sr = _safe_float(sims.get("research_details", 0.0))
        sa = _safe_float(sims.get("award_details", 0.0))
        si = _safe_float(sims.get("internship_details", 0.0))
        sp = _safe_float(sims.get("paper_details", 0.0))

        lr = np.log1p(rc)
        la = np.log1p(ac)
        li = np.log1p(ic)
        lp = np.log1p(pc)

        delta = b
        delta += wr * sr + wa * sa + wi * si + wp * sp
        delta += ur * (sr * lr) + ua * (sa * la) + ui * (si * li) + up * (sp * lp)

        if delta < 0:
            delta = 0.0

        return float(delta), sims

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

        s_values = list(sims.values()) if sims else []
        if s_values:
            ssum = sum(s_values)
            smax = max(s_values)
            if (ssum < self._sim_gate_sum_min) or (smax < self._sim_gate_max_min):
                return probabilities, ""

        effective_delta = float(delta_logit) * self._smoothing

        updated: list[float] = []
        boosts: list[float] = []

        s_values = list(sims.values()) if sims else []
        if s_values:
            q_raw = 0.7 * max(s_values) + 0.3 * (sum(s_values) / max(1, len(s_values)))
        else:
            q_raw = 0.0
        q_adj = (q_raw ** max(1.0, self._cap_quality_gamma)) if q_raw > 0 else 0.0
        cap_factor = min(1.0, max(self._cap_min_factor, q_adj))
        for p in probabilities:
            p0 = _safe_float(p, 0.0)
            if 0.1 <= p0 <= 0.9 and effective_delta > 0:
                new_p = _sigmoid(_logit(p0) + effective_delta)
                scale = max(0.0, 1.0 - 2.0 * abs(p0 - 0.5))
                cap_boost = (self._max_total_boost * cap_factor) * scale
                cap = p0 * (1.0 + cap_boost)
                new_p = min(new_p, cap)
                new_p = max(0.0, min(1.0, new_p))
                updated.append(new_p)
                if p0 > 0:
                    boosts.append((new_p / p0) - 1.0)
            else:
                updated.append(p0)

        summary = ""
        if delta_logit > 0 and sims:
            parts: list[str] = []
            name_map = {
                "research_details": "科研项目",
                "award_details": "获奖情况",
                "internship_details": "实习经历",
                "paper_details": "论文发表",
            }
            for k in (
                "research_details",
                "award_details",
                "internship_details",
                "paper_details",
            ):
                s = sims.get(k, 0.0)
                if s > 0.15:
                    parts.append(f"{name_map[k]}: {s:.2f}")
            avg_boost = max(0.0, float(np.mean(boosts))) if boosts else 0.0
            summary = f"+{avg_boost:.1%} ({', '.join(parts)})" if avg_boost > 0 else ""

        return updated, summary
