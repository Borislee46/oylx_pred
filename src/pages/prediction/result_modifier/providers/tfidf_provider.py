from __future__ import annotations

import hashlib
import time

import joblib
import numpy as np

from src.pages.prediction.result_modifier.text_boost_provider import TextBoostProvider


def _sha1_text(d: dict[str, str]) -> str:
    text = "\n".join(
        [
            str(d.get(k, "") or "")
            for k in ("research_details", "award_details", "internship_details", "paper_details")
        ]
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class TfidfBoostProvider(TextBoostProvider):
    def __init__(
        self,
        vectorizer_path: str,
        centroids_path: str,
        max_total_boost: float = 0.10,
        timeout_ms: int = 100,
        cache: dict | None = None,
        similarity_thresholds: list[list[float]] | None = None,
    ):
        self._vectorizer_path = vectorizer_path
        self._centroids_path = centroids_path
        self._max_total_boost = max_total_boost
        self._timeout_ms = timeout_ms
        self._cache = cache if isinstance(cache, dict) else {}
        self._thresholds: list[tuple[float, float]] = []
        if similarity_thresholds and isinstance(similarity_thresholds, list):
            try:
                pairs = [(float(t), float(b)) for t, b in similarity_thresholds]
                self._thresholds = sorted(pairs, key=lambda x: x[0], reverse=True)
            except Exception:
                self._thresholds = []
        if not self._thresholds:
            self._thresholds = [(0.40, 0.05), (0.30, 0.03), (0.20, 0.02)]

        self._vectorizer = None
        self._centroids = None

    def _lazy_load(self):
        if self._vectorizer is None:
            self._vectorizer = joblib.load(self._vectorizer_path)
        if self._centroids is None:
            data = np.load(self._centroids_path, mmap_mode="r")
            centroids = {k: data[k] for k in data.files}
            normed = {}
            for k, arr in centroids.items():
                try:
                    v = np.asarray(arr, dtype=np.float32)
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                    normed[k] = v
                except Exception:
                    normed[k] = np.asarray(arr)
            self._centroids = normed

    @staticmethod
    def _prep_text(s: str) -> str:
        if not isinstance(s, str):
            return ""
        return s.strip()

    def _compute_similarity(self, details: dict[str, str]) -> dict[str, float]:
        self._lazy_load()
        assert self._vectorizer is not None
        assert self._centroids is not None

        texts = {
            k: self._prep_text(details.get(k, ""))
            for k in ["research_details", "award_details", "internship_details", "paper_details"]
        }
        sims: dict[str, float] = {k: 0.0 for k in texts}

        keys = list(texts.keys())
        corpus = [texts[k] for k in keys]
        X = self._vectorizer.transform(corpus)

        for idx, k in enumerate(keys):
            row = X.getrow(idx)
            if row is None:
                sims[k] = 0.0
                continue
            data = getattr(row, "data", None)
            if data is None or data.size == 0:
                sims[k] = 0.0
                continue
            nn = float(np.linalg.norm(data))
            if nn > 0:
                row = row.copy()
                row.data = row.data / nn
            centroid = self._centroids.get(k)
            if centroid is None or centroid.size == 0:
                sims[k] = 0.0
            else:
                dot_val = row.dot(centroid)
                try:
                    dot_scalar = float(np.asarray(dot_val).ravel()[0])
                except Exception:
                    dot_scalar = 0.0
                sims[k] = float(np.clip(dot_scalar, 0.0, 1.0))
        return sims

    def _has_strong_signal(self, details: dict[str, str]) -> bool:
        try:
            for exp_type in ["research", "award", "internship", "paper"]:
                from src.pages.prediction.result_modifier.keyword_config import KEYWORD_WEIGHTS

                cfg = KEYWORD_WEIGHTS.get(exp_type, {})
                top_keywords = cfg.get("top_tier", [])
                text = (details.get(f"{exp_type}_details", "") or "").lower()
                for kw in top_keywords:
                    if kw.lower() in text:
                        return True
        except Exception:
            return False
        return False

    def _map_sims_to_boost(self, sims: dict[str, float]) -> float:
        total = 0.0
        for s in sims.values():
            for th, add in self._thresholds:
                if s > th:
                    total += add
                    break
        return min(total, self._max_total_boost)

    def apply(
        self, probabilities: list[float], experience_details: dict[str, str]
    ) -> tuple[list[float], str]:
        if not probabilities or not isinstance(experience_details, dict):
            return probabilities, ""

        try:
            if not self._has_strong_signal(experience_details):
                return probabilities, ""
        except Exception:
            return probabilities, ""

        start = time.time()
        cache_key = _sha1_text(experience_details)
        cached = self._cache.get(cache_key)
        sims = {}
        if cached is not None:
            total_boost = cached
        else:
            sims = self._compute_similarity(experience_details)
            total_boost = self._map_sims_to_boost(sims)
            self._cache[cache_key] = total_boost

        elapsed_ms = (time.time() - start) * 1000
        if elapsed_ms > self._timeout_ms:
            return probabilities, f"计算超时({elapsed_ms:.1f}ms > {self._timeout_ms}ms)"

        boosted = []
        for p in probabilities:
            if 0.2 <= float(p) <= 0.8 and total_boost > 0:
                boosted.append(min(float(p) * (1.0 + total_boost), 1.0))
            else:
                boosted.append(float(p))

        parts = []
        if total_boost > 0:
            if sims:
                for name, s in sims.items():
                    if s > 0.15:
                        cn = {
                            "research_details": "科研项目",
                            "award_details": "获奖情况",
                            "internship_details": "实习经历",
                            "paper_details": "论文发表",
                        }[name]
                        parts.append(f"{cn}: {s:.2f}")
        summary = f"+{total_boost:.1%} ({', '.join(parts)})" if total_boost > 0 else ""
        return boosted, summary
