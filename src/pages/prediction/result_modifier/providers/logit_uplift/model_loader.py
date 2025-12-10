from __future__ import annotations

import json
import threading
from typing import Any

import joblib
import numpy as np

from src.pages.prediction.result_modifier.providers.logit_uplift.utils import safe_float
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class ModelLoader:
    def __init__(
        self,
        vectorizer_path: str,
        centroids_path: str,
        weights_path: str,
    ) -> None:
        self._vectorizer_path = vectorizer_path
        self._centroids_path = centroids_path
        self._weights_path = weights_path

        self._vectorizer: Any | None = None
        self._centroids: dict[str, np.ndarray] | None = None
        self._weights: dict[str, float] | None = None
        self._weights_array: np.ndarray | None = None
        self._load_lock = threading.Lock()

    def lazy_load(self) -> None:
        if (
            self._vectorizer is not None
            and self._centroids is not None
            and self._weights_array is not None
        ):
            return

        with self._load_lock:
            if (
                self._vectorizer is not None
                and self._centroids is not None
                and self._weights_array is not None
            ):
                return

            try:
                if self._vectorizer is None:
                    self._vectorizer = joblib.load(self._vectorizer_path)
                    logger.debug(f"加载向量器: {self._vectorizer_path}")

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
                    logger.debug(f"加载质心: {self._centroids_path}")

                if self._weights_array is None:
                    with open(self._weights_path, encoding="utf-8") as f:
                        self._weights = json.load(f) or {}
                    self._weights_array = np.array(
                        [
                            safe_float(self._weights.get("b", 0.0)),
                            safe_float(self._weights.get("w_r", 0.0)),
                            safe_float(self._weights.get("w_a", 0.0)),
                            safe_float(self._weights.get("w_i", 0.0)),
                            safe_float(self._weights.get("w_p", 0.0)),
                            safe_float(self._weights.get("u_r", 0.0)),
                            safe_float(self._weights.get("u_a", 0.0)),
                            safe_float(self._weights.get("u_i", 0.0)),
                            safe_float(self._weights.get("u_p", 0.0)),
                        ],
                        dtype=np.float64,
                    )
                    logger.debug(f"加载权重: {self._weights_path}")
            except Exception as e:
                logger.error(f"延迟加载模型文件失败: {str(e)}", exc_info=True)
                raise

    @property
    def vectorizer(self) -> Any:
        self.lazy_load()
        assert self._vectorizer is not None
        return self._vectorizer

    @property
    def centroids(self) -> dict[str, np.ndarray]:
        self.lazy_load()
        assert self._centroids is not None
        return self._centroids

    @property
    def weights_array(self) -> np.ndarray:
        self.lazy_load()
        assert self._weights_array is not None
        return self._weights_array
