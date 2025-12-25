from __future__ import annotations

import json
import threading
from typing import Any

import joblib
import numpy as np

from src.pages.prediction.result_modifier.providers.logit_uplift.utils import safe_float
from src.pages.prediction.result_modifier.streamlit_cache import cache_resource
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class ModelLoader:
    """
    模型资源加载器。

    该类负责加载加成模型运行所需的重型资源（TF-IDF 矩阵、聚类中心、权重系数）。

    设计要点：
    1. **延迟加载 (Lazy Loading)**: 仅在第一次需要使用模型时才从磁盘读取，缩短应用启动时间。
    2. **线程安全**: 使用 `threading.Lock` 确保在并发环境下（如 Streamlit 多会话）资源只被初始化一次。
    3. **性能优化**:
       - 使用 `joblib` 加载向量器。
       - 使用 `mmap_mode="r"` 读取大数组，节省内存。
       - 对质心进行预归一化（L2 Normalize），加速后续相似度计算。
       - 使用 Python 原生 `tuple` 存储小型权重数组，减少 numpy 调用的开销。
    """

    # 使用 __slots__ 严格限制实例属性，微量提升内存效率和访问速度
    __slots__ = (
        "_vectorizer_path",
        "_centroids_path",
        "_weights_path",
        "_vectorizer",
        "_centroids",
        "_weights",
        "_weights_array",
        "_load_lock",
    )

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
        self._weights_array: tuple[float, ...] | None = None
        self._load_lock = threading.Lock()

    def lazy_load(self) -> None:
        """
        线程安全地执行模型懒惰加载。
        """
        if (
            self._vectorizer is not None
            and self._centroids is not None
            and self._weights_array is not None
        ):
            return

        with self._load_lock:
            # 双重检查锁
            if (
                self._vectorizer is not None
                and self._centroids is not None
                and self._weights_array is not None
            ):
                return

            try:
                if self._vectorizer is None:
                    # 加载 Scikit-Learn 的 TfidfVectorizer
                    self._vectorizer = joblib.load(self._vectorizer_path, mmap_mode="r")
                    logger.debug(f"加载向量器: {self._vectorizer_path}")

                if self._centroids is None:
                    # 加载预计算的高质量案例质心
                    data = np.load(self._centroids_path, mmap_mode="r")
                    centroids = {k: data[k] for k in data.files}
                    normed: dict[str, np.ndarray] = {}
                    for k, arr in centroids.items():
                        v = np.asarray(arr, dtype=np.float32)
                        n = np.linalg.norm(v)
                        # 预归一化，使得点积运算直接等同于余弦相似度
                        if n > 0:
                            v = v / n
                        normed[k] = v
                    self._centroids = normed
                    logger.debug(f"加载质心: {self._centroids_path}")

                if self._weights_array is None:
                    # 加载线性回归权重 (Logit Uplift Regression)
                    with open(self._weights_path, encoding="utf-8") as f:
                        self._weights = json.load(f) or {}

                    # 权重格式解析：b(bias), w(linear weights), u(interaction weights)
                    self._weights_array = (
                        safe_float(self._weights.get("b", 0.0)),
                        safe_float(self._weights.get("w_r", 0.0)),
                        safe_float(self._weights.get("w_a", 0.0)),
                        safe_float(self._weights.get("w_i", 0.0)),
                        safe_float(self._weights.get("w_p", 0.0)),
                        safe_float(self._weights.get("u_r", 0.0)),
                        safe_float(self._weights.get("u_a", 0.0)),
                        safe_float(self._weights.get("u_i", 0.0)),
                        safe_float(self._weights.get("u_p", 0.0)),
                    )
                    logger.debug(f"加载权重: {self._weights_path}")
            except (
                FileNotFoundError,
                OSError,
                EOFError,
                ValueError,
                TypeError,
                ImportError,
                AttributeError,
                json.JSONDecodeError,
            ) as e:
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
    def weights_array(self) -> tuple[float, ...]:
        self.lazy_load()
        assert self._weights_array is not None
        return self._weights_array


@cache_resource(show_spinner=False)
def get_model_loader(vectorizer_path: str, centroids_path: str, weights_path: str) -> ModelLoader:
    return ModelLoader(
        vectorizer_path=vectorizer_path,
        centroids_path=centroids_path,
        weights_path=weights_path,
    )
