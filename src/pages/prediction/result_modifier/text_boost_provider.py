from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from src.pages.prediction.result_modifier.utils import has_valid_experience_details
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class TextBoostProvider:
    """文本加成提供者接口"""

    def apply(
        self, probabilities: list[float], experience_details: dict[str, str]
    ) -> tuple[list[float], str]:
        """
        应用文本加成到概率列表

        Args:
            probabilities: 原始概率列表
            experience_details: 经验详情字典

        Returns:
            (调整后的概率列表, 加成说明字符串)
        """
        raise NotImplementedError


class NullTextBoostProvider(TextBoostProvider):
    """空文本加成提供者（不进行任何调整）"""

    def apply(
        self, probabilities: list[float], experience_details: dict[str, str]
    ) -> tuple[list[float], str]:
        return probabilities, ""


class GatedTextBoostProvider(TextBoostProvider):
    """门控文本加成提供者（仅在经验详情有效时应用）"""

    def __init__(self, inner: TextBoostProvider):
        """
        初始化门控提供者

        Args:
            inner: 内部文本加成提供者
        """
        self._inner = inner

    def apply(
        self, probabilities: list[float], experience_details: dict[str, str]
    ) -> tuple[list[float], str]:
        if not has_valid_experience_details(experience_details):
            return probabilities, ""
        return self._inner.apply(probabilities, experience_details)


def get_text_boost_provider(config: dict[str, Any] | None) -> TextBoostProvider:
    """
    获取文本加成提供者实例

    Args:
        config: 配置字典

    Returns:
        文本加成提供者实例
    """
    if not config or not config.get("enabled"):
        return NullTextBoostProvider()

    try:
        key = json.dumps(config or {}, ensure_ascii=False, sort_keys=True)
        return _get_text_boost_provider_cached(key)
    except (TypeError, ValueError) as e:
        logger.warning(f"序列化配置失败，使用空提供者: {str(e)}")
        return NullTextBoostProvider()
    except Exception as e:
        logger.error(f"获取文本加成提供者时发生未知错误: {str(e)}", exc_info=True)
        return NullTextBoostProvider()


@lru_cache(maxsize=16)
def _get_text_boost_provider_cached(config_key: str) -> TextBoostProvider:
    """
    从缓存获取文本加成提供者（内部函数）

    Args:
        config_key: 配置的JSON字符串键

    Returns:
        文本加成提供者实例
    """
    try:
        config = json.loads(config_key)
        from src.pages.prediction.result_modifier.providers.logit_uplift_provider import (
            LogitUpliftProvider,
        )

        model_paths = (config or {}).get("model_paths", {})
        vec_path = model_paths.get("tfidf_vectorizer")
        cen_path = model_paths.get("tfidf_centroids")
        w_path = model_paths.get("text_uplift_weights")
        if not vec_path or not cen_path or not w_path:
            logger.warning("文本加成模型路径配置不完整，使用空提供者")
            return NullTextBoostProvider()

        max_total_boost = config.get("max_total_boost", 0.05)
        sim_gate_sum_min = config.get("sim_gate_sum_min")
        sim_gate_max_min = config.get("sim_gate_max_min")
        smoothing = config.get("smoothing")
        cap_min_factor = config.get("cap_min_factor")
        cap_quality_gamma = config.get("cap_quality_gamma")

        provider = LogitUpliftProvider(
            vectorizer_path=vec_path,
            centroids_path=cen_path,
            weights_path=w_path,
            max_total_boost=max_total_boost,
            sim_gate_sum_min=sim_gate_sum_min,
            sim_gate_max_min=sim_gate_max_min,
            smoothing=smoothing,
            cap_min_factor=cap_min_factor,
            cap_quality_gamma=cap_quality_gamma,
        )
        logger.info("成功创建LogitUpliftProvider实例")
        return GatedTextBoostProvider(provider)
    except json.JSONDecodeError as e:
        logger.error(f"解析配置JSON失败: {str(e)}")
        return NullTextBoostProvider()
    except (ImportError, AttributeError) as e:
        logger.error(f"导入或创建LogitUpliftProvider失败: {str(e)}")
        return NullTextBoostProvider()
    except Exception as e:
        logger.error(f"创建文本加成提供者时发生未知错误: {str(e)}", exc_info=True)
        return NullTextBoostProvider()
