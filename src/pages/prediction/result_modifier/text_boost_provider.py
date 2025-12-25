from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from src.pages.prediction.result_modifier.utils import has_any_experience
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class TextBoostProvider:
    """
    文本加成提供者基类。

    该接口定义了如何根据用户的背景描述文本（如科研、实习等）对原始录取概率进行修正。
    实现类应根据文本内容的"含金量"或"匹配度"返回调整后的概率列表。
    """

    def apply(self, probabilities: list[float], experience_details: dict[str, str]) -> list[float]:
        """
        应用文本加成逻辑。

        Args:
            probabilities: 原始模型预测的各学校录取概率列表。
            experience_details: 包含背景描述文本和数量的字典。

        Returns:
            list[float]: 调整后的概率列表。
        """
        raise NotImplementedError


class NullTextBoostProvider(TextBoostProvider):
    """
    空实现提供者，不对概率做任何修改。
    用于配置禁用或模型加载失败的回退场景。
    """

    def apply(self, probabilities: list[float], experience_details: dict[str, str]) -> list[float]:
        return probabilities


class GatedTextBoostProvider(TextBoostProvider):
    """
    带门控的加成提供者包装器。

    逻辑：
    只有当用户填写了至少一项背景经历（如科研、实习等）时，才会调用内部的提供者。
    这可以避免在没有输入文本时进行昂贵的计算或产生无效的加成。
    """

    def __init__(self, inner: TextBoostProvider):
        self._inner = inner

    def apply(self, probabilities: list[float], experience_details: dict[str, str]) -> list[float]:
        if not has_any_experience(experience_details):
            return probabilities
        return self._inner.apply(probabilities, experience_details)


def get_text_boost_provider(config: dict[str, Any] | None) -> TextBoostProvider:
    """
    根据配置动态创建 TextBoostProvider 实例。

    设计模式：工厂模式 + 缓存（lru_cache）。

    Args:
        config: 提供者配置字典，通常来自 app_config.json 或数据库。
               包含 enabled 状态、模型路径、超参数等。

    Returns:
        TextBoostProvider: 具体的提供者实例，默认返回 NullTextBoostProvider。
    """
    if not config or not config.get("enabled"):
        return NullTextBoostProvider()

    try:
        # 将配置字典序列化为字符串，以便作为缓存的 key
        key = json.dumps(config or {}, ensure_ascii=False, sort_keys=True)
        return _get_text_boost_provider_cached(key)
    except (TypeError, ValueError) as e:
        logger.warning(f"序列化配置失败，使用空提供者: {str(e)}")
        return NullTextBoostProvider()
    except (OSError, RuntimeError, ImportError, AttributeError) as e:
        logger.error(f"获取文本加成提供者失败: {str(e)}", exc_info=True)
        return NullTextBoostProvider()


@lru_cache(maxsize=16)
def _get_text_boost_provider_cached(config_key: str) -> TextBoostProvider:
    """
    带有缓存的工厂方法，避免频繁解析配置和重新初始化昂贵的模型资源（如 TF-IDF 向量器）。
    """
    try:
        config = json.loads(config_key)
        from src.pages.prediction.result_modifier.providers.logit_uplift_provider import (
            LogitUpliftProvider,
        )

        # 提取模型路径，这是 LogitUpliftProvider 运行的核心
        model_paths = (config or {}).get("model_paths", {})
        vec_path = model_paths.get("tfidf_vectorizer")
        cen_path = model_paths.get("tfidf_centroids")
        w_path = model_paths.get("text_uplift_weights")

        if not vec_path or not cen_path or not w_path:
            logger.warning("文本加成模型路径配置不完整，使用空提供者")
            return NullTextBoostProvider()

        # 提取控制加成强度的数学超参数
        max_total_boost = config.get("max_total_boost", 0.05)
        sim_gate_sum_min = config.get("sim_gate_sum_min")
        sim_gate_max_min = config.get("sim_gate_max_min")
        smoothing = config.get("smoothing")
        cap_min_factor = config.get("cap_min_factor")
        cap_quality_gamma = config.get("cap_quality_gamma")
        high_signal = config.get("high_signal")

        # 实例化 Logit 提升提供者
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
            high_signal=high_signal if isinstance(high_signal, dict) else None,
        )
        logger.info("成功创建LogitUpliftProvider实例")

        # 使用 GatedTextBoostProvider 封装，确保只在有内容时运行
        return GatedTextBoostProvider(provider)
    except json.JSONDecodeError as e:
        logger.error(f"解析配置JSON失败: {str(e)}")
        return NullTextBoostProvider()
    except (ImportError, AttributeError) as e:
        logger.error(f"导入或创建LogitUpliftProvider失败: {str(e)}")
        return NullTextBoostProvider()
    except (TypeError, ValueError, OSError, RuntimeError) as e:
        logger.error(f"创建文本加成提供者失败: {str(e)}", exc_info=True)
        return NullTextBoostProvider()
