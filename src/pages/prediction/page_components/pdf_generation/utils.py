import functools
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.pages.prediction.page_components.pdf_generation.config import pdf_config
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class PDFCacheManager:
    def __init__(self):
        self._cache = {}
        self._cache_stats = {"hits": 0, "misses": 0}

    def get_cache_key(self, *args, **kwargs) -> str:
        key_parts = []
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif hasattr(arg, "__dict__"):
                key_parts.append(str(hash(str(arg.__dict__))))
            else:
                key_parts.append(str(hash(str(arg))))

        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        return "|".join(key_parts)

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._cache_stats["hits"] += 1
            return self._cache[key]
        self._cache_stats["misses"] += 1
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        self._cache[key] = {"value": value, "expires": time.time() + ttl}

    def clear_expired(self) -> None:
        current_time = time.time()
        expired_keys = [key for key, data in self._cache.items() if data["expires"] < current_time]
        for key in expired_keys:
            del self._cache[key]

    def get_stats(self) -> Dict:
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total if total > 0 else 0
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
        }


pdf_cache = PDFCacheManager()


def cached_result(ttl: int = 3600):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = pdf_cache.get_cache_key(func.__name__, *args, **kwargs)
            cached_data = pdf_cache.get(cache_key)

            if cached_data:
                return cached_data["value"]

            result = func(*args, **kwargs)
            pdf_cache.set(cache_key, result, ttl)
            return result

        return wrapper

    return decorator


class DataNormalizer:
    @staticmethod
    def normalize_university_name(university: str) -> str:
        if not university or not isinstance(university, str):
            return "未知大学"
        return university.strip()

    @staticmethod
    def normalize_major_name(major: str) -> str:
        if not major or not isinstance(major, str):
            return "未知专业"
        return major.strip()

    @staticmethod
    def normalize_probability(prob: Any) -> float:
        try:
            if isinstance(prob, (int, float)):
                return max(0.0, min(1.0, float(prob)))
            elif isinstance(prob, str):
                return max(0.0, min(1.0, float(prob)))
            else:
                return 0.0
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def get_field_value(data: Dict, field_names: List[str], default: Any = None) -> Any:
        for field_name in field_names:
            if field_name in data and data[field_name] is not None:
                return data[field_name]
        return default


class ContentFormatter:
    @staticmethod
    def get_probability_display_text(probability: float, with_color: bool = True) -> str:
        prob_text = f"{probability:.1%}"
        if with_color:
            color = pdf_config.get_probability_color(probability)
            return f'<font color="{color}"><b>录取概率: {prob_text}</b></font>'
        return f"录取概率: {prob_text}"


class SchoolDataProcessor:
    def __init__(self):
        self.normalizer = DataNormalizer()
        self.formatter = ContentFormatter()

    def extract_school_info(self, school_data: Dict) -> Tuple[str, str, float]:
        university = self.normalizer.get_field_value(
            school_data, ["university", "目标院校", "target_university", "school"], "未知大学"
        )

        major = self.normalizer.get_field_value(
            school_data, ["major", "目标专业", "target_major", "原始专业名称"], "未知专业"
        )

        probability = self.normalizer.get_field_value(
            school_data, ["probability", "调整后概率", "predicted_probability"], 0.0
        )

        return (
            self.normalizer.normalize_university_name(university),
            self.normalizer.normalize_major_name(major),
            self.normalizer.normalize_probability(probability),
        )

    def group_schools_by_university(self, schools: List[Dict]) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for school in schools:
            university, _, _ = self.extract_school_info(school)
            if university not in grouped:
                grouped[university] = []
            grouped[university].append(school)
        return grouped

    def filter_and_limit_schools(self, schools: List[Dict], max_count: int = None) -> List[Dict]:
        if not schools:
            return []

        max_count = max_count or pdf_config.max_schools_per_strategy

        sorted_schools = sorted(
            schools,
            key=lambda x: self.normalizer.normalize_probability(
                self.normalizer.get_field_value(x, ["probability", "调整后概率"], 0)
            ),
            reverse=True,
        )

        return sorted_schools[:max_count]


def create_responsive_image(image_path: str, max_width: float, max_height: float):
    from reportlab.platypus import Image

    try:
        if not image_path or not Path(image_path).exists():
            return None

        img = Image(image_path)
        img_width, img_height = img.imageWidth, img.imageHeight

        if img_width == 0 or img_height == 0:
            return None

        width_ratio = max_width / img_width
        height_ratio = max_height / img_height
        scale_ratio = min(width_ratio, height_ratio)

        img.drawWidth = img_width * scale_ratio
        img.drawHeight = img_height * scale_ratio

        return img
    except Exception as e:
        logger.warning(f"创建响应式图片失败 {image_path}: {e}")
        return None
