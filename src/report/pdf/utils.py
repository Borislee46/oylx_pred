from typing import Any

from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce

logger = setup_logger("page3", "prediction")


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
        if isinstance(prob, (int, float)):
            return clip_probability_coerce(prob)
        if isinstance(prob, str):
            return clip_probability_coerce(prob, default=0.0)
        return 0.0

    @staticmethod
    def get_field_value(data: dict, field_names: list[str], default: Any = None) -> Any:
        for field_name in field_names:
            if field_name in data and data[field_name] is not None:
                return data[field_name]
        return default


def _coerce_language_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and (not value.strip() or value == "未填写"):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def resolve_language_score(lang_raw: Any, lang_norm: Any) -> float | None:
    raw = _coerce_language_score(lang_raw)
    if raw is not None:
        return raw
    norm = _coerce_language_score(lang_norm)
    if norm is not None and norm > 1:
        return norm
    return None


class SchoolDataProcessor:
    def __init__(self):
        self.normalizer = DataNormalizer()

    def extract_school_info(self, school_data: dict) -> tuple[str, str, float]:
        university = self.normalizer.get_field_value(
            school_data, ["university", "目标院校", "target_university", "school"], "未知大学"
        )

        major = self.normalizer.get_field_value(
            school_data,
            ["major", "major_cn", "目标专业", "target_major", "原始专业名称"],
            "未知专业",
        )

        probability = self.normalizer.get_field_value(
            school_data, ["probability", "调整后概率", "predicted_probability"], 0.0
        )

        return (
            self.normalizer.normalize_university_name(university),
            self.normalizer.normalize_major_name(major),
            self.normalizer.normalize_probability(probability),
        )

    def group_schools_by_university(self, schools: list[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {}
        for school in schools:
            university, _, _ = self.extract_school_info(school)
            if university not in grouped:
                grouped[university] = []
            grouped[university].append(school)
        return grouped


def create_responsive_image(image_path: str, max_width: float, max_height: float):
    from reportlab.platypus import Image

    img = Image(image_path)
    img_width, img_height = img.imageWidth, img.imageHeight
    if img_width <= 0 or img_height <= 0:
        return img
    width_ratio = max_width / img_width
    height_ratio = max_height / img_height
    scale_ratio = min(width_ratio, height_ratio)
    img.drawWidth = img_width * scale_ratio
    img.drawHeight = img_height * scale_ratio
    return img


_trimmed_logo_cache: dict[str, Any] = {}


def _trim_logo_buffer(image_path: str):
    import io
    import os

    try:
        mtime = os.path.getmtime(image_path)
    except OSError:
        return None
    cache_key = f"{image_path}:{mtime}"
    if cache_key in _trimmed_logo_cache:
        buf = _trimmed_logo_cache[cache_key]
        buf.seek(0)
        return buf

    from PIL import Image as PILImage
    from PIL import ImageChops

    with PILImage.open(image_path) as im:
        if im.mode == "RGBA":
            white_bg = PILImage.new("RGBA", im.size, (255, 255, 255, 255))
            try:
                im = PILImage.alpha_composite(white_bg, im).convert("RGB")
            finally:
                white_bg.close()
        elif im.mode != "RGB":
            im = im.convert("RGB")
        bg = PILImage.new("RGB", im.size, (255, 255, 255))
        try:
            bbox = ImageChops.difference(im, bg).getbbox()
        finally:
            bg.close()
        if bbox:
            w, h = im.size
            pad = int(min(w, h) * 0.04)
            im = im.crop(
                (
                    max(0, bbox[0] - pad),
                    max(0, bbox[1] - pad),
                    min(w, bbox[2] + pad),
                    min(h, bbox[3] + pad),
                )
            )
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    buf.seek(0)
    _trimmed_logo_cache[cache_key] = buf
    return buf


def create_trimmed_logo_image(image_path: str, max_width: float, max_height: float):
    from reportlab.platypus import Image

    buf = None
    try:
        buf = _trim_logo_buffer(image_path)
    except Exception as e:
        logger.warning(f"logo 裁剪失败，回退原图: {e}")

    img = Image(buf) if buf is not None else Image(image_path)
    img_width, img_height = img.imageWidth, img.imageHeight
    if img_width <= 0 or img_height <= 0:
        return img
    scale_ratio = min(max_width / img_width, max_height / img_height)
    img.drawWidth = img_width * scale_ratio
    img.drawHeight = img_height * scale_ratio
    return img
