"""AI 选校报告 — newspaper layout, streaming, product-aware rendering."""

from .renderer import build_matched_products, render_ai_section, render_static_frame
from .school_stats import SchoolFeatureStats
from .sections import render_ai_section_streaming

__all__ = [
    "build_matched_products",
    "render_ai_section",
    "render_static_frame",
    "render_ai_section_streaming",
    "SchoolFeatureStats",
]
