from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class PDFConfig:
    theme_primary: str = "#0D47A1"
    theme_background: str = "#F5F5F5"
    border_color: str = "#E0E0E0"

    high_prob_color: str = "#4CAF50"
    medium_prob_color: str = "#FFC107"
    low_prob_color: str = "#F44336"

    watermark_opacity: float = 0.08
    watermark_font_size: int = 10
    watermark_rotation: int = 35
    watermark_x_spacing: int = 200
    watermark_y_spacing: int = 150

    top_margin: float = 2.0
    bottom_margin: float = 2.0
    left_margin: float = 1.8
    right_margin: float = 1.8

    strategy_descriptions: Dict[str, str] = field(
        default_factory=lambda: {
            "申请策略1": "<b>普通策略</b>：根据学生背景和目标院校，推荐适合的申请组合。",
            "申请策略2": "<b>联申策略</b>：联申多所学校，增加录取机会。",
            "申请策略3": "<b>高端策略</b>：冲刺名校，最大化申请机会。",
        }
    )

    high_prob_threshold: float = 0.7
    medium_prob_threshold: float = 0.4
    max_schools_per_strategy: int = 10

    school_logos_path: Path = Path("assets/school_logos")
    product_logo_path: Path = Path("assets/product_logo.png")

    def get_school_logo_path(self, university_name: str) -> Optional[Path]:
        for ext in [".png", ".jpg", ".jpeg"]:
            logo_path = self.school_logos_path / f"{university_name}{ext}"
            if logo_path.exists():
                return logo_path
        return None

    def get_product_logo_path(self) -> Optional[Path]:
        return self.product_logo_path if self.product_logo_path.exists() else None

    def get_probability_color(self, probability: float) -> str:
        if probability >= self.high_prob_threshold:
            return self.high_prob_color
        elif probability >= self.medium_prob_threshold:
            return self.medium_prob_color
        return self.low_prob_color

    def get_strategy_description(self, strategy_name: str, school_count: int) -> str:
        base_desc = self.strategy_descriptions.get(
            strategy_name, f"推荐 {school_count} 所院校的申请组合方案。"
        )
        if "<b>" in base_desc and "</b>" in base_desc:
            base_desc = base_desc.replace("</b>：", f" ({school_count}所学校)</b>：")
        return base_desc


pdf_config = PDFConfig()
