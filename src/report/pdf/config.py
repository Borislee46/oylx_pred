from dataclasses import dataclass, field
from pathlib import Path


def _get_project_root() -> Path:
    start = Path(__file__).resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    return start.parents[3]


_PROJECT_ROOT = _get_project_root()


@dataclass
class PDFConfig:
    theme_primary: str = "#0F2A43"
    theme_secondary: str = "#2C4A63"
    accent_gold: str = "#B6862C"
    accent_gold_soft: str = "#EBD9A8"

    theme_background: str = "#F6F8FB"
    card_background: str = "#FFFFFF"
    panel_background: str = "#F2F5FA"
    border_color: str = "#E1E7EF"
    divider_color: str = "#D5DEE9"

    text_primary: str = "#1A2733"
    text_secondary: str = "#52606D"
    text_muted: str = "#8A97A5"

    high_prob_color: str = "#2E7D55"
    medium_prob_color: str = "#C0892A"
    low_prob_color: str = "#B0413E"

    watermark_opacity: float = 0.05
    watermark_font_size: int = 10
    watermark_rotation: int = 35
    watermark_x_spacing: int = 200
    watermark_y_spacing: int = 150

    top_margin: float = 2.1
    bottom_margin: float = 2.0
    left_margin: float = 1.9
    right_margin: float = 1.9

    strategy_descriptions: dict[str, str] = field(
        default_factory=lambda: {
            "申请策略1": "<b>普通策略</b>：根据学生背景和目标院校，推荐适合的申请组合。",
            "申请策略2": "<b>联申策略</b>：联申多所学校，增加录取机会。",
            "申请策略3": "<b>高端策略</b>：冲刺名校，最大化申请机会。",
        }
    )

    high_prob_threshold: float = 0.7
    medium_prob_threshold: float = 0.4
    max_schools_per_strategy: int = 10

    school_logos_path: Path = _PROJECT_ROOT / "assets" / "school_logos"
    product_logo_path: Path = _PROJECT_ROOT / "assets" / "product_logo.png"

    def get_school_logo_path(self, university_name: str) -> Path | None:
        from src.pages.prediction.school_logo_loader import (
            _CN_TO_KEY,
        )

        candidates = [_CN_TO_KEY.get(university_name, university_name), university_name]
        for name in candidates:
            for ext in (".png", ".jpg", ".jpeg"):
                logo_path = self.school_logos_path / f"{name}{ext}"
                if logo_path.exists():
                    return logo_path
        return None

    def get_product_logo_path(self) -> Path | None:
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
