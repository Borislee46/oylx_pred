import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PDFStyleConfig:
    theme_primary: str = "#0D47A1"
    theme_secondary: str = "#1976D2"
    theme_accent: str = "#42A5F5"
    theme_background: str = "#F5F5F5"
    text_primary: str = "#212121"
    text_secondary: str = "#757575"
    border_color: str = "#E0E0E0"

    high_prob_color: str = "#4CAF50"
    medium_prob_color: str = "#FFC107"
    low_prob_color: str = "#F44336"

    title_font_size: int = 22
    section_font_size: int = 15
    subtitle_font_size: int = 12
    body_font_size: int = 9
    small_text_font_size: int = 8

    watermark_opacity: float = 0.08
    watermark_font_size: int = 10
    watermark_rotation: int = 35
    watermark_x_spacing: int = 200
    watermark_y_spacing: int = 150


@dataclass
class PDFLayoutConfig:
    top_margin: float = 2.0
    bottom_margin: float = 2.0
    left_margin: float = 1.8
    right_margin: float = 1.8

    school_logo_width: float = 0.8
    school_logo_height: float = 0.8
    product_logo_width: float = 1.2
    product_logo_height: float = 1.2

    background_table_widths: List[float] = field(default_factory=lambda: [4.0, 13.5])
    soft_skills_table_widths: List[float] = field(default_factory=lambda: [4.0, 4.0])


@dataclass
class FontPathConfig:
    windows_fonts: List[str] = field(
        default_factory=lambda: [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simkai.ttf",
        ]
    )

    macos_fonts: List[str] = field(
        default_factory=lambda: [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode MS.ttf",
        ]
    )

    linux_fonts: List[str] = field(
        default_factory=lambda: [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/chinese/SourceHanSansCN-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )

    fallback_font: str = "Helvetica"


@dataclass
class PDFContentConfig:
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
    max_displayed_results: int = 20
    max_application_tips: int = 3
    max_detailed_info_lines: int = 8


class PDFConfigManager:
    def __init__(self):
        self.style = PDFStyleConfig()
        self.layout = PDFLayoutConfig()
        self.fonts = FontPathConfig()
        self.content = PDFContentConfig()
        self._asset_paths = self._initialize_asset_paths()

    def _initialize_asset_paths(self) -> Dict[str, Path]:
        return {
            "school_logos": Path("assets/school_logos"),
            "product_logo": Path("assets/product_logo.png"),
            "root_assets": Path("assets"),
        }

    def get_font_paths_for_current_os(self) -> List[str]:
        system = platform.system()
        if system == "Windows":
            return self.fonts.windows_fonts
        elif system == "Darwin":
            return self.fonts.macos_fonts
        else:
            return self.fonts.linux_fonts

    def get_school_logo_path(self, university_name: str) -> Optional[Path]:
        extensions = [".png", ".jpg", ".jpeg"]
        logo_base_path = self._asset_paths["school_logos"]

        for ext in extensions:
            logo_path = logo_base_path / f"{university_name}{ext}"
            if logo_path.exists():
                return logo_path

        return None

    def get_product_logo_path(self) -> Optional[Path]:
        logo_path = self._asset_paths["product_logo"]
        return logo_path if logo_path.exists() else None

    def get_probability_color(self, probability: float) -> str:
        if probability >= self.content.high_prob_threshold:
            return self.style.high_prob_color
        elif probability >= self.content.medium_prob_threshold:
            return self.style.medium_prob_color
        else:
            return self.style.low_prob_color

    def get_strategy_description(self, strategy_name: str, school_count: int) -> str:
        base_desc = self.content.strategy_descriptions.get(
            strategy_name, f"推荐 {school_count} 所院校的申请组合方案。"
        )
        if "<b>" in base_desc and "</b>" in base_desc:
            base_desc = base_desc.replace("</b>：", f" ({school_count}所学校)</b>：")
        return base_desc


pdf_config = PDFConfigManager()
