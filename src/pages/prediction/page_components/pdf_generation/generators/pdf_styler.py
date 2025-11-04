import threading
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from src.pages.prediction.page_components.pdf_generation.config import pdf_config

_font_registered = False
_cached_font_name = None
_font_lock = threading.Lock()


class PDFStyler:
    def __init__(self):
        self.chinese_font_name = self._get_or_register_chinese_fonts()
        self.styles = self._setup_custom_styles()

    def _get_or_register_chinese_fonts(self):
        global _font_registered, _cached_font_name

        with _font_lock:
            if _font_registered and _cached_font_name:
                return _cached_font_name

        return self._register_chinese_fonts()

    def _register_chinese_fonts(self):
        global _font_registered, _cached_font_name

        with _font_lock:
            if _font_registered and _cached_font_name:
                return _cached_font_name

            try:
                font_paths = pdf_config.get_font_paths_for_current_os()

                for font_path in font_paths:
                    font_path_obj = Path(font_path)

                    if font_path_obj.exists():
                        try:
                            if font_path.endswith(".ttc"):
                                for subfont_index in [0, 1, 2, 3]:
                                    try:
                                        pdfmetrics.registerFont(
                                            TTFont(
                                                "ChineseFont", font_path, subfontIndex=subfont_index
                                            )
                                        )
                                        _font_registered = True
                                        _cached_font_name = "ChineseFont"
                                        return "ChineseFont"
                                    except Exception as sub_e:
                                        continue
                            else:
                                pdfmetrics.registerFont(TTFont("ChineseFont", font_path))
                                _font_registered = True
                                _cached_font_name = "ChineseFont"
                                return "ChineseFont"
                        except Exception as e:
                            continue
                    else:
                        continue

                _font_registered = True
                _cached_font_name = pdf_config.fonts.fallback_font
                return pdf_config.fonts.fallback_font

            except Exception as e:
                _font_registered = True
                _cached_font_name = pdf_config.fonts.fallback_font
                return pdf_config.fonts.fallback_font

    def _setup_custom_styles(self):
        styles = getSampleStyleSheet()

        styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=styles["Title"],
                fontSize=pdf_config.style.title_font_size,
                textColor=colors.HexColor(pdf_config.style.theme_primary),
                spaceAfter=20,
                alignment=TA_CENTER,
                fontName=self.chinese_font_name,
            )
        )

        styles.add(
            ParagraphStyle(
                name="SectionTitle",
                parent=styles["Heading1"],
                fontSize=pdf_config.style.section_font_size,
                textColor=colors.HexColor(pdf_config.style.theme_secondary),
                spaceBefore=18,
                spaceAfter=12,
                fontName=self.chinese_font_name,
            )
        )

        styles.add(
            ParagraphStyle(
                name="SubTitle",
                parent=styles["Heading2"],
                fontSize=pdf_config.style.subtitle_font_size,
                textColor=colors.HexColor(pdf_config.style.text_primary),
                spaceBefore=12,
                spaceAfter=6,
                fontName=self.chinese_font_name,
            )
        )

        styles.add(
            ParagraphStyle(
                name="CustomBody",
                parent=styles["Normal"],
                fontSize=pdf_config.style.body_font_size,
                textColor=colors.HexColor(pdf_config.style.text_primary),
                spaceAfter=6,
                alignment=TA_JUSTIFY,
                fontName=self.chinese_font_name,
                leading=14,
            )
        )

        styles.add(
            ParagraphStyle(
                name="SmallText",
                parent=styles["Normal"],
                fontSize=pdf_config.style.small_text_font_size,
                textColor=colors.HexColor(pdf_config.style.text_secondary),
                spaceAfter=4,
                alignment=TA_JUSTIFY,
                fontName=self.chinese_font_name,
                leading=12,
            )
        )

        styles.add(
            ParagraphStyle(
                name="Highlight",
                parent=styles["Normal"],
                fontSize=pdf_config.style.subtitle_font_size,
                textColor=colors.HexColor(pdf_config.style.theme_accent),
                spaceBefore=8,
                spaceAfter=8,
                fontName=self.chinese_font_name,
            )
        )

        styles.add(
            ParagraphStyle(name="LeftSmallText", parent=styles["SmallText"], alignment=TA_LEFT)
        )

        return styles

    def get_table_style_with_font(self, base_style_commands: list = None) -> list:
        if base_style_commands is None:
            base_style_commands = []

        font_commands = [
            ("FONTNAME", (0, 0), (-1, -1), self.chinese_font_name),
            ("FONTSIZE", (0, 0), (-1, -1), pdf_config.style.body_font_size),
        ]

        return font_commands + base_style_commands
