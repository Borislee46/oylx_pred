import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, Spacer, Table, TableStyle

from ..config import pdf_config

_CHINESE_FONT_NAME = "SimHei"

_FONT_CANDIDATES = [
    "assets/fonts/NotoSansSC-Regular.ttf",
    "assets/fonts/simhei.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]


def _find_cjk_font() -> str | None:
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


_font_path = _find_cjk_font()
if _font_path:
    pdfmetrics.registerFont(TTFont(_CHINESE_FONT_NAME, _font_path))
else:
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    _CHINESE_FONT_NAME = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(_CHINESE_FONT_NAME))


class PDFStyler:
    def __init__(self):
        self.chinese_font_name = _CHINESE_FONT_NAME
        self.cfg = pdf_config
        self.styles = self._setup_custom_styles()

    def _setup_custom_styles(self):
        styles = getSampleStyleSheet()
        font = self.chinese_font_name

        styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=styles["Title"],
                fontSize=25,
                leading=30,
                textColor=colors.HexColor(pdf_config.theme_primary),
                spaceAfter=8,
                alignment=TA_CENTER,
                fontName=font,
            )
        )
        styles.add(
            ParagraphStyle(
                name="Eyebrow",
                parent=styles["Normal"],
                fontSize=10.5,
                leading=14,
                textColor=colors.HexColor(pdf_config.accent_gold),
                alignment=TA_CENTER,
                spaceAfter=10,
                fontName=font,
            )
        )
        styles.add(
            ParagraphStyle(
                name="CoverSubtitle",
                parent=styles["Normal"],
                fontSize=13,
                leading=18,
                textColor=colors.HexColor(pdf_config.text_secondary),
                alignment=TA_CENTER,
                spaceAfter=6,
                fontName=font,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SectionTitle",
                parent=styles["Heading1"],
                fontSize=16,
                leading=20,
                textColor=colors.HexColor(pdf_config.theme_primary),
                spaceBefore=16,
                spaceAfter=4,
                fontName=font,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SubTitle",
                parent=styles["Heading2"],
                fontSize=12.5,
                leading=17,
                textColor=colors.HexColor(pdf_config.theme_secondary),
                spaceBefore=10,
                spaceAfter=5,
                fontName=font,
            )
        )
        styles.add(
            ParagraphStyle(
                name="CustomBody",
                parent=styles["Normal"],
                fontSize=10.5,
                textColor=colors.HexColor(pdf_config.text_primary),
                spaceAfter=5,
                alignment=TA_JUSTIFY,
                fontName=font,
                leading=15.5,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SmallText",
                parent=styles["Normal"],
                fontSize=9,
                textColor=colors.HexColor(pdf_config.text_muted),
                spaceAfter=3,
                alignment=TA_JUSTIFY,
                fontName=font,
                leading=12.5,
            )
        )
        styles.add(
            ParagraphStyle(
                name="Highlight",
                parent=styles["Normal"],
                fontSize=13,
                textColor=colors.HexColor(pdf_config.accent_gold),
                spaceBefore=6,
                spaceAfter=6,
                fontName=font,
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
            ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ]
        return font_commands + base_style_commands

    def make_section_header(self, title: str, content_width: float) -> list:
        bar = Table([[""]], colWidths=[3.2], rowHeights=[18])
        bar.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(pdf_config.accent_gold)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        title_p = Paragraph(title, self.styles["SectionTitle"])
        header = Table([[bar, title_p]], colWidths=[3.2, content_width - 14])
        header.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (0, 0), 0),
                    ("LEFTPADDING", (1, 0), (1, 0), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        return [
            header,
            Spacer(1, 0.06 * inch),
            HRFlowable(
                width="100%",
                thickness=0.8,
                color=colors.HexColor(pdf_config.divider_color),
                spaceBefore=0,
                spaceAfter=10,
            ),
        ]
