# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

_CHINESE_FONT = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(_CHINESE_FONT))


class PDFStyler:
    def __init__(self):
        self.chinese_font_name = _CHINESE_FONT
        self.styles = self._setup_custom_styles()

    def _setup_custom_styles(self):
        styles = getSampleStyleSheet()

        styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=styles["Title"],
                fontSize=18,
                textColor=colors.HexColor("#1E90FF"),
                spaceAfter=20,
                alignment=TA_CENTER,
                fontName=self.chinese_font_name,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SectionTitle",
                parent=styles["Heading1"],
                fontSize=16,
                textColor=colors.HexColor("#333333"),
                spaceBefore=18,
                spaceAfter=12,
                fontName=self.chinese_font_name,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SubTitle",
                parent=styles["Heading2"],
                fontSize=14,
                textColor=colors.HexColor("#555555"),
                spaceBefore=12,
                spaceAfter=6,
                fontName=self.chinese_font_name,
            )
        )
        styles.add(
            ParagraphStyle(
                name="CustomBody",
                parent=styles["Normal"],
                fontSize=12,
                textColor=colors.HexColor("#333333"),
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
                fontSize=10,
                textColor=colors.HexColor("#777777"),
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
                fontSize=14,
                textColor=colors.HexColor("#FF4500"),
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
            ("FONTSIZE", (0, 0), (-1, -1), 12),
        ]
        return font_commands + base_style_commands
