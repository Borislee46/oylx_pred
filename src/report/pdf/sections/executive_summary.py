from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import KeepTogether, Paragraph, Spacer, Table, TableStyle

from src.report.pdf.config import pdf_config
from src.report.pdf.content.school_logo_strip import build_school_logo_strip
from src.report.pdf.generators.pdf_styler import PDFStyler
from src.report.pdf.sections.pathfinder_insights import build_ai_narrative_flowables
from src.report.pdf.tier_labels import tier_display_label
from src.report.pdf.utils import resolve_language_score
from src.utils.numeric import clip_probability_coerce


class ExecutiveSummaryBuilder:
    def __init__(self, styler: PDFStyler):
        self.styles = styler.styles
        self.styler = styler

    def create_executive_summary(
        self,
        user_data: dict,
        unified_results: list[dict] | None,
        ai_explanation: dict | None = None,
    ) -> list:
        story = []
        if not isinstance(user_data, dict):
            user_data = {}
        unified = [r for r in (unified_results or []) if isinstance(r, dict)]

        page_width = A4[0] - (pdf_config.left_margin + pdf_config.right_margin) * cm
        story.extend(self.styler.make_section_header("方案总览", page_width))
        story.append(Spacer(1, 0.04 * inch))

        if not unified:
            story.append(
                Paragraph(
                    "暂无预测结果，完善信息后重新生成可获得完整的行动建议。",
                    self.styles["SmallText"],
                )
            )
            return story

        story.extend(
            build_school_logo_strip(
                unified,
                styles=self.styles,
                chinese_font_name=self.styler.chinese_font_name,
            )
        )

        reach, target, safe = self._band_counts(unified)
        total = reach + target + safe
        best = max(unified, key=lambda r: clip_probability_coerce(r.get("probability")))
        best_prob = clip_probability_coerce(best.get("probability"))
        best_label = tier_display_label(best.get("label"))

        story.append(self._band_chips(reach, target, safe, page_width))
        story.append(Spacer(1, 0.14 * inch))

        story.append(
            Paragraph(
                f"<b>当前最高把握</b>：{best.get('university', '')} | {best.get('major', '')} "
                f"<font color='{pdf_config.high_prob_color}'><b>{best_label} {best_prob:.0%}</b></font>"
                f" &nbsp;|&nbsp; 共 <b>{total}</b> 所院校组合",
                self._lead_style(),
            )
        )
        story.append(Spacer(1, 0.16 * inch))

        ai_blocks = build_ai_narrative_flowables(self.styler, ai_explanation, page_width)
        if ai_blocks:
            story.extend(ai_blocks)
        else:
            story.append(
                Paragraph(
                    self._positioning_text(reach, target, safe, best_prob), self._lead_style()
                )
            )
            story.append(Spacer(1, 0.16 * inch))
            story.append(Paragraph("现在最该做的 3 件事", self.styles["SubTitle"]))
            story.append(Spacer(1, 0.08 * inch))
            actions = self._priority_actions(user_data)
            story.append(KeepTogether(self._action_cards(actions, page_width)))
            story.append(Spacer(1, 0.16 * inch))
            story.append(Paragraph("建议时间线", self.styles["SubTitle"]))
            story.append(Spacer(1, 0.08 * inch))
            story.append(self._timeline_table(actions, page_width))

        return story

    def _band_counts(self, unified: list[dict]) -> tuple[int, int, int]:
        reach = target = safe = 0
        for r in unified:
            label = r.get("label", "")
            if label == "保底":
                safe += 1
            elif label in ("适中", "目标"):
                target += 1
            elif label == "冲刺":
                reach += 1
            else:
                reach += 1
        return reach, target, safe

    def _positioning_text(self, reach: int, target: int, safe: int, best_prob: float) -> str:
        total = reach + target + safe
        if safe == 0 and best_prob < 0.4:
            tone = "目前以冲刺为主，建议同时补强背景并增加稳妥院校，让方案攻守兼备。"
        elif safe / max(total, 1) > 0.5:
            tone = "保底充足、录取无虞，可适当增加冲刺院校去争取更理想的结果。"
        elif reach / max(total, 1) > 0.6:
            tone = "冲刺占比偏高，建议补充目标与保底院校来分散风险。"
        else:
            tone = "冲刺、目标、保底分布均衡，是一套攻守兼备的方案。"
        return (
            f"系统为学生匹配了 <b>{total}</b> 所院校组合，最高把握约 <b>{best_prob:.0%}</b>。{tone}"
        )

    def _priority_actions(self, user_data: dict) -> list[dict]:
        actions: list[dict] = []

        lang = resolve_language_score(
            user_data.get("language_score_raw"), user_data.get("language_score")
        )
        lang_type = user_data.get("language_type", "语言") or "语言"
        gpa = self._to_float(user_data.get("gpa_raw") or user_data.get("gpa"))
        research = int(user_data.get("research_count", 0) or 0)
        internship = int(user_data.get("internship_count", 0) or 0)

        if lang is not None and lang < 6.5:
            actions.append(
                {
                    "title": f"提升{lang_type}成绩",
                    "target": "目标雅思 6.5+ / 托福 90+",
                    "impact": "当前最快见效的提升项，可解锁更多目标与冲刺院校",
                    "phase": "近 1-3 个月",
                }
            )
        if gpa is not None and 0 < gpa < 3.3:
            actions.append(
                {
                    "title": "提升均分或备考 GRE/GMAT",
                    "target": "均分争取 3.3+，或 GRE 320+ / GMAT 650+",
                    "impact": "突破名校硬性门槛，弥补学术短板",
                    "phase": "本学期持续",
                }
            )
        if research == 0 and internship == 0:
            actions.append(
                {
                    "title": "积累 1 段对口实习或科研",
                    "target": "1 段与目标专业相关的经历",
                    "impact": "补齐软背景空白，显著增强文书说服力",
                    "phase": "近 3-6 个月",
                }
            )
        elif internship == 0:
            actions.append(
                {
                    "title": "补充 1 段对口实习",
                    "target": "1 段目标行业实习",
                    "impact": "强化职业方向，利好商科/职业导向项目",
                    "phase": "近 3-6 个月",
                }
            )
        elif research == 0:
            actions.append(
                {
                    "title": "补充 1 段科研或项目",
                    "target": "1 段研究型经历或课程项目",
                    "impact": "强化学术潜力，利好研究型项目",
                    "phase": "近 3-6 个月",
                }
            )

        actions.append(
            {
                "title": "打磨文书与推荐信",
                "target": "讲清申请动机与背景匹配度",
                "impact": "在硬件接近时拉开差距，提升冲刺院校竞争力",
                "phase": "申请季前",
            }
        )
        return actions[:3]

    def _lead_style(self) -> ParagraphStyle:
        return ParagraphStyle(
            "_ExecLead",
            parent=self.styles["CustomBody"],
            alignment=TA_LEFT,
            leading=17,
            spaceAfter=0,
        )

    def _band_chips(self, reach: int, target: int, safe: int, page_width: float) -> Table:
        def chip(count: int, label: str, color: str):
            num = Paragraph(
                f"<font color='{color}'><b>{count}</b></font>",
                ParagraphStyle(
                    "_ChipNum",
                    parent=self.styles["CustomBody"],
                    fontName=self.styler.chinese_font_name,
                    fontSize=26,
                    leading=30,
                    alignment=TA_CENTER,
                    spaceAfter=0,
                ),
            )
            lab = Paragraph(
                f"<font color='{pdf_config.text_secondary}'>{label}</font>",
                ParagraphStyle(
                    "_ChipLab", parent=self.styles["SmallText"], alignment=TA_CENTER, spaceAfter=0
                ),
            )
            return [[num], [lab]]

        cells = [
            Table(chip(reach, "冲刺", pdf_config.low_prob_color)),
            Table(chip(target, "目标", pdf_config.medium_prob_color)),
            Table(chip(safe, "保底", pdf_config.high_prob_color)),
        ]
        gap = 0.14 * inch
        col_w = (page_width - 2 * gap) / 3
        outer = Table(
            [[cells[0], "", cells[1], "", cells[2]]], colWidths=[col_w, gap, col_w, gap, col_w]
        )
        style = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]
        for col, color in (
            (0, pdf_config.low_prob_color),
            (2, pdf_config.medium_prob_color),
            (4, pdf_config.high_prob_color),
        ):
            style += [
                ("BACKGROUND", (col, 0), (col, 0), colors.HexColor(pdf_config.panel_background)),
                ("LINEBEFORE", (col, 0), (col, 0), 3, colors.HexColor(color)),
                ("BOX", (col, 0), (col, 0), 0.8, colors.HexColor(pdf_config.border_color)),
            ]
        outer.setStyle(TableStyle(style))
        return outer

    def _action_cards(self, actions: list[dict], page_width: float) -> list:
        out = []
        for i, a in enumerate(actions, 1):
            badge = Paragraph(
                f"<font color='white'><b>{i}</b></font>",
                ParagraphStyle(
                    "_ActBadge",
                    parent=self.styles["CustomBody"],
                    fontName=self.styler.chinese_font_name,
                    fontSize=14,
                    textColor=colors.white,
                    alignment=TA_CENTER,
                    leading=18,
                ),
            )
            body = Paragraph(
                f"<b>{a['title']}</b><br/>"
                f"<font color='{pdf_config.text_secondary}' size='9'>目标：{a['target']}</font><br/>"
                f"<font color='{pdf_config.text_muted}' size='9'>{a['impact']}</font>",
                ParagraphStyle(
                    "_ActBody",
                    parent=self.styles["CustomBody"],
                    alignment=TA_LEFT,
                    leading=15,
                    spaceAfter=0,
                ),
            )
            card = Table([[badge, body]], colWidths=[0.5 * inch, page_width - 0.5 * inch])
            card.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(pdf_config.accent_gold)),
                        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(pdf_config.card_background)),
                        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(pdf_config.border_color)),
                        ("LEFTPADDING", (0, 0), (0, 0), 0),
                        ("RIGHTPADDING", (0, 0), (0, 0), 0),
                        ("LEFTPADDING", (1, 0), (1, 0), 12),
                        ("RIGHTPADDING", (1, 0), (1, 0), 12),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ]
                )
            )
            out.append(card)
            if i < len(actions):
                out.append(Spacer(1, 0.08 * inch))
        return out

    def _timeline_table(self, actions: list[dict], page_width: float) -> Table:
        phases = [
            ("近 1-3 个月", "补考语言 / 启动背景提升"),
            ("申请季前", "确定选校清单，打磨文书与推荐信"),
            ("申请季", "按冲稳保梯度递交，关注各校截止时间"),
        ]
        rows = []
        head_style = ParagraphStyle(
            "_TLHead",
            parent=self.styles["CustomBody"],
            textColor=colors.white,
            alignment=TA_LEFT,
            spaceAfter=0,
        )
        cell_style = ParagraphStyle(
            "_TLCell", parent=self.styles["CustomBody"], alignment=TA_LEFT, spaceAfter=0, leading=15
        )
        for phase, desc in phases:
            rows.append(
                [
                    Paragraph(f"<b>{phase}</b>", head_style),
                    Paragraph(desc, cell_style),
                ]
            )
        table = Table(rows, colWidths=[1.5 * inch, page_width - 1.5 * inch])
        table.setStyle(
            TableStyle(
                self.styler.get_table_style_with_font(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(pdf_config.theme_primary)),
                        (
                            "BACKGROUND",
                            (1, 0),
                            (1, -1),
                            colors.HexColor(pdf_config.panel_background),
                        ),
                        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(pdf_config.border_color)),
                        (
                            "LINEBELOW",
                            (0, 0),
                            (-1, -2),
                            0.6,
                            colors.HexColor(pdf_config.divider_color),
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
        )
        return table

    @staticmethod
    def _to_float(v) -> float | None:
        try:
            return float(v) if v not in (None, "", "未填写", "N/A") else None
        except (ValueError, TypeError):
            return None
