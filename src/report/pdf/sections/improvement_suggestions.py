from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import Paragraph, Spacer

from src.report.pdf.config import pdf_config
from src.report.pdf.generators.pdf_styler import PDFStyler
from src.report.pdf.utils import resolve_language_score


class ImprovementSectionBuilder:
    def __init__(self, styler: PDFStyler):
        self.styles = styler.styles
        self.styler = styler

    def create_improvement_section(
        self,
        user_data: dict,
        unified_results: list[dict] | None,
        ai_explanation: dict | None = None,
    ) -> list:
        from src.report.pdf.sections.pathfinder_insights import has_rich_ai_narrative

        story = []
        if not user_data or not isinstance(user_data, dict):
            user_data = {}

        ai_rich = has_rich_ai_narrative(ai_explanation)
        page_width = A4[0] - (pdf_config.left_margin + pdf_config.right_margin) * cm

        header = "背景诊断补充" if ai_rich else "提升建议与择校策略"
        story.extend(self.styler.make_section_header(header, page_width))
        story.append(Spacer(1, 0.06 * inch))

        content = []
        profile_items = self._build_profile_items(user_data)
        if profile_items:
            content.append("<b>背景评估要点</b>" if ai_rich else "<b>背景提升方向</b>")
            for item in profile_items:
                content.append(item)

        if unified_results and not ai_rich:
            strategy_text = self._build_strategy_text(user_data, unified_results)
            if strategy_text:
                if content:
                    content.append("")
                content.append("<b>择校策略建议</b>")
                content.append(strategy_text)

        if not content:
            if ai_rich:
                return []
            story.append(
                Paragraph(
                    "暂无可用数据，建议完善个人信息后重新生成报告。",
                    self.styles["SmallText"],
                )
            )
            return story

        body = "<br/>".join(content)
        body_style = ParagraphStyle(
            "_ImproveBody",
            parent=self.styles["CustomBody"],
            alignment=TA_LEFT,
            leading=16,
            spaceAfter=0,
        )
        story.append(Paragraph(body, body_style))
        return story

    def _build_profile_items(self, user_data: dict) -> list[str]:
        items = []

        gpa = user_data.get("gpa") or user_data.get("gpa_raw")
        if gpa is not None and gpa != "未填写":
            try:
                gpa_val = float(gpa)
                if gpa_val >= 3.7:
                    items.append(
                        f"&bull; GPA {gpa_val:.2f} "
                        f"<font color='{pdf_config.high_prob_color}'><b>优秀</b></font> "
                        "&mdash; 核心竞争力，文书中充分展现学术能力"
                    )
                elif gpa_val >= 3.3:
                    items.append(
                        f"&bull; GPA {gpa_val:.2f} "
                        f"<font color='{pdf_config.medium_prob_color}'><b>良好</b></font> "
                        "&mdash; 争取提升至 3.5+ 以增强冲刺竞争力"
                    )
                else:
                    target = min(gpa_val + 0.3, 3.5)
                    items.append(
                        f"&bull; GPA {gpa_val:.2f} "
                        f"<font color='{pdf_config.low_prob_color}'><b>需提升</b></font> "
                        f"&mdash; 建议提升至 {target:.1f}+，或通过 GRE/GMAT 弥补"
                    )
            except (ValueError, TypeError):
                pass

        lang_val = resolve_language_score(
            user_data.get("language_score_raw"), user_data.get("language_score")
        )
        lang_type = user_data.get("language_type", "语言")
        if lang_val is not None:
            if lang_val >= 7.0:
                items.append(
                    f"&bull; {lang_type} {lang_val} "
                    f"<font color='{pdf_config.high_prob_color}'><b>优秀</b></font>"
                )
            elif lang_val >= 6.0:
                items.append(
                    f"&bull; {lang_type} {lang_val} "
                    f"<font color='{pdf_config.medium_prob_color}'><b>基本达标</b></font> "
                    "&mdash; 冲刺名校建议刷至 7.0+"
                )
            else:
                items.append(
                    f"&bull; {lang_type} {lang_val} "
                    f"<font color='{pdf_config.low_prob_color}'><b>需提升</b></font> "
                    "&mdash; 目标 6.5+，部分院校接受六级或语言班"
                )

        research = user_data.get("research_count", 0) or 0
        internship = user_data.get("internship_count", 0) or 0
        award = user_data.get("award_count", 0) or 0
        paper = user_data.get("paper_count", 0) or 0
        total_soft = research + internship + award + paper

        if total_soft == 0:
            items.append(
                f"&bull; 软实力：<font color='{pdf_config.low_prob_color}'><b>空白</b></font> "
                "&mdash; 优先积累 1-2 段研究或实习经历"
            )
        else:
            strong = []
            if research >= 2:
                strong.append(f"研究 {research} 段")
            if internship >= 2:
                strong.append(f"实习 {internship} 段")
            if award >= 2:
                strong.append(f"获奖 {award} 项")
            if paper >= 1:
                strong.append(f"论文 {paper} 篇")
            weak = []
            if research == 0:
                weak.append("研究")
            if internship == 0:
                weak.append("实习")
            if award == 0:
                weak.append("获奖")

            if strong:
                items.append(
                    f"&bull; 软实力：{'、'.join(strong)} "
                    f"<font color='{pdf_config.high_prob_color}'>积累充足</font>"
                )
            if weak:
                items.append(f"&bull; 待补强：{'、'.join(weak)}方面可进一步积累")
            if not strong and not weak:
                items.append("&bull; 软实力各项均衡，建议持续积累")

        return items

    def _build_strategy_text(self, user_data: dict, unified_results: list[dict]) -> str | None:
        if not unified_results:
            return None

        reach_count = 0
        target_count = 0
        safe_count = 0
        for r in unified_results:
            label = r.get("label", "")
            if label == "保底":
                safe_count += 1
            elif label in ("适中", "目标"):
                target_count += 1
            elif label == "冲刺":
                reach_count += 1
            else:
                reach_count += 1

        total = safe_count + target_count + reach_count
        if total == 0:
            return None

        dist = []
        if reach_count:
            dist.append(
                f"<font color='{pdf_config.low_prob_color}'><b>冲刺 {reach_count} 所</b></font>"
            )
        if target_count:
            dist.append(
                f"<font color='{pdf_config.medium_prob_color}'><b>目标 {target_count} 所</b></font>"
            )
        if safe_count:
            dist.append(
                f"<font color='{pdf_config.high_prob_color}'><b>保底 {safe_count} 所</b></font>"
            )

        if reach_count / max(total, 1) > 0.6:
            advice = "冲刺占比偏高，建议增加目标与保底院校分散风险。"
        elif safe_count / max(total, 1) > 0.5:
            advice = "保底占比偏高，可适当增加冲刺院校挑战更高目标。"
        else:
            advice = "冲刺、目标、保底搭配合理，方案攻守兼备。"

        actions = []
        gpa = user_data.get("gpa") or user_data.get("gpa_raw")
        if gpa is not None and gpa != "未填写":
            try:
                if float(gpa) < 3.3:
                    actions.append("提升 GPA 或准备 GRE/GMAT")
            except (ValueError, TypeError):
                pass
        lang2 = resolve_language_score(
            user_data.get("language_score_raw"), user_data.get("language_score")
        )
        if lang2 is not None and lang2 < 6.5:
            actions.append("备考语言考试，目标 6.5+")
        research = user_data.get("research_count", 0) or 0
        internship = user_data.get("internship_count", 0) or 0
        if research == 0 and internship == 0:
            actions.append("积累研究或实习经历")
        if not actions:
            actions.append("精心打磨文书与推荐信")

        action_text = "、".join(actions[:3])

        return (
            f"共 <b>{total}</b> 所院校 &nbsp;|&nbsp; {' &nbsp;|&nbsp; '.join(dist)}"
            f"<br/>&bull; {advice}"
            f"<br/>&bull; 优先行动：{action_text}"
        )
