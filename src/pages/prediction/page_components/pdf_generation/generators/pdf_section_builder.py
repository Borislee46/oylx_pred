import io
import random
from datetime import datetime
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from src.utils.logger import setup_logger
from src.pages.prediction.prediction_utils import get_school_major_details
from src.pages.prediction.page_components.pdf_generation.content.school_specific_content import SchoolSpecificContentGenerator
from src.pages.prediction.page_components.pdf_generation.content.radar import create_student_background_radar_chart
from src.pages.prediction.page_components.pdf_generation.config import pdf_config
from src.pages.prediction.page_components.pdf_generation.generators.pdf_styler import PDFStyler
from src.pages.prediction.page_components.pdf_generation.utils import SchoolDataProcessor, ContentFormatter, create_responsive_image

logger = setup_logger("page3", "prediction")

class PDFSectionBuilder:

    def __init__(self, styler: PDFStyler):
        self.styles = styler.styles
        self.styler = styler
        self.school_content_generator = SchoolSpecificContentGenerator()
        self.data_processor = SchoolDataProcessor()
        self.formatter = ContentFormatter()

    def create_cover_page(self, user_nickname: str, report_title: str) -> List:
        story = []
        logo_path = str(pdf_config.get_product_logo_path())
        
        page_width = A4[0] - (pdf_config.layout.left_margin + pdf_config.layout.right_margin) * cm
        page_height = A4[1] - (pdf_config.layout.top_margin + pdf_config.layout.bottom_margin) * cm
        
        logo = create_responsive_image(logo_path, page_width * 0.3, page_height * 0.25)
        if logo:
            logo.hAlign = 'CENTER'
            story.append(logo)
        
        story.append(Spacer(1, 0.5 * inch))
        story.append(Paragraph("EasyApply 留学择校系统", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(report_title, self.styles['SectionTitle']))
        story.append(Spacer(1, 0.5 * inch))
        story.append(Paragraph(f"为 <b>{user_nickname}</b> 定制", self.styles['SubTitle']))
        story.append(Spacer(1, 0.3 * inch))

        intro_text = f"""
        尊敬的 <b>{user_nickname}</b>,
        <br/><br/>
        本报告基于您提供的个人背景信息，通过AI模型深度分析，为您量身定制了一套个性化的留学申请方案。报告将包含以下核心内容：
        <br/>
        - <b>个人背景竞争力分析</b>：全面评估您的学术、语言及软实力背景。
        <br/>
        - <b>智能择校策略解读</b>：系统已为您生成多种不同风险偏好的申请组合，本报告将重点为您解读最具参考价值的<b>“平衡策略”</b>。
        <br/><br/>
        希望这份报告能为您的申请之路提供有力的支持。
        """
        
        left_aligned_body_style = ParagraphStyle(
            name='LeftBodyOnCover',
            parent=self.styles['CustomBody'],
            alignment=TA_LEFT
        )
        story.append(Paragraph(intro_text, left_aligned_body_style))
        story.append(Spacer(1, 0.5 * inch))
        
        current_time = datetime.now().strftime("%Y年%m月%d日")
        story.append(Paragraph(f"报告生成时间: {current_time}", self.styles['CustomBody']))
        
        return story

    def create_background_section(self, user_data: Dict) -> List:
        story = []
        story.append(Paragraph("个人背景竞争力分析", self.styles['SectionTitle']))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("核心维度分析", self.styles['SubTitle']))
        radar_fig = None
        try:
            from src.utils.school_level_service import SCHOOL_LEVEL_PRIORITY
            radar_fig = create_student_background_radar_chart(user_data, SCHOOL_LEVEL_PRIORITY)
            buf = io.BytesIO()
            radar_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            
            page_width = A4[0] - (pdf_config.layout.left_margin + pdf_config.layout.right_margin) * cm
            radar_image = Image(buf)
            radar_image.drawWidth = page_width * 0.5
            radar_image.drawHeight = radar_image.drawHeight * (radar_image.drawWidth / radar_image.imageWidth)
            radar_image.hAlign = 'CENTER'

            story.append(radar_image)
            story.append(Spacer(1, 0.3 * inch))
        except Exception as e:
            logger.error(f"无法生成雷达图: {e}")
            story.append(Paragraph("无法生成核心维度分析图。", self.styles['CustomBody']))
        finally:
            if radar_fig is not None:
                plt.close(radar_fig)

        story.append(Paragraph("背景详情", self.styles['SubTitle']))
        gpa_display = user_data.get('gpa_raw', user_data.get('gpa_score', 'N/A'))
        gpa_scale = user_data.get('gpa_scale', 'N/A')
        language_type = user_data.get('language_type', '语言')
        language_score = user_data.get('language_score_raw', user_data.get('language_score', 'N/A'))

        academic_data = [
            [Paragraph("<b>本科院校</b>", self.styles['CustomBody']), Paragraph(user_data.get('background_university', 'N/A'), self.styles['CustomBody'])],
            [Paragraph("<b>本科专业</b>", self.styles['CustomBody']), Paragraph(user_data.get('background_major', 'N/A'), self.styles['CustomBody'])],
            [Paragraph("<b>GPA成绩</b>", self.styles['CustomBody']), Paragraph(f"{gpa_display} (<b>{gpa_scale}</b>制)", self.styles['CustomBody'])],
            [Paragraph(f"<b>{language_type}成绩</b>", self.styles['CustomBody']), Paragraph(str(language_score), self.styles['CustomBody'])],
        ]
        academic_table = Table(academic_data, colWidths=[2.5*inch, None])
        table_style_commands = self.styler.get_table_style_with_font([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor(pdf_config.style.border_color)),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ])
        academic_table.setStyle(TableStyle(table_style_commands))
        story.append(academic_table)
        story.append(Spacer(1, 0.2 * inch))

        soft_skills = {
            '研究经历': user_data.get('research_count', 0),
            '实习经历': user_data.get('internship_count', 0),
            '获奖经历': user_data.get('award_count', 0),
            '论文发表': user_data.get('paper_count', 0),
        }

        soft_skills_data = []
        for name, count in soft_skills.items():
            soft_skills_data.append([
                Paragraph(name, self.styles['CustomBody']),
                Paragraph(f"{count} 项", self.styles['CustomBody'])
            ])

        soft_skills_table = Table(soft_skills_data, colWidths=[2.5*inch, None])
        table_style_commands = self.styler.get_table_style_with_font([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor(pdf_config.style.border_color)),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ])
        soft_skills_table.setStyle(TableStyle(table_style_commands))
        story.append(soft_skills_table)
        story.append(Spacer(1, 0.3 * inch))

        analyst_notes = self._generate_analyst_notes(user_data, soft_skills)
        
        left_aligned_body_style = ParagraphStyle(
            name='LeftBody',
            parent=self.styles['CustomBody'],
            alignment=TA_LEFT
        )
        notes_paragraph = Paragraph(analyst_notes, left_aligned_body_style)

        notes_table = Table([[notes_paragraph]], style=[
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(pdf_config.style.theme_background)),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor(pdf_config.style.border_color)),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
            ('TOPPADDING', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ])
        story.append(notes_table)

        return story

    def create_optimization_strategies_section(self, optimization_results: Dict, cases_df: pd.DataFrame) -> List:
        story = []
        recommendations = optimization_results.get('recommendations', [])
        core_strategy = next((s for s in recommendations if '平衡' in s.get('strategy_name', '')), recommendations[0] if recommendations else None)

        if not core_strategy:
            story.append(Paragraph("核心申请策略解读", self.styles['SectionTitle']))
            story.append(Paragraph("未能生成核心申请策略的详细建议。", self.styles['CustomBody']))
            return story

        strategy_name = core_strategy.get('strategy_name', '核心策略')
        story.append(Paragraph(f"核心申请策略解读：{strategy_name}", self.styles['SectionTitle']))
        
        strategy_desc = pdf_config.get_strategy_description(strategy_name, len(core_strategy.get('schools', [])))
        story.append(Paragraph(strategy_desc, self.styles['CustomBody']))
        story.append(Spacer(1, 0.2*inch))

        schools = core_strategy.get('schools', [])
        if not schools:
            story.append(Paragraph("此策略下暂无推荐院校。", self.styles['CustomBody']))
        else:
            grouped_schools = self.data_processor.group_schools_by_university(schools)
            rank = 1
            for university, school_group in grouped_schools.items():
                story.extend(self._create_grouped_school_detail(university, school_group, rank, cases_df))
                rank += 1
                if rank <= len(grouped_schools):
                    story.append(Spacer(1, 0.2*inch))
            
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph("祝您申请顺利！", self.styles['CustomBody']))
        
        return story

    def _generate_analyst_notes(self, user_data: Dict, soft_skills: Dict) -> str:
        notes = ["<b>分析师建议:</b>"]
        
        try:
            gpa = float(user_data.get('gpa_score', 0))
            if gpa >= 3.8:
                notes.append("• <b>学术表现卓越</b>：您拥有极具竞争力的GPA成绩，这将是您申请中的核心优势，请务必在文书中突出您的学术能力。")
            elif gpa >= 3.5:
                notes.append("• <b>学术背景良好</b>：您的GPA成绩达到了大部分名校的要求，具备扎实的学术基础。")
            else:
                notes.append("• <b>学术背景提示</b>：您的GPA成绩可能在申请顶尖项目时面临挑战，建议通过高质量的文书、研究经历或GMAT/GRE成绩来弥补。")
        except (ValueError, TypeError):
            pass

        strong_points = [k for k, v in soft_skills.items() if v >= 3]
        if strong_points:
            notes.append(f"• <b>软实力突出</b>：您在 <b>{'、'.join(strong_points)}</b> 方面积累了丰富的经验，这极大地增强了您的申请竞争力。请确保在简历和文书中详细阐述这些经历的深度和成果。")
        else:
            weak_points = [k for k, v in soft_skills.items() if v == 0]
            if len(weak_points) >=2:
                notes.append(f"• <b>软实力建议</b>：我们注意到您在 <b>{'、'.join(weak_points)}</b> 等方面的经历尚有提升空间。在未来的申请或准备过程中，强化相关背景将对您大有裨益。")

        notes.append("• <b>综合建议</b>：请结合我们的三种申请策略，选择最符合您个人目标和风险偏好的方案。祝您申请顺利！")

        return "<br/>".join(notes)

    def _create_grouped_school_detail(self, university: str, school_group: List[Dict], rank: int, cases_df: pd.DataFrame) -> List:
        logo_path = str(pdf_config.get_school_logo_path(university))
        logo_cell_content = create_responsive_image(logo_path, 0.8*inch, 0.8*inch) or Paragraph("", self.styles['SmallText'])

        details_content = []
        
        school_details = self.school_content_generator.school_db.get(university, {})
        ranking = school_details.get("排名", {}).get("QS世界排名", "N/A")
        feature = school_details.get("特点", "")
        
        title_text = f"<b>{university}</b>"
        if ranking != "N/A":
            title_text += f" | <font size='-1'>{ranking}</font>"
        if feature:
            title_text += f" | <font size='-1'>{feature}</font>"
        details_content.append(Paragraph(title_text, self.styles['SubTitle']))

        details_content.append(Spacer(1, 0.1*inch))
        
        for idx, school_data in enumerate(school_group):
            _, major, probability = self.data_processor.extract_school_info(school_data)
            prob_color = pdf_config.get_probability_color(probability)
            prob_text = f"{probability:.0%}"
            
            prob_p = Paragraph(f"<font color='white'>{prob_text}</font>", ParagraphStyle(
                name='ProbTag', fontName=self.styles['CustomBody'].fontName, fontSize=8, textColor=colors.white,
                backColor=prob_color, alignment=TA_CENTER, borderRadius=3, borderPadding=(4, 2, 4, 2)
            ))
            
            major_p = Paragraph(major, self.styles['CustomBody'])
            
            major_table = Table([[major_p, prob_p]], colWidths=[None, '15%'])
            major_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
            details_content.append(major_table)
            
            detailed_info = get_school_major_details(university, major)
            if detailed_info and detailed_info != "暂无详细信息":
                compact_info = self._format_detailed_info_compact(detailed_info)
                details_content.append(Paragraph(compact_info, self.styles['SmallText']))
            
            if idx < len(school_group) - 1:
                details_content.append(Spacer(1, 0.05*inch))

        application_advice_list = school_details.get("申请建议", [])
        if application_advice_list and len(application_advice_list) > 0:
            num_to_select = min(len(application_advice_list), 2)
            if num_to_select > 0:
                selected_advice = random.sample(application_advice_list, num_to_select)
                
                details_content.append(Spacer(1, 0.1*inch))
                advice_points = [f"• {item}" for item in selected_advice]
                advice_text = "<b>申请要点:</b><br/>" + "<br/>".join(advice_points)
                details_content.append(Paragraph(advice_text, self.styles['LeftSmallText']))

        card_table_data = [[logo_cell_content, details_content]]
        card_table = Table(card_table_data, colWidths=[1.0*inch, None])
        table_style_commands = self.styler.get_table_style_with_font([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor(pdf_config.style.border_color)),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ])
        card_table.setStyle(TableStyle(table_style_commands))
        
        return [card_table]

    def _format_detailed_info_compact(self, detailed_info: str) -> str:
        import re
        try:
            lines = detailed_info.split('\n')
            key_fields = {'GPA': 'GPA要求', 'IELTS': 'IELTS', ' TOEFL': 'TOEFL', '学费': '学费', '截止': '申请截止时间'}
            compact_parts = []
            
            for key, field in key_fields.items():
                found_line = next((line for line in lines if field in line), None)
                if found_line and ':' in found_line:
                    value = found_line.split(':', 1)[1].strip()

                    if key == '学费':
                        match = re.search(r'(HK\$|RMB|USD|\$|¥)?\s*[\d,]+', value)
                        if match:
                            value = match.group(0).strip()
                        else:
                            continue
                    
                    if value and value != 'N/A':
                        compact_parts.append(f"<b>{key}</b>: {value}")
            
            return " | ".join(compact_parts) if compact_parts else ""
        except Exception:
            return ""
