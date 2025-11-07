from typing import Dict

from src.pages.prediction.page_components.pdf_generation.generators.pdf_ai_agent import (
    PDFAIAgent,
)


class AnalystNotesGenerator:
    def __init__(self, ai_agent: PDFAIAgent):
        self.ai_agent = ai_agent

    def generate_analyst_notes(self, user_data: Dict, soft_skills: Dict) -> str:
        if not user_data:
            user_data = {}
        if not soft_skills:
            soft_skills = {}

        ai_notes = self.ai_agent.generate_analyst_notes(user_data, soft_skills)
        if ai_notes and ai_notes.strip():
            if not ai_notes.startswith("<b>分析师建议:</b>"):
                ai_notes = "<b>分析师建议:</b><br/>" + ai_notes
            return ai_notes

        return self._generate_fallback_analyst_notes(user_data, soft_skills)

    def _generate_fallback_analyst_notes(self, user_data: Dict, soft_skills: Dict) -> str:
        notes = ["<b>分析师建议:</b>"]

        gpa_value = user_data.get("gpa_score", 0)
        if isinstance(gpa_value, str) and (gpa_value == "未填写" or not gpa_value.strip()):
            gpa = 0.0
        else:
            try:
                gpa = float(gpa_value) if gpa_value else 0.0
            except (ValueError, TypeError):
                gpa = 0.0

        if gpa >= 3.8:
            notes.append(
                "- <b>学术表现卓越</b>：该学生拥有极具竞争力的GPA成绩，这将是其申请中的核心优势，"
                "建议顾问在指导文书写作时突出学生的学术能力。"
            )
        elif gpa >= 3.5:
            notes.append(
                "- <b>学术背景良好</b>：学生的GPA成绩达到了大部分名校的要求，具备扎实的学术基础。"
            )
        elif gpa > 0:
            notes.append(
                "- <b>学术背景提示</b>：学生的GPA成绩可能在申请顶尖项目时面临挑战，"
                "建议顾问通过高质量的文书、研究经历或GMAT/GRE成绩来帮助学生弥补短板。"
            )

        strong_points = [
            k for k, v in soft_skills.items() if isinstance(v, (int, float)) and v >= 3
        ]
        if strong_points:
            points_str = "、".join(strong_points)
            notes.append(
                f"- <b>软实力突出</b>：该学生在 <b>{points_str}</b> 方面积累了丰富的经验，"
                "这极大地增强了其申请竞争力。建议顾问在指导学生撰写简历和文书时，详细阐述这些经历的深度和成果。"
            )
        else:
            weak_points = [
                k for k, v in soft_skills.items() if isinstance(v, (int, float)) and v == 0
            ]
            if len(weak_points) >= 2:
                points_str = "、".join(weak_points)
                notes.append(
                    f"- <b>软实力建议</b>：我们注意到该学生在 <b>{points_str}</b> 等方面的经历尚有提升空间。"
                    "建议顾问在后续指导过程中，帮助学生强化相关背景。"
                )

        notes.append(
            "- <b>综合建议</b>：建议顾问结合三种申请策略，为学生选择最符合其个人目标和风险偏好的方案。"
            "报告仅供参考，具体申请方案需结合实际情况进行调整。"
        )

        return "<br/>".join(notes)

