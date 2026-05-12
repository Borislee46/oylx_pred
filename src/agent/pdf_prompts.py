# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
from typing import Any


def build_analyst_notes_prompt(user_data: dict[str, Any], soft_skills: dict[str, Any]) -> str:
    from src.pages.prediction.data_sort_config import (
        UNIVERSITY_SORT_ORDER,
    )

    available_universities = "、".join(UNIVERSITY_SORT_ORDER[:15])

    system_prompt = f"""你是一位资深的留学申请顾问分析师。请根据学生的背景信息，生成专业、有针对性的分析师建议。

重要限制：
- 建议中只能提及以下地区的院校：香港、新加坡、澳门、马来西亚
- 可提及的具体院校包括（但不限于）：{available_universities}
- 严禁提及美国、英国、加拿大、澳大利亚等其他国家或地区的院校
- 如需要提及院校，请仅从上述范围内选择

要求：
1. 分析学生的学术背景（GPA、院校、专业等）
2. 评估学生的软实力表现（研究经历、实习经历、获奖情况、论文发表等）
3. 提供具体的申请指导建议，但仅针对香港、新加坡、澳门、马来西亚的院校
4. 使用专业但友好的语气
5. 如果提及院校名称，必须确保是上述范围内的院校

输出格式要求：
- 使用HTML格式
- 使用 <b>标签突出关键点，如：<b>学术表现卓越</b>
- 每条建议以 "- " 开头
- 使用 <br/> 分隔各项建议
- 第一行必须是：<b>分析师建议:</b>
- 严格控制字数：总共3-4条建议，每条不超过60字，总字数不超过200字
- 语言简洁精炼，避免冗长描述，每条建议直接点明核心要点"""

    background_info = []

    gpa_value = user_data.get("gpa_score") or user_data.get("gpa_raw")
    gpa_scale = user_data.get("gpa_scale", "")
    if gpa_value and gpa_value not in ("未填写", "", None):
        gpa_str = str(gpa_value)
        if gpa_scale and gpa_scale not in ("未知", "", None):
            gpa_str += f" ({gpa_scale}制)"
        background_info.append(f"GPA成绩: {gpa_str}")

    university = user_data.get("background_university")
    if university and university not in ("未填写", "", None):
        background_info.append(f"本科院校: {university}")

    major = user_data.get("background_major")
    if major and major not in ("未填写", "", None):
        background_info.append(f"本科专业: {major}")

    language_type = user_data.get("language_type", "语言")
    language_score = user_data.get("language_score_raw") or user_data.get("language_score")
    if language_score and language_score not in ("未填写", "", None):
        background_info.append(f"{language_type}成绩: {language_score}")

    soft_skills_info = []
    for skill_name, count in soft_skills.items():
        if isinstance(count, (int, float)) and count > 0:
            soft_skills_info.append(f"{skill_name}: {int(count)}项")

    if soft_skills_info:
        background_info.append(f"软实力: {', '.join(soft_skills_info)}")

    background_text = "\n".join(background_info) if background_info else "学生背景信息不完整"

    return f"""{system_prompt}

[学生背景信息]
{background_text}

请基于以上信息生成分析师建议，格式必须严格按照要求，并严格遵守院校范围限制。"""
