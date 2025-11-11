from typing import Any, Dict

DEFAULT_SYSTEM_PROMPT = (
    """你是一位资深的留学顾问。请根据以下信息，用亲切、专业的口吻回答用户的问题。"""
)


def format_user_profile(profile: Dict[str, Any]) -> str:
    if not profile:
        return "用户尚未填写背景信息。"

    profile_lines = [f"- {key}: {value}" for key, value in profile.items() if value]
    return "\n".join(profile_lines) if profile_lines else "用户尚未填写背景信息。"


def format_prediction_results(prediction_results: Any) -> str:
    if (
        not hasattr(prediction_results, "unified_results")
        or prediction_results.unified_results is None
    ):
        return "用户尚未进行预测。"

    try:
        results_summary = prediction_results.unified_results.head().to_string()
        return f"以下是部分预测结果摘要:\n{results_summary}"
    except Exception:
        return "无法格式化预测结果。"


def build_consultation_prompt(
    user_query: str,
    user_profile: Dict[str, Any],
    prediction_results: Any,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    profile_str = format_user_profile(user_profile)
    results_str = format_prediction_results(prediction_results)

    return f"""
{system_prompt}

[用户信息]
{profile_str}

[系统预测结果摘要]
{results_str}

[用户当前问题]
{user_query}
"""


def build_field_validation_prompt(field_type: str, content: str) -> str:
    field_type_map = {
        "research_details": "科研项目",
        "award_details": "获奖情况",
        "internship_details": "实习经历",
        "paper_details": "论文发表",
    }
    field_name = field_type_map.get(field_type, field_type)

    return f"""请判断以下文本内容是否与"{field_name}"相关。

字段类型: {field_name}
文本内容: {content}

请只回答"是"或"否"，不要添加任何其他说明。
- 如果文本内容与{field_name}相关，回答"是"
- 如果文本内容与{field_name}不相关（例如在{field_name}中填写了其他类型的内容），回答"否"
- 如果文本内容为空或无效，回答"否"

你的回答:"""


def build_analyst_notes_prompt(user_data: Dict[str, Any], soft_skills: Dict[str, Any]) -> str:
    from src.pages.prediction.data_sort_config.top_result_school_order import (
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


def build_boundary_evaluation_prompt(
    background_major: str,
    boundary_cases: list[dict[str, Any]],
    mode: str,
) -> str:
    mode_desc = "放宽" if mode == "relax" else "收紧"
    mode_instruction = (
        "判断以下专业是否应该被包含到相似专业推荐中"
        if mode == "relax"
        else "判断以下专业是否应该从相似专业推荐中移除"
    )

    cases_text = []
    for i, case in enumerate(boundary_cases, 1):
        university = case.get("university", "")
        major = case.get("major", "")
        similarity = case.get("similarity", 0.0)
        probability = case.get("probability", 0.0)
        cases_text.append(
            f"{i}. {university} - {major} (相似度: {similarity:.3f}, 录取概率: {probability:.3f})"
        )

    cases_str = "\n".join(cases_text) if cases_text else "无"

    strictness_instruction = (
        "严格标准：只有在专业与背景专业存在明确的学科关联、课程内容高度重叠、或职业路径直接相关时，才应包含。"
        "相似度低于阈值意味着专业相关性不足，除非存在特殊的跨学科合理性，否则不应包含。"
        "录取概率也应作为重要参考因素。"
        if mode == "relax"
        else "严格标准：只有在专业与背景专业关联度明显不足、课程内容差异较大、或职业路径不匹配时，才应移除。"
        "相似度达到阈值说明专业相关性较高，除非存在明显的不匹配，否则不应移除。"
    )

    return f"""你是一位资深的留学申请顾问。请严格评估以下边界情况下的专业推荐。

[背景专业]
{background_major}

[当前模式]
{mode_desc}模式：{mode_instruction}

[待评估专业列表]
{cases_str}

[严格评估标准]
{strictness_instruction}

[评估维度]
1. 学科关联性：专业与背景专业是否属于同一学科领域或密切相关领域
2. 课程内容重叠度：核心课程是否有实质性重叠
3. 职业路径相关性：职业发展方向是否一致或高度相关
4. 相似度指标：当前相似度低于阈值，需要特别谨慎评估
5. 录取概率：作为辅助参考因素

[要求]
1. 采用严格标准，只有在{"明确相关且合理" if mode == "relax" else "明显不匹配"}的情况下才{"包含" if mode == "relax" else "移除"}
2. 对于{"放宽" if mode == "relax" else "收紧"}模式，应倾向于{"保守" if mode == "relax" else "保留"}，避免过度{"放宽" if mode == "relax" else "收紧"}
3. 如果所有专业都无需调整，设置 needs_adjustment 为 false
4. 返回 JSON 格式，严格遵循以下结构：
   {{
     "decisions": [bool, bool, ...],
     "needs_adjustment": bool,
     "reason": "简要说明原因"
   }}

[输出格式]
请只返回 JSON，不要添加任何其他文字说明。
decisions 数组的长度必须与待评估专业列表的数量一致，按顺序对应。
如果 needs_adjustment 为 false，表示无需转移/替换/截断，所有专业保持原状。

你的回答:"""


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "format_user_profile",
    "format_prediction_results",
    "build_consultation_prompt",
    "build_field_validation_prompt",
    "build_analyst_notes_prompt",
    "build_boundary_evaluation_prompt",
]
