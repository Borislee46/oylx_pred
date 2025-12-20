from typing import Any

from src.agent.utils import to_float, to_str_singleline


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
        university = to_str_singleline(case.get("university"))
        major = to_str_singleline(case.get("major"))
        similarity = to_float(case.get("similarity"))
        probability = to_float(case.get("probability"))
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

[要求]
1. 采用严格标准，只有在{"明确相关且合理" if mode == "relax" else "明显不匹配"}的情况下才{"包含" if mode == "relax" else "移除"}
2. 对于{"放宽" if mode == "relax" else "收紧"}模式，应倾向于{"保守" if mode == "relax" else "保留"}，避免过度{"放宽" if mode == "relax" else "收紧"}
3. 如果所有专业都无需调整，设置 needs_adjustment 为 false
4. 返回 JSON 格式，必须包含 reasoning 字段说明评估逻辑，严格遵循以下结构：
   {{
     "reasoning": "对本次评估逻辑的简要说明",
     "decisions": [bool, bool, ...],
     "needs_adjustment": bool
   }}

[输出格式]
请只返回 JSON，不要添加任何其他文字说明。
decisions 数组的长度必须与待评估专业列表的数量一致，按顺序对应。
如果 needs_adjustment 为 false，表示无需转移/替换/截断，所有专业保持原状。

你的回答:"""
