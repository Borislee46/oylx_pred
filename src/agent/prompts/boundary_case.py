from typing import Any

from src.agent.utils import to_str_singleline


def build_boundary_evaluation_prompt(
    background_major: str,
    boundary_cases: list[dict[str, Any]],
    mode: str,
) -> str:
    is_relax = mode == "relax"
    mode_desc = "放宽(含)" if is_relax else "收紧(删)"

    mode_rule = (
        "放宽模式下：若目标专业与本科背景相关则返回 true，否则返回 false。"
        if is_relax
        else "收紧模式下：若目标专业与本科背景不相关（应删除）则返回 true，否则返回 false。"
    )

    cases_text = []
    for i, case in enumerate(boundary_cases, 1):
        u = to_str_singleline(case.get("university"))
        m = to_str_singleline(case.get("major"))
        cases_text.append(f"{i}. {u} - {m}")

    cases_str = "\n".join(cases_text) if cases_text else "无"

    examples = ""
    if is_relax:
        examples = """[判定示例 - 放宽模式]
    - 本科[Computer Science] vs 目标[Data Science]: true (高度相关)
    - 本科[Economics] vs 目标[Finance]: true (常规跨专业路径)
    - 本科[Mechanical Engineering] vs 目标[History]: false (跨度过大)"""
    else:
        examples = """[判定示例 - 收紧模式]
    - 本科[Computer Science] vs 目标[Software Engineering]: false (不应删除，属于同专业)
    - 本科[Biology] vs 目标[Finance]: true (应删除，相关性极低)
    - 本科[Mathematics] vs 目标[Statistics]: false (不应删除，数学背景申请统计是常规)"""

    return f"""请判断以下[待评专业]与用户本科背景专业[{background_major}]是否为相似专业。
    模式: {mode_desc}
    规则: {mode_rule}

    {examples}

    [待评列表]
    {cases_str}

    [输出示例]
    {{
      "decisions": [bool, ...],
      "needs_adjustment": bool
    }}
    你的回答:"""
