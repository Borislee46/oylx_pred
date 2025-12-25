from typing import Any

from src.agent.utils import to_str_singleline


def build_boundary_evaluation_prompt(
    background_major: str,
    boundary_cases: list[dict[str, Any]],
    mode: str,
) -> str:
    is_relax = mode == "relax"
    mode_desc = "放宽(含)" if is_relax else "收紧(删)"
    decision_rule = (
        "若相关则设为 true，否则 false。" if is_relax else "若不相关则设为 true，否则 false。"
    )

    cases_text = []
    for i, case in enumerate(boundary_cases, 1):
        u = to_str_singleline(case.get("university"))
        m = to_str_singleline(case.get("major"))
        cases_text.append(f"{i}. {u} - {m}")

    cases_str = "\n".join(cases_text) if cases_text else "无"

    return f"""请判断以下[待评专业]与用户本科背景专业[{background_major}]是否为相似专业。
    模式: {mode_desc}

    [待评列表]
    {cases_str}

    [规则]
    1. 相似判定标准：学科领域相同、核心课程重叠度高，或在该本科背景下跨专业申请此目标专业属于常规路径（非大幅跨度）。
    2. {mode_desc}模式下，{decision_rule}
    3. 只输出 JSON，不要任何解释。

    [输出示例]
    {{
      "decisions": [bool, ...],
      "needs_adjustment": bool
    }}
    你的回答:"""
