from typing import Any

import pandas as pd

from src.agent.utils import truncate_text

DEFAULT_SYSTEM_PROMPT = """你是一位资深的留学顾问。
你的回答要求：
1. **数据驱动**：请务必结合[系统预测结果摘要]中的具体数据（如相似度、录取概率）来回答。
2. **专业口吻**：口吻亲切、专业。
3. **Markdown 格式**：使用加粗、列表等 Markdown 语法提高可读性。
4. **诚实原则**：如果预测结果中没有相关数据，请根据你的经验给出建议，并说明这是基于经验而非系统数据。"""


def format_user_profile(profile: dict[str, Any]) -> str:
    if not profile:
        return "用户尚未填写背景信息。"

    profile_lines = []
    for key, value in profile.items():
        if value is None:
            continue
        v = truncate_text(value, 200)
        if not v:
            continue
        profile_lines.append(f"- {key}: {v}")
    return "\n".join(profile_lines) if profile_lines else "用户尚未填写背景信息。"


def format_prediction_results(prediction_results: Any) -> str:
    if (
        not hasattr(prediction_results, "unified_results")
        or prediction_results.unified_results is None
    ):
        return "用户尚未进行预测。"

    results = prediction_results.unified_results

    if isinstance(results, pd.DataFrame):
        results_summary = results.head(5).to_string()
        return f"以下是部分预测结果摘要(前5行):\n{truncate_text(results_summary, 2000)}"

    if isinstance(results, list):
        if not results:
            return "预测结果为空。"
        top_results = results[:5]
        formatted_items = []
        for i, item in enumerate(top_results):
            if not isinstance(item, dict):
                formatted_items.append(f"{i + 1}. {truncate_text(item, 300)}")
                continue
            item_str = ", ".join(
                f"{k}: {truncate_text(v, 120)}"
                for k, v in item.items()
                if isinstance(k, str) and not k.startswith("_")
            )
            formatted_items.append(f"{i + 1}. {truncate_text(item_str, 500)}")

        results_summary = "\n".join(formatted_items)
        return f"以下是部分预测结果摘要(前5条):\n{truncate_text(results_summary, 2000)}"

    return f"预测结果类型未知: {type(results)}"


def build_consultation_prompt(
    user_query: str,
    user_profile: dict[str, Any],
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
{truncate_text(user_query, 1200)}
"""


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "format_user_profile",
    "format_prediction_results",
    "build_consultation_prompt",
]
