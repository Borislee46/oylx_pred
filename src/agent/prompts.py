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


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "format_user_profile",
    "format_prediction_results",
    "build_consultation_prompt",
]
