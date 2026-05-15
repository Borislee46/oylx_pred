"""
表单提交日志：构建可读的用户输入摘要 + 防重复记录。

为什么需要 build_user_form_log？
  表单原始数据可能包含几百字的经历描述、特殊字符等。
  日志需要结构化、可搜索、可读，但不需要逐字记录全文。
  → 经历文本截断到 100 字符，数字格式化，列表合并。
"""

from typing import Any

from src.pages.prediction.core.utils import format_field, format_float, format_list_field
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

submission_logger = setup_logger("page3", "prediction")

DEFAULT_SNIPPET_MAX_LEN = 100  # 经历文本的最大截断长度


def _snippet(val: Any, max_len: int = DEFAULT_SNIPPET_MAX_LEN) -> str:
    """截断文本到 max_len 字符，超出部分用 "..." 表示。"""
    if not val:
        return ""
    s = str(val).strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def build_user_form_log(
    session_manager: SessionManager, log_data_source: dict[str, Any]
) -> dict[str, Any]:
    """将原始表单数据转换为结构化日志记录。

    分层处理：
    - 简单字段（院校、GPA分制、考试类型）→ format_field（直接映射）
    - 浮点字段（GPA、语言成绩）→ format_float（指定精度）
    - 列表字段（目标院校、专业）→ format_list_field（合并）
    - 经历详情（四段文本）→ _snippet（截断到 100 字符）

    Returns:
        结构化 dict，适合 JSON 序列化后写入日志。
    """
    exp = log_data_source.get("experience_details", {})

    # 简单字段映射（日志键 → 数据源键）
    mapping = {
        "background_university": "background_university",
        "background_major": "background_major_original",
        "gpa_scale": "gpa_scale",
        "exam_type": "exam_type",
        "exam_score": "exam_score",
        "language_type": "language_type",
        "research_count": "research_count",
        "award_count": "award_count",
        "internship_count": "internship_count",
        "paper_count": "paper_count",
    }

    res = {k: format_field(log_data_source.get(v)) for k, v in mapping.items()}
    res.update(
        {
            "gpa_score": format_float(log_data_source.get("gpa_raw"), 2),
            "language_score": format_float(log_data_source.get("language_score_raw"), 2),
            "target_universities": format_list_field(
                log_data_source.get("target_universities", [])
            ),
            "major_categories": format_list_field(
                session_manager.get("selected_major_categories", [])
            ),
            "target_majors": format_list_field(log_data_source.get("target_majors", [])),
        }
    )

    # 经历文本截断（不记录全文，控制在 100 字符内）
    res.update(
        {
            f: _snippet(exp.get(f))
            for f in ("research_details", "award_details", "internship_details", "paper_details")
        }
    )

    return res


def log_first_submission_if_needed(
    session_manager: SessionManager,
    original_form_data: dict[str, Any] | None,
    input_data_from_form: dict[str, Any],
    session_key_last_submission_logged: str,
) -> None:
    """每个会话只记录一次提交日志（防重复）。

    通过 session_key_last_submission_logged flag 实现幂等：
    第一次提交 → 记录日志 + 设置 flag
    后续提交（重试、跨学部确认后重试）→ 跳过

    为什么需要防重复？
      同一个预测可能因跨学部确认、重试等触发多次 handle_form_submission。
      每次都打日志会导致日志膨胀 + 可读性下降。
    """
    if not session_manager.get(session_key_last_submission_logged, False):
        user_form_log = build_user_form_log(
            session_manager, original_form_data or input_data_from_form
        )
        submission_logger.info(f"用户输入: {user_form_log}")
        session_manager.set(**{session_key_last_submission_logged: True})
