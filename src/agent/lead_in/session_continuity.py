from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from src.agent.context import StudentContext
from src.pages.prediction.handler_config import DEFAULT_FORM_KEYS, DEFAULT_SESSION_KEYS

SessionContinuity = Literal["continue", "fresh"]
_log = logging.getLogger("session_continuity")

TURN_KEY = "_lead_in_turn"

# 注意：历史上这里还有一个 `_SCHOOL_HINT`（枚举"大学|中大|北大|…"）和一个
# `_TARGET_OR_QUESTION_RE`（含目标词就判为"非新档案"）。两者都已删除：
# 前者认不出"复旦计算机"这类没有"大学"二字的写法，
# 后者把新学生几乎必然会带的"想申/想去/要求"当成了延续信号。
# 现在校名识别改用 `_school_prefix_index()` 走院校库。

_BG_IDENTITY_RE = re.compile(
    r"我是|我本科|我就读|就读于|毕业于|来自|我研究生|我硕士|我博士|"
    r"换一个学生|换人|新学生|另一位|另一个学生|"
    r"study at|studying at|graduate from|my university|my school|my major",
    re.IGNORECASE,
)

_PROFILE_FIELD_RE = re.compile(
    r"GPA|gpa|均分|绩点|雅思|托福|IELTS|TOEFL|GRE|GMAT|专业|major|成绩|\d+\s*分",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SessionContinuityResult:
    continuity: SessionContinuity
    source: Literal["default", "llm", "disabled"]
    confidence: str = "high"
    reason: str = ""


def has_prior_context(session_manager: Any | None, ctx: StudentContext) -> bool:
    if ctx.conversation_turns:
        return True
    if session_manager is None:
        return False
    if session_manager.get(DEFAULT_SESSION_KEYS.has_predicted, False):
        return True
    if session_manager.get(DEFAULT_FORM_KEYS.background_university):
        return True
    bg = ctx.extracted_background or {}
    return bool(bg.get("university") or bg.get("major"))


def build_form_snapshot(session_manager: Any | None) -> str:
    if session_manager is None:
        return "（无）"
    parts: list[str] = []
    uni = session_manager.get(DEFAULT_FORM_KEYS.background_university)
    if uni:
        parts.append(f"背景院校={uni}")
    uh = session_manager.get(DEFAULT_FORM_KEYS.user_history_data, {}) or {}
    major = uh.get("background_major_original") or session_manager.get("background_major")
    if major:
        parts.append(f"背景专业={major}")
    gpa = session_manager.get(DEFAULT_FORM_KEYS.gpa_raw_input)
    if gpa is not None:
        scale = session_manager.get(DEFAULT_FORM_KEYS.gpa_scale) or ""
        parts.append(f"GPA={gpa}" + (f"/{scale}" if scale else ""))
    lang_type = session_manager.get(DEFAULT_FORM_KEYS.language_type)
    lang_score = session_manager.get(DEFAULT_FORM_KEYS.language_score_input)
    if lang_type and lang_score is not None:
        parts.append(f"语言={lang_type}{lang_score}")
    countries = session_manager.get(DEFAULT_FORM_KEYS.selected_target_countries) or []
    if countries:
        parts.append(f"目标地区={','.join(str(c) for c in countries)}")
    schools = session_manager.get(DEFAULT_FORM_KEYS.selected_target_universities) or []
    if schools:
        parts.append(f"目标院校={','.join(str(s) for s in schools[:4])}")
    majors = session_manager.get(DEFAULT_FORM_KEYS.selected_target_majors) or []
    if majors:
        parts.append(f"目标专业={','.join(str(m) for m in majors[:3])}")
    if session_manager.get(DEFAULT_SESSION_KEYS.has_predicted, False):
        parts.append("已生成预测结果=是")
    return " | ".join(parts) if parts else "（空）"


def build_continuity_user_prompt(
    text: str,
    ctx: StudentContext,
    session_manager: Any | None,
) -> str:
    blocks: list[str] = []
    snapshot = build_form_snapshot(session_manager)
    blocks.append(f"## 当前表单快照\n{snapshot}")
    turns = ctx.conversation_turns or []
    if turns:
        blocks.append("## 对话历史")
        for t in turns[-6:]:
            role = "顾问" if t.get("role") == "user" else "AI"
            content = (t.get("content") or t.get("summary") or "").strip()
            if content:
                blocks.append(f"- {role}：{content[:200]}")
    blocks.append(f"## 当前输入\n{(text or '').strip()}")
    return "\n\n".join(blocks)


def _prior_university(ctx: StudentContext, session_manager: Any | None) -> str:
    bg = ctx.extracted_background or {}
    uni = str(bg.get("university") or "").strip()
    if uni:
        return uni
    if session_manager is None:
        return ""
    return str(session_manager.get(DEFAULT_FORM_KEYS.background_university) or "").strip()


@lru_cache(maxsize=1)
def _school_prefix_index() -> dict[str, str]:
    """{校名前缀: 完整校名}，用于在自由文本里认出**背景院校**实体。

    为什么不继续用正则：`复旦计算机 GPA3.6…` 里根本没有"大学"二字，
    `大学|中大|北大|港大…` 这类枚举必然漏（实测 turn 4 就是这么漏掉的）。

    ⚠️ 必须用**背景院校库**（`school_base.feather`，8814 所）而不是目标院校库
    （`school_major_details` 只有 80 所项目院校）——"北大/复旦"是背景院校，
    在目标库里根本不存在（这一点是被单测抓出来的）。
    """
    names: list[str] = []
    try:
        from src.utils.schools.level_service import _get_school_level_mapping

        names = list(_get_school_level_mapping().keys())
    except Exception:
        _log.debug("SESSION_CONTINUITY | background school index unavailable", exc_info=True)

    if not names:
        try:  # 退化到目标院校库，聊胜于无
            from src.agent.tools.probe_tools import _valid_universities

            names = list(_valid_universities())
        except Exception:
            return {}

    index: dict[str, str] = {}
    for name in names:
        for size in (4, 3, 2):
            if len(name) >= size:
                index.setdefault(name[:size], name)
    return index


def _other_school_in_text(text: str, prior_uni: str) -> str:
    """文本里是否出现了"另一个"学校实体？返回命中的完整校名。

    先长后短地扫 n-gram，降低"中山路"误撞"中山大学"这类误判。
    """
    index = _school_prefix_index()
    if not index:
        return ""
    for size in (4, 3, 2):
        for i in range(0, max(0, len(text) - size + 1)):
            name = index.get(text[i : i + size])
            if name and name != prior_uni:
                return name
    return ""


def should_reset_for_new_profile(
    text: str,
    ctx: StudentContext,
    session_manager: Any | None = None,
) -> bool:
    """客观兜底：判定这条输入是不是"另一位学生"。

    ⚠️ 这是**兜底**，不是主要机制。主要机制是智能体自己调用 `start_new_profile()`
    —— 会话边界是意图判断，只有 LLM 读得懂。兜底存在的理由是漏判代价很高：
    两个学生的字段混在一起，还会拿上一位的背景去做预测。

    判据（任一满足即重置）：
    1. **明确的身份陈述 / 换人指令**（"我是…""换一个学生""重新开始"）；
    2. **出现了另一个学校实体，且这句话在陈述背景字段**（GPA/均分/雅思/专业…）。

    刻意不再使用"含目标或提问关键词就不重置"这条：新学生几乎必然带着
    "想申/想去/要求"，那条规则等于把最常见的新档案输入全排除了（实测 turn 5）。
    """
    stripped = (text or "").strip()
    if not stripped or not has_prior_context(session_manager, ctx):
        return False

    prior_uni = _prior_university(ctx, session_manager)
    if not prior_uni:
        return False

    if prior_uni in stripped:
        return False

    if _BG_IDENTITY_RE.search(stripped):
        _log.info(
            "SESSION_CONTINUITY | identity fresh | prior_uni=%s text=%s",
            prior_uni[:40],
            stripped[:80],
        )
        return True

    # 必须同时是"在陈述背景" + "提到了另一个学校"，才动档案。
    # 只有"想申港大"这种目标表述、没有任何背景字段时，不动。
    if not _PROFILE_FIELD_RE.search(stripped):
        return False

    other = _other_school_in_text(stripped, prior_uni)
    if other:
        _log.info(
            "SESSION_CONTINUITY | other-school fresh | prior_uni=%s found=%s text=%s",
            prior_uni[:40],
            other,
            stripped[:80],
        )
        return True

    _log.info(
        "SESSION_CONTINUITY | keep continue | prior_uni=%s text=%s",
        prior_uni[:40],
        stripped[:80],
    )
    return False


def apply_fresh_session(
    session_manager: Any,
    ctx: StudentContext,
    *,
    state_machine: Any | None = None,
) -> None:
    from src.agent.lead_in.dispatch_constants import MSGS_KEY
    from src.pages.prediction.form_bridge import reset_lead_in_profile

    reset_lead_in_profile(session_manager, ctx, reset_state_machine=False)
    ctx.conversation_turns = []
    ctx.extracted_background = {}
    # 关键：pydantic 的 message_history 也必须清掉。
    # 只清表单不清历史，模型下一轮仍能看到上一位学生的对话与工具调用记录，
    # 于是"换了个学生"但上下文还是旧的（实测 turn 递增、prompt 越来越长就是这个原因）。
    session_manager.set(**{TURN_KEY: 0, MSGS_KEY: None, "lead_in_consumed": False})
    if state_machine is not None:
        state_machine.update(conversation_turns=[], turn=0)
    _log.info("SESSION_CONTINUITY | applied fresh reset")
