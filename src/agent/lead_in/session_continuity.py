from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from src.agent.context import StudentContext
from src.pages.prediction.handler_config import DEFAULT_FORM_KEYS, DEFAULT_SESSION_KEYS

SessionContinuity = Literal["continue", "fresh"]
_log = logging.getLogger("session_continuity")

TURN_KEY = "_lead_in_turn"

_SCHOOL_HINT = re.compile(
    r"(大学|学院|学校|University|College|Univ\.?|中大|北大|清华|港大|港中文|港科大)",
    re.IGNORECASE,
)

_BG_IDENTITY_RE = re.compile(
    r"我是|我本科|我就读|就读于|毕业于|来自|我研究生|我硕士|我博士|"
    r"换一个学生|换人|新学生|另一位|另一个学生|"
    r"study at|studying at|graduate from|my university|my school|my major",
    re.IGNORECASE,
)

_TARGET_OR_QUESTION_RE = re.compile(
    r"想去|想申|想申请|申请|目标|考虑|门槛|要求|怎么样|怎样|如何|哪个|哪所|"
    r"录取概率|难吗|难不难|好不好|好申吗|对比|是不是|能申|可以申|有机会|录取率|"
    r"能不能|需要|适合|匹配|认可|接受|了解|咨询|打听|情况|条件|可以读|可以吗",
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


def should_reset_for_new_profile(
    text: str,
    ctx: StudentContext,
    session_manager: Any | None = None,
) -> bool:
    stripped = (text or "").strip()
    if not stripped or not has_prior_context(session_manager, ctx):
        return False

    prior_uni = _prior_university(ctx, session_manager)
    if not prior_uni:
        return False

    if prior_uni in stripped:
        return False

    if not _SCHOOL_HINT.search(stripped):
        return False

    if _BG_IDENTITY_RE.search(stripped):
        _log.info(
            "SESSION_CONTINUITY | identity fresh | prior_uni=%s text=%s",
            prior_uni[:40],
            stripped[:80],
        )
        return True

    if _TARGET_OR_QUESTION_RE.search(stripped):
        return False

    if _PROFILE_FIELD_RE.search(stripped):
        _log.info(
            "SESSION_CONTINUITY | profile-field fresh | prior_uni=%s text=%s",
            prior_uni[:40],
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
    from src.pages.prediction.form_bridge import reset_lead_in_profile

    reset_lead_in_profile(session_manager, ctx, reset_state_machine=False)
    ctx.conversation_turns = []
    ctx.extracted_background = {}
    session_manager.set(**{TURN_KEY: 0, "lead_in_consumed": False})
    if state_machine is not None:
        state_machine.update(conversation_turns=[], turn=0)
    _log.info("SESSION_CONTINUITY | applied fresh reset")
