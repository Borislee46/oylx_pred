from __future__ import annotations

import re
from typing import Any

_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0000fe00-\U0000fe0f"
    "\U0001f1e0-\U0001f1ff"
    "\U0000200d"
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    if not text:
        return ""
    return _EMOJI_RE.sub("", text).strip()


_TABLE_SEP_RE = re.compile(r"^\|?[\s:\-|]+$", re.UNICODE)


def sanitize_feedback(text: str) -> str:
    if not text:
        return ""
    out: list[str] = []
    for raw in strip_emoji(text).splitlines():
        line = raw.strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if _TABLE_SEP_RE.match(line.replace(" ", "")):
            continue
        if line.count("|") >= 2:
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            if len(cells) >= 2:
                if len(cells) % 2 == 0:
                    pairs = [f"{cells[i]}：{cells[i + 1]}" for i in range(0, len(cells), 2)]
                    out.append("；".join(pairs))
                elif len(cells) == 3:
                    out.append(f"{cells[0]}：{cells[1]}（{cells[2]}）")
                else:
                    out.append("、".join(cells))
                continue
        out.append(line)
    cleaned: list[str] = []
    for line in out:
        if line == "" and (not cleaned or cleaned[-1] == ""):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


_FIELD_CHIP_ORDER: tuple[tuple[str, str], ...] = (
    ("university", "院校"),
    ("background_university", "本科院校"),
    ("major", "专业"),
    ("background_major", "本科专业"),
    ("gpa", "GPA"),
    ("language_type", "语言"),
    ("language_score", "成绩"),
    ("country", "地区"),
    ("target_country", "目标地区"),
    ("target_schools", "目标院校"),
    ("target_majors", "目标专业"),
    ("standardized_test_type", "标化"),
    ("standardized_test_score", "分数"),
)

FIELD_LABELS: dict[str, str] = {}
for _k, _lbl in _FIELD_CHIP_ORDER:
    if _lbl not in FIELD_LABELS.values():
        FIELD_LABELS[_k] = _lbl

_PATH_NARRATIVE = {
    "router": "背景较完整 · 快速一次提取",
    "tools": "信息较散 · 逐步对齐填表",
    "fresh": "新学生 · 已清空旧档案",
    "default": "正在读懂你的背景",
}


def path_narrative(kind: str) -> str:
    return _PATH_NARRATIVE.get(kind, _PATH_NARRATIVE["default"])


def format_field_value(key: str, val: Any) -> str:
    if isinstance(val, list):
        return "、".join(str(v) for v in val[:4] if v)
    if isinstance(val, float) and key in ("gpa", "language_score", "standardized_test_score"):
        return f"{val:.1f}"
    return str(val)


def build_field_chips(
    fields: dict[str, Any] | None,
    *,
    low_confidence: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if not fields:
        return []
    low = low_confidence or {}
    seen: set[str] = set()
    chips: list[dict[str, str]] = []
    for key, label in _FIELD_CHIP_ORDER:
        val = fields.get(key)
        if val in (None, "", []) or label in seen:
            continue
        seen.add(label)
        conf = "low" if (key in low or label in low) else "medium"
        chips.append(
            {
                "label": label,
                "value": format_field_value(key, val),
                "confidence": conf,
            }
        )
    return chips
