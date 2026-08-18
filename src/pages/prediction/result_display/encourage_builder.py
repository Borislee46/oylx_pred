from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "sales_encourage.json"

_FALLBACK_CONFIG: dict[str, Any] = {
    "language_tiers": {
        "IELTS": {"aliases": ["IELTS", "雅思"], "strong": 7.0, "label": "雅思 {score}"},
        "TOEFL": {"aliases": ["TOEFL", "托福"], "strong": 100, "label": "托福 {score}"},
        "CET-6": {"aliases": ["CET-6", "CET6", "六级"], "strong": 500, "label": "六级 {score}"},
    },
    "gpa_tiers": {"strong_gpa": 3.5, "strong_gpa_raw": 85, "mid_gpa": 3.0, "mid_gpa_raw": 80},
    "penalty_aliases": {
        "cross_major": ["penalty_cross_major", "Cross Major Penalty"],
        "gpa": ["penalty_gpa", "GPA 惩罚"],
        "language": ["penalty_language", "Language Penalty"],
    },
    "rules": {
        "strong_elite": [
            {
                "when": {"gpa_strong": True, "lang_strong": True},
                "lines": [
                    "GPA {gpa} 搭配 {lang_label}，硬实力处于申请池头部",
                    "GPA 与语言双线突出，港三新二值得重点冲刺",
                ],
            },
            {
                "when": {"gpa_strong": True},
                "lines": [
                    "GPA {gpa} 是你最强的筹码，以下是择校版图",
                    "学术底子过硬，择校向上空间充裕",
                ],
            },
            {
                "when": {},
                "lines": [
                    "你的背景在申请池处于前列，以下是择校版图",
                    "竞争力突出，冲刺区值得重点关注",
                ],
            },
        ],
        "cross_major": [
            {
                "when": {"gpa_strong": True},
                "lines": [
                    "跨专业申请，但你 GPA {gpa} 的学术底子是最大筹码",
                    "跨专业不是劣势——你的学科交叉背景恰是差异化优势",
                    "换个赛道，关键在用学术底子撑起新方向的说服力",
                ],
            },
            {
                "when": {"gpa_mid": True},
                "lines": [
                    "跨专业申请策略先行，以下是差异化择校分析",
                    "跨专业需要策略性选校，学科交叉点就是你的优势区",
                    "不是从零开始——可迁移的学术能力是新方向的支点",
                ],
            },
            {
                "when": {},
                "lines": [
                    "跨专业申请需策略先行，选对学科交叉点是关键",
                    "跨专业+学术待补强，但选校策略可以放大你的迁移优势",
                ],
            },
        ],
        "weak_gaps": [
            {
                "when": {"has_lang": True, "has_gpa": True},
                "lines": [
                    "{lang_label} 与 GPA 均有提升空间，以下是精准补强方向",
                    "语言与学术双线待提升，但每个短板都有可操作的补救路径",
                ],
            },
            {
                "when": {"has_lang": True},
                "lines": [
                    "{lang_label} 是当前最值得投入的突破点",
                    "语言关是性价比最高的提升杠杆，以下是补强路线",
                ],
            },
            {
                "when": {"has_gpa": True},
                "lines": [
                    "GPA {gpa} 存在提升空间，以下是可操作的补救方向",
                    "学术成绩是重点补强项，策略性择校可弥补硬指标短板",
                ],
            },
            {
                "when": {},
                "lines": [
                    "你的背景存在可提升空间，方向已标出",
                    "短板已识别，每个缺口都有对应补救路径",
                    "认清差距是第一步，以下是可操作的提升方向",
                ],
            },
        ],
        "medium_mixed": [
            {
                "when": {"gpa_mid": True, "has_gpa": False},
                "lines": ["GPA {gpa} 是你的基本盘，选校矩阵已展开"],
            },
            {
                "when": {"has_lang": True},
                "lines": [
                    "基础扎实，{lang_label} 再进一步就是质的飞跃",
                    "底子不错，语言稍加打磨即可打开更多选项",
                ],
            },
            {"when": {"has_gpa": True}, "lines": ["底子扎实，GPA 再往上走一步，选择面会宽很多"]},
            {
                "when": {},
                "lines": [
                    "你的背景基础扎实，选校梯度已为你展开",
                    "学术底子扎实，冲稳保三层已分层",
                    "中坚区间的竞争力，细节处见提升空间",
                ],
            },
        ],
    },
    "fallback_line": "以下是为你定制的择校版图",
}


def _load_config() -> dict[str, Any]:
    try:
        if _CONFIG_PATH.exists():
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            _logger.info("sales_encourage config loaded from %s", _CONFIG_PATH)
            return data
    except Exception:
        _logger.warning(
            "Failed to load sales_encourage.json, using in-module defaults", exc_info=True
        )
    return _FALLBACK_CONFIG


_CFG = _load_config()


@dataclass
class EncourageSignals:
    gpa: float
    gpa_strong: bool
    gpa_mid: bool
    lang_strong: bool
    lang_label: str
    has_gpa: bool
    has_lang: bool

    def template_ctx(self) -> dict[str, Any]:
        gpa_str = f"{self.gpa:.2f}" if self.gpa else "0.00"
        return {"gpa": gpa_str, "lang_label": self.lang_label}


def _classify_language(lang_type: str, lang_score: float, cfg: dict) -> tuple[bool, str]:
    tiers = cfg.get("language_tiers", {})
    for spec in tiers.values():
        aliases = [str(a).upper() for a in spec.get("aliases", [])]
        if lang_type in aliases:
            strong = lang_score >= float(spec.get("strong", 0))
            label = spec.get("label", "").format(score=lang_score) if lang_score > 0 else ""
            return strong, label
    if lang_score > 0:
        return False, f"语言 {lang_score}"
    return False, ""


def _collect_active_penalties(all_candidates: list[dict]) -> set[str]:
    active: set[str] = set()
    for r in (all_candidates or [])[:5]:
        trace = r.get("_adjustment_trace")
        if isinstance(trace, dict):
            for k, v in trace.items():
                if k.startswith("penalty_") and isinstance(v, (int, float)) and v < 0:
                    active.add(k)
        elif isinstance(trace, list):
            for item in trace:
                if isinstance(item, dict) and item.get("factor_type") == "penalty":
                    active.add(item.get("name", ""))
    return active


def _extract_signals(input_data: dict, all_candidates: list[dict], cfg: dict) -> EncourageSignals:
    gpa = float(input_data.get("gpa", 0) or 0)
    gpa_raw = float(input_data.get("gpa_raw", 0) or 0)
    lang_score = float(
        input_data.get("language_score_raw") or input_data.get("language_score") or 0
    )
    lang_type = str(input_data.get("language_type", "")).upper()

    gpa_cfg = cfg.get("gpa_tiers", {})
    gpa_strong = gpa >= float(gpa_cfg.get("strong_gpa", 3.5)) or gpa_raw >= float(
        gpa_cfg.get("strong_gpa_raw", 85)
    )
    gpa_mid = gpa >= float(gpa_cfg.get("mid_gpa", 3.0)) or gpa_raw >= float(
        gpa_cfg.get("mid_gpa_raw", 80)
    )

    lang_strong, lang_label = _classify_language(lang_type, lang_score, cfg)

    active = _collect_active_penalties(all_candidates)
    aliases = cfg.get("penalty_aliases", {})

    def _has(logical: str) -> bool:
        return any(p in active for p in aliases.get(logical, []))

    return EncourageSignals(
        gpa=gpa,
        gpa_strong=gpa_strong,
        gpa_mid=gpa_mid,
        lang_strong=lang_strong,
        lang_label=lang_label,
        has_gpa=_has("gpa"),
        has_lang=_has("language"),
    )


def _match_rule(when: dict[str, Any], signals: EncourageSignals) -> bool:
    for key, expected in when.items():
        if getattr(signals, key, None) != expected:
            return False
    return True


def build_encourage(input_data: dict, profile: str, all_candidates: list[dict]) -> str:
    signals = _extract_signals(input_data, all_candidates, _CFG)
    rules = _CFG.get("rules", {}).get(profile) or []
    ctx = signals.template_ctx()
    for rule in rules:
        if _match_rule(rule.get("when") or {}, signals):
            lines = rule.get("lines") or []
            if lines:
                return lines[0].format(**ctx)
    return _CFG.get("fallback_line", "以下是为你定制的择校版图")


__all__ = ["EncourageSignals", "build_encourage"]
