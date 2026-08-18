import json
import logging
from typing import Any

from src.adjustment.config import (
    FACULTY_PENALTY_HEAVY,
    FACULTY_PENALTY_LIGHT,
    FACULTY_PENALTY_MEDIUM,
    PROJECT_ROOT,
)

_logger = logging.getLogger(__name__)

_DEFAULT_CROSS_FACULTY_RULES: dict[str, set[str]] = {
    "文学院": {"文学院", "社会科学院", "教育学院", "商学院", "艺术学院"},
    "社会科学院": {
        "社会科学院",
        "文学院",
        "商学院",
        "教育学院",
        "艺术学院",
        "建筑学院",
    },
    "法学院": {"法学院"},
    "教育学院": {"教育学院", "文学院", "社会科学院"},
    "商学院": {"商学院", "社会科学院", "文学院"},
    "理学院": {
        "理学院",
        "工程学院",
        "商学院",
        "经济金融学院",
        "科学学院",
        "计算机学院",
    },
    "工程学院": {
        "工程学院",
        "理学院",
        "商学院",
        "计算机学院",
        "建筑学院",
        "设计学院",
        "科学学院",
    },
    "计算机学院": {"计算机学院", "工程学院", "理学院", "商学院"},
    "艺术学院": {"艺术学院", "社会科学院", "文学院", "设计学院", "建筑学院"},
    "医学院": {"医学院"},
    "建筑学院": {"建筑学院", "工程学院", "设计学院", "艺术学院"},
    "设计学院": {"设计学院", "艺术学院", "建筑学院", "社会科学院"},
}

# Cross-faculty severity for pairs NOT in the whitelist above.
# Only light and medium pairs are listed; unlisted out-of-scope pairs default to heavy (×0.30).
# Design rationale: faculty distance is not binary — some cross-faculty moves have
# genuine methodological bridges (quantitative methods, applied paths) and should not
# be penalized as harshly as fundamental domain switches.
_DEFAULT_CROSS_FACULTY_SEVERITY: dict[tuple[str, str], str] = {
    # ── Light (×0.70): quantitative/methodological bridges ──
    ("理学院", "社会科学院"): "light",
    ("理学院", "医学院"): "light",
    ("理学院", "教育学院"): "light",
    ("工程学院", "经济金融学院"): "light",
    ("工程学院", "医学院"): "light",
    ("计算机学院", "经济金融学院"): "light",
    ("计算机学院", "社会科学院"): "light",
    ("计算机学院", "设计学院"): "light",
    ("计算机学院", "医学院"): "light",
    ("计算机学院", "教育学院"): "light",
    ("计算机学院", "艺术学院"): "light",
    ("社会科学院", "商学院"): "light",
    ("社会科学院", "法学院"): "light",
    ("商学院", "经济金融学院"): "light",
    ("商学院", "计算机学院"): "light",
    ("商学院", "社会科学院"): "light",
    ("商学院", "教育学院"): "light",
    ("艺术学院", "商学院"): "light",
    ("医学院", "理学院"): "light",
    ("医学院", "工程学院"): "light",
    ("医学院", "计算机学院"): "light",
    ("教育学院", "商学院"): "light",
    ("教育学院", "计算机学院"): "light",
    ("教育学院", "社会科学院"): "light",
    ("设计学院", "计算机学院"): "light",
    ("设计学院", "商学院"): "light",
    ("建筑学院", "计算机学院"): "light",
    # ── Medium (×0.50): partial overlap, applied path possible ──
    ("理学院", "法学院"): "medium",
    ("理学院", "经济金融学院"): "medium",
    ("理学院", "建筑学院"): "medium",
    ("工程学院", "社会科学院"): "medium",
    ("工程学院", "教育学院"): "medium",
    ("工程学院", "艺术学院"): "medium",
    ("工程学院", "设计学院"): "medium",
    ("计算机学院", "建筑学院"): "medium",
    ("社会科学院", "医学院"): "medium",
    ("社会科学院", "计算机学院"): "medium",
    ("商学院", "法学院"): "medium",
    ("商学院", "医学院"): "medium",
    ("文学院", "法学院"): "medium",
    ("文学院", "计算机学院"): "medium",
    ("文学院", "社会科学院"): "medium",
    ("文学院", "商学院"): "medium",
    ("艺术学院", "计算机学院"): "medium",
    ("艺术学院", "社会科学院"): "medium",
    ("艺术学院", "教育学院"): "medium",
    ("法学院", "商学院"): "medium",
    ("法学院", "社会科学院"): "medium",
    ("医学院", "社会科学院"): "medium",
    ("医学院", "教育学院"): "medium",
    ("建筑学院", "商学院"): "medium",
    ("设计学院", "社会科学院"): "medium",
    ("设计学院", "教育学院"): "medium",
}


def _load_cross_faculty_rules() -> tuple[dict[str, set[str]], dict[tuple[str, str], str]]:
    json_path = PROJECT_ROOT / "config" / "cross_faculty_rules.json"
    if not json_path.exists():
        _logger.info("cross_faculty_rules.json not found, using hardcoded defaults")
        return dict(_DEFAULT_CROSS_FACULTY_RULES), dict(_DEFAULT_CROSS_FACULTY_SEVERITY)

    try:
        cfg = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        _logger.warning("Failed to parse cross_faculty_rules.json: %s, using hardcoded defaults", e)
        return dict(_DEFAULT_CROSS_FACULTY_RULES), dict(_DEFAULT_CROSS_FACULTY_SEVERITY)

    whitelist: dict[str, set[str]] = {}
    wl = cfg.get("whitelist", {})
    if isinstance(wl, dict):
        for faculty, allowed in wl.items():
            if isinstance(faculty, str) and not faculty.startswith("_"):
                if isinstance(allowed, list):
                    whitelist[faculty] = {str(x) for x in allowed if isinstance(x, str)}
        if whitelist:
            _logger.info("Loaded %d faculty whitelist entries from JSON", len(whitelist))

    severity: dict[tuple[str, str], str] = {}
    sev = cfg.get("severity", {})
    if isinstance(sev, dict):
        for level in ("light", "medium"):
            level_cfg = sev.get(level, {})
            if isinstance(level_cfg, dict):
                pairs = level_cfg.get("pairs", [])
                if isinstance(pairs, list):
                    for pair in pairs:
                        if isinstance(pair, list) and len(pair) == 2:
                            bg, tg = str(pair[0]), str(pair[1])
                            if bg and tg:
                                severity[(bg, tg)] = level
        if severity:
            _logger.info(
                "Loaded %d cross-faculty severity entries from JSON (%d light, %d medium)",
                len(severity),
                sum(1 for v in severity.values() if v == "light"),
                sum(1 for v in severity.values() if v == "medium"),
            )

    if not whitelist:
        _logger.warning("JSON whitelist empty, using hardcoded defaults")
        whitelist = dict(_DEFAULT_CROSS_FACULTY_RULES)
    if not severity:
        _logger.warning("JSON severity empty, using hardcoded defaults")
        severity = dict(_DEFAULT_CROSS_FACULTY_SEVERITY)

    return whitelist, severity


CROSS_FACULTY_RULES, CROSS_FACULTY_SEVERITY = _load_cross_faculty_rules()


def get_cross_faculty_penalty_factor(
    background_faculty: str | None,
    target_faculty: str | None,
) -> float:
    if not background_faculty or not target_faculty:
        return 1.0

    bg = background_faculty.strip()
    tg = target_faculty.strip()

    if not bg or not tg:
        return 1.0

    allowed = get_allowed_target_faculties(bg)
    if tg in allowed:
        return 1.0

    severity = CROSS_FACULTY_SEVERITY.get((bg, tg))
    if severity == "light":
        return FACULTY_PENALTY_LIGHT
    elif severity == "medium":
        return FACULTY_PENALTY_MEDIUM
    else:
        return FACULTY_PENALTY_HEAVY


def get_allowed_target_faculties(background_faculty: str | None) -> set[str]:
    if not background_faculty:
        return set()
    return CROSS_FACULTY_RULES.get(background_faculty, set())


def filter_schools_by_allowed_faculties(
    schools: list[dict[str, Any]], allowed_faculties: set[str]
) -> list[dict[str, Any]]:
    if not schools or not allowed_faculties:
        return schools
    return [
        school
        for school in schools
        if not (faculty := school.get("faculty", "").strip()) or faculty in allowed_faculties
    ]


def filter_schools_by_faculty_rules(
    schools: list[dict[str, Any]],
    background_faculty: str | None,
) -> list[dict[str, Any]]:
    if not schools or not background_faculty:
        return schools

    allowed_faculties = get_allowed_target_faculties(background_faculty)
    if not allowed_faculties:
        return schools

    return filter_schools_by_allowed_faculties(schools, allowed_faculties)


def is_faculty_out_of_scope(background_faculty: str | None, target_faculty: str | None) -> bool:
    if not background_faculty or not target_faculty:
        return False
    allowed = get_allowed_target_faculties(background_faculty)
    if not allowed:
        return False
    return target_faculty.strip() not in allowed
