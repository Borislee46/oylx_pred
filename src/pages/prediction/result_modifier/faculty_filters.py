# =============================================================================
# 学部过滤与跨学部惩罚 (Faculty Filters)
# ─────────────────────────────────────────────────────────────────────────────
# 核心思想：录取不是"全校统一分数线"，而是"按学部/学院独立竞争"。
# 理学院的录取评估委员会 ≠ 法学院的委员会，专业背景要求完全不同。
#
# 为什么硬编码规则而不是从数据学习？
# 1. 数据稀疏：跨学部录取案例极少（<1%），机器学习无法学到可靠模式。
#    一个 year-cohort 可能只有 2 个社科→法律的录取案例，不足以建模。
# 2. Domain knowledge 可靠：学部归属关系稳定且明确。
#    理学院和工程学院的交叉是公认的（数学基础共享），
#    法学院和医学院各自独立也是公认的。
# 3. 可审核与可调整：顾问可以对规则提出异议（如"教育学院的哪里属于社科？"），
#    然后直接修改字典。黑盒模型参数无法如此讨论。
#
# 何时从规则切换到 ML？
#   当跨学部录取案例积累到足够量（估计 >200 cross-faculty cases per rule），
#   可以训练一个轻量分类器判断"跨学部是否可接受"。
#   但目前（~1000 cases total）远不够。
#
# 规则来源：香港/新加坡/澳门/马来西亚高校的典型学部划分 + 顾问反馈迭代
# =============================================================================

from typing import Any

from src.pages.prediction.result_modifier.config import FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR
from src.pages.prediction.result_modifier.utils import clip_probability

CROSS_FACULTY_RULES: dict[str, set[str]] = {
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


def get_allowed_target_faculties_from_background_faculties(
    background_faculties: list[str] | None, max_allowed: int = 6
) -> set[str]:
    if not background_faculties:
        return set()

    allowed: set[str] = set()
    for bg in background_faculties:
        if not bg:
            continue
        allowed |= CROSS_FACULTY_RULES.get(bg, {bg})
        if max_allowed > 0 and len(allowed) >= max_allowed:
            break

    if max_allowed > 0 and len(allowed) > max_allowed:
        return set(list(allowed)[:max_allowed])
    return allowed


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


# 跨学部外范围判断：目标学部不在背景学部的允许列表中 → 超出范围。
# 被 adjustment_pipeline.py 的 Layer 3 调用，触发 ×0.3 的严重惩罚。
# 返回值是真/假（是否超范围），具体惩罚力度由调用方控制，
# 保持职责单一：这里只做"是否合理"的判断，不混杂"惩罚多少"。
def is_faculty_out_of_scope(background_faculty: str | None, target_faculty: str | None) -> bool:
    if not background_faculty or not target_faculty:
        return False
    allowed = get_allowed_target_faculties(background_faculty)
    if not allowed:
        return False
    return target_faculty.strip() not in allowed


def apply_out_of_scope_faculty_penalty(
    schools: list[dict[str, Any]],
    background_faculty: str | None,
    factor: float = FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR,
) -> list[dict[str, Any]]:
    if not schools or not background_faculty:
        return schools

    allowed_faculties = get_allowed_target_faculties(background_faculty)
    if not allowed_faculties:
        return schools

    adjusted: list[dict[str, Any]] = []
    for s in schools:
        if not isinstance(s, dict):
            continue
        faculty = str(s.get("faculty", "")).strip()
        if faculty and faculty not in allowed_faculties:
            prob = s.get("probability", 0.0)
            adjusted_prob = clip_probability(prob) * factor
            s = s.copy()
            s["probability"] = clip_probability(adjusted_prob)
        adjusted.append(s)
    return adjusted
