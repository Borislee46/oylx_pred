from src.pages.prediction.core.utils import denormalize_language_score
from src.pages.prediction.handler_config import DEFAULT_SESSION_KEYS

_COUNT_LABELS = (
    ("internship_count", "实习"),
    ("paper_count", "论文"),
    ("research_count", "科研"),
    ("award_count", "奖项"),
)


def _fmt_gpa(val: float) -> str:
    if val >= 10:
        return f"{val:.1f}"
    return f"{val:.2f}"


def build_background_summary(session_manager) -> str:
    data = session_manager.get(DEFAULT_SESSION_KEYS.input_data) or {}
    parts: list[str] = []

    uni = str(data.get("background_university", "") or "")
    major = str(data.get("background_major_original") or data.get("background_major", "") or "")
    if uni or major:
        major_display = major
        if data.get("is_dual_degree") and data.get("background_major_2"):
            major2 = str(
                data.get("background_major_2_original") or data.get("background_major_2", "")
            )
            degree_type = str(data.get("degree_type", "辅修") or "辅修")
            if major2:
                major_display = f"{major} + {major2}({degree_type})"
        parts.append(f"{uni} · {major_display}".strip(" ·"))

    gpa_raw = data.get("gpa_raw")
    if gpa_raw is None:
        gpa_raw = data.get("gpa")
    if gpa_raw is not None and float(gpa_raw) > 0:
        parts.append(f"GPA {_fmt_gpa(float(gpa_raw))}")

    lang_type = str(data.get("language_type", "") or "")
    lang_raw = data.get("language_score_raw")
    if (lang_raw is None or lang_raw == 0) and lang_type:
        lang_norm = data.get("language_score")
        if lang_norm is not None and float(lang_norm) > 0:
            lang_raw = denormalize_language_score(float(lang_norm), lang_type, round_to_half=True)
    if lang_type and lang_raw is not None and float(lang_raw) > 0:
        parts.append(f"{lang_type} {lang_raw}")

    exam_type = str(data.get("exam_type", "") or "").strip()
    exam_score = data.get("exam_score")
    if exam_type and exam_type != "无" and exam_score is not None and float(exam_score) > 0:
        parts.append(f"{exam_type} {int(float(exam_score))}")

    count_items: list[str] = []
    for key, label in _COUNT_LABELS:
        val = data.get(key)
        try:
            n = int(float(val)) if val is not None else 0
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            count_items.append(f"{n}{label}")
    if count_items:
        parts.append(" · ".join(count_items))

    return "  |  ".join(p for p in parts if p) or "已提交背景信息"
