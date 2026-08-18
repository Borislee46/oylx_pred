import matplotlib.pyplot as plt
import numpy as np

from src.pages.prediction.result_display.radar_scoring import (
    calculate_academic_score,
    calculate_gpa_score,
    calculate_language_score,
    calculate_practical_score,
    calculate_school_score,
)

_matplotlib_configured = False


def _configure_matplotlib_once():
    global _matplotlib_configured
    if not _matplotlib_configured:
        plt.rcParams["font.sans-serif"] = ["SimHei"] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        _matplotlib_configured = True


def create_student_background_radar_chart(input_data):
    _configure_matplotlib_once()

    labels = ["School Tier", "GPA", "Language Score", "Academic", "Practical"]
    num_vars = len(labels)

    bg_uni = str(input_data.get("background_university", ""))
    school_level = None
    if bg_uni:
        try:
            from src.utils.schools.level_service import SchoolLevelService

            info = SchoolLevelService().get_school_info(bg_uni)
            school_level = info.get("school_level", None)
        except Exception:
            pass
    val_school = calculate_school_score(school_level)

    gpa_value = input_data.get("gpa") or input_data.get("gpa_score", 0)
    if isinstance(gpa_value, str) and (gpa_value == "未填写" or not gpa_value.strip()):
        gpa = 0.0
    elif gpa_value is None:
        gpa = 0.0
    else:
        gpa = float(gpa_value)
    val_gpa = calculate_gpa_score(gpa) if gpa > 0 else 0.0

    lang_raw = input_data.get("language_score_raw")
    lang_norm = input_data.get("language_score")
    if lang_raw not in (None, "", "未填写"):
        try:
            lang_score = float(lang_raw)
        except (ValueError, TypeError):
            lang_score = 0.0
    elif lang_norm is None:
        lang_score = 0.0
    elif isinstance(lang_norm, str) and (lang_norm == "未填写" or not lang_norm.strip()):
        lang_score = 0.0
    else:
        try:
            lang_score = float(lang_norm)
        except (ValueError, TypeError):
            lang_score = 0.0
    val_lang = calculate_language_score(lang_score) if lang_score > 0 else 0.0

    research = int(input_data.get("research_count", 0) or 0)
    papers = int(input_data.get("paper_count", 0) or 0)
    val_academic = calculate_academic_score(research, papers)

    internships = int(input_data.get("internship_count", 0) or 0)
    awards = int(input_data.get("award_count", 0) or 0)
    val_practical = calculate_practical_score(internships, awards)

    values = [val_school, val_gpa, val_lang, val_academic, val_practical]

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})

    plot_color = "#0F2A43"
    fill_color = "#B6862C"

    ax.plot(angles, values, color=plot_color, linewidth=2, label="学生背景")
    ax.fill(angles, values, color=fill_color, alpha=0.22)
    ax.scatter(angles, values, color=plot_color, s=18, zorder=5)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontweight="bold")

    for label, angle_deg in zip(ax.get_xticklabels(), np.degrees(angles[:-1]), strict=True):
        angle_rad = np.deg2rad(angle_deg)
        if angle_rad == 0 or angle_rad == np.pi:
            label.set_horizontalalignment("center")
        elif 0 < angle_rad < np.pi:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("right")

    ax.set_ylim(0, 100)
    ax.set_rlabel_position(180 / num_vars)

    ax.tick_params(colors="#333333")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(color="#CCCCCC")
    ax.spines["polar"].set_color("#555555")

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    return fig
