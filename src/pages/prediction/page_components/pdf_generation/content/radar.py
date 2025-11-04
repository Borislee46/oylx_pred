import matplotlib.pyplot as plt
import numpy as np

MAX_GPA = 4.0
MAX_ACADEMIC_ITEMS = 3
MAX_PRACTICAL_ITEMS = 7

_matplotlib_configured = False


def _configure_matplotlib_once():
    global _matplotlib_configured
    if not _matplotlib_configured:
        try:
            plt.rcParams["font.sans-serif"] = ["SimHei"]
        except Exception:
            plt.rcParams["font.sans-serif"] = ["sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        _matplotlib_configured = True


def create_student_background_radar_chart(input_data, school_level_scores_map):
    _configure_matplotlib_once()

    labels = ["School Tier", "GPA", "Language Score", "Academic", "Practical"]
    num_vars = len(labels)

    school_level_str = input_data.get("school_level", "Unknown")
    raw_school_score = school_level_scores_map.get(
        school_level_str, school_level_scores_map.get("未知", 13)
    )

    valid_scores = [s for s in school_level_scores_map.values() if isinstance(s, (int, float))]
    min_raw_score_val = min(valid_scores) if valid_scores else 1
    max_raw_score_val = max(valid_scores) if valid_scores else 13

    if (max_raw_score_val - min_raw_score_val) != 0:
        val_school = (
            (max_raw_score_val - raw_school_score) / (max_raw_score_val - min_raw_score_val)
        ) * 100
    else:
        val_school = 50
    val_school = max(0, min(val_school, 100))

    gpa = input_data.get("gpa", 0)
    val_gpa = min(gpa / MAX_GPA, 1.0) * 100 if MAX_GPA > 0 else 0
    val_gpa = max(0, min(val_gpa, 100))

    lang_score = input_data.get("language_score", 0)
    val_lang = lang_score * 100
    val_lang = max(0, min(val_lang, 100))

    research = input_data.get("research_count", 0)
    papers = input_data.get("paper_count", 0)
    academic_sum = research + papers
    val_academic = (
        min(academic_sum / MAX_ACADEMIC_ITEMS, 1.0) * 100 if MAX_ACADEMIC_ITEMS > 0 else 0
    )
    val_academic = max(0, min(val_academic, 100))

    internships = input_data.get("internship_count", 0)
    awards = input_data.get("award_count", 0)
    practical_sum = internships + awards
    val_practical = (
        min(practical_sum / MAX_PRACTICAL_ITEMS, 1.0) * 100 if MAX_PRACTICAL_ITEMS > 0 else 0
    )
    val_practical = max(0, min(val_practical, 100))

    values = [val_school, val_gpa, val_lang, val_academic, val_practical]

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    plot_color = "#007ACC"
    fill_color = "#007ACC"

    ax.plot(angles, values, color=plot_color, linewidth=2, label="学生背景")
    ax.fill(angles, values, color=fill_color, alpha=0.2)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontweight="bold")

    for label, angle_deg in zip(ax.get_xticklabels(), np.degrees(angles[:-1])):
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
