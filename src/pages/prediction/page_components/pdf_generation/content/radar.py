# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
import matplotlib.pyplot as plt
import numpy as np

from src.pages.prediction.result_modifier.config import (
    GPA_MINIMUM,
    GPA_PENALTY_MAX_COEFFICIENT,
    GPA_PENALTY_QUADRATIC_COEFFICIENT,
    GPA_PENALTY_SEVERE_THRESHOLD,
    LANGUAGE_MINIMUM,
    LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER,
    LANGUAGE_PENALTY_LEVEL_1_THRESHOLD,
    LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER,
    LANGUAGE_PENALTY_LEVEL_2_THRESHOLD,
    LANGUAGE_PENALTY_LEVEL_3_THRESHOLD,
    LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER,
    LANGUAGE_PENALTY_SEVERE_THRESHOLD,
)

MAX_GPA = 4.0
MAX_ACADEMIC_ITEMS = 3
MAX_PRACTICAL_ITEMS = 7

DEFAULT_GPA_MEAN = 3.0
DEFAULT_GPA_STD = 0.5
DEFAULT_LANGUAGE_MEAN = 0.75
DEFAULT_LANGUAGE_STD = 0.15

_matplotlib_configured = False


def _calculate_gpa_score(
    gpa: float,
    gpa_mean: float = DEFAULT_GPA_MEAN,
    gpa_std: float = DEFAULT_GPA_STD,
) -> float:
    if gpa < GPA_MINIMUM:
        penalty = GPA_PENALTY_SEVERE_THRESHOLD
        return max(0, min(100, (1 - penalty) * 100))

    base_score = (gpa / MAX_GPA) * 100

    if gpa >= gpa_mean:
        return min(100, base_score)

    gpa_gap = (gpa_mean - gpa) / gpa_std if gpa_std > 0 else 0
    penalty = min(
        GPA_PENALTY_MAX_COEFFICIENT,
        GPA_PENALTY_QUADRATIC_COEFFICIENT * gpa_gap**2,
    )

    adjusted_score = base_score * (1 - penalty)
    return max(0, min(100, adjusted_score))


def _calculate_language_score(
    language_score: float,
    language_mean: float = DEFAULT_LANGUAGE_MEAN,
    language_std: float = DEFAULT_LANGUAGE_STD,
) -> float:
    if language_score < LANGUAGE_MINIMUM:
        penalty = LANGUAGE_PENALTY_SEVERE_THRESHOLD
        return max(0, min(100, (1 - penalty) * 100))

    base_score = language_score * 100

    language_excellent_line = language_mean + 0.5 * language_std
    language_pass_line = language_mean - LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER * language_std

    if language_score >= language_excellent_line:
        return min(100, base_score)
    elif language_score >= language_pass_line:
        gap_ratio = (
            (language_excellent_line - language_score)
            / (language_excellent_line - language_pass_line)
            if (language_excellent_line - language_pass_line) > 0
            else 0
        )
        reduction = LANGUAGE_PENALTY_LEVEL_3_THRESHOLD * gap_ratio * 0.15
        return max(0, min(100, base_score * (1 - reduction)))
    elif language_score < (language_pass_line - LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER * language_std):
        penalty = LANGUAGE_PENALTY_LEVEL_1_THRESHOLD
    elif language_score < (language_pass_line - LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER * language_std):
        penalty = LANGUAGE_PENALTY_LEVEL_2_THRESHOLD
    else:
        penalty = LANGUAGE_PENALTY_LEVEL_3_THRESHOLD

    return max(0, min(100, base_score * (1 - penalty)))


def _configure_matplotlib_once():
    global _matplotlib_configured
    if not _matplotlib_configured:
        plt.rcParams["font.sans-serif"] = ["STSong-Light"] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        _matplotlib_configured = True


def create_student_background_radar_chart(input_data, school_level_scores_map):
    _configure_matplotlib_once()

    labels = ["School Tier", "GPA", "Language Score", "Academic", "Practical"]
    num_vars = len(labels)

    school_level_str = input_data.get("school_level", "未知")
    raw_school_score = school_level_scores_map.get(
        school_level_str, school_level_scores_map.get("未知", 11)
    )

    valid_scores = [s for s in school_level_scores_map.values() if isinstance(s, (int, float))]
    min_raw_score_val = min(valid_scores) if valid_scores else 1
    max_raw_score_val = max(valid_scores) if valid_scores else 11

    if (max_raw_score_val - min_raw_score_val) != 0:
        val_school = (
            (max_raw_score_val - raw_school_score) / (max_raw_score_val - min_raw_score_val)
        ) * 100
    else:
        val_school = 50
    val_school = max(0, min(val_school, 100))

    gpa_value = input_data.get("gpa") or input_data.get("gpa_score", 0)
    if isinstance(gpa_value, str) and (gpa_value == "未填写" or not gpa_value.strip()):
        gpa = 0
    elif gpa_value is None:
        gpa = 0
    else:
        gpa = float(gpa_value)
    val_gpa = _calculate_gpa_score(gpa) if gpa > 0 else 0

    lang_score_value = input_data.get("language_score")
    if lang_score_value is None:
        lang_score = 0
    elif isinstance(lang_score_value, str) and (
        lang_score_value == "未填写" or not lang_score_value.strip()
    ):
        lang_score = 0
    else:
        lang_score = float(lang_score_value)
    val_lang = _calculate_language_score(lang_score) if lang_score > 0 else 0

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

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})

    plot_color = "#007ACC"
    fill_color = "#007ACC"

    ax.plot(angles, values, color=plot_color, linewidth=2, label="学生背景")
    ax.fill(angles, values, color=fill_color, alpha=0.2)

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
