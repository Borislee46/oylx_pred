import base64
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.agent.schemas import (
    _TIER_DIFF_SAFETY_BASE,
    _TIER_DIFF_SAFETY_COEFF,
    _TIER_DIFF_TARGET_BASE,
    _TIER_DIFF_TARGET_COEFF,
)
from src.pages.prediction.school_logo_loader import (
    get_logo_path,
)
from src.utils.logger import setup_logger

_logger = setup_logger("page3", "prediction")

PROFILE_FEATURES = [
    "gpa",
    "language_score",
    "research_count",
    "paper_count",
    "internship_count",
    "award_count",
]
FEATURE_LABELS = {
    "gpa": "GPA",
    "language_score": "语言",
    "research_count": "科研",
    "paper_count": "论文",
    "internship_count": "实习",
    "award_count": "获奖",
}

MIN_SAMPLES_FOR_PROFILE = 5

_TIER_COLORS = {"保底": "#22c55e", "目标": "#3b82f6", "冲刺": "#f97316"}

_TIER_LABEL_NORMALISE = {"保底": "保底", "适中": "目标", "冲刺": "冲刺"}

SCHOOL_DIFFICULTY_CACHE: dict[str, float] | None = None


def compute_school_difficulty(cases_df: pd.DataFrame) -> dict[str, float]:
    global SCHOOL_DIFFICULTY_CACHE
    if SCHOOL_DIFFICULTY_CACHE is not None:
        _logger.info(
            "compute_school_difficulty: cache hit (%d schools)", len(SCHOOL_DIFFICULTY_CACHE)
        )
        return SCHOOL_DIFFICULTY_CACHE

    df = cases_df
    if "target_university" not in df.columns or "admitted" not in df.columns:
        _logger.warning("compute_school_difficulty: missing required columns in cases_df")
        SCHOOL_DIFFICULTY_CACHE = {}
        return {}

    admit_rate = df.groupby("target_university")["admitted"].mean()
    avg_gpa_admitted = df[df["admitted"] == 1].groupby("target_university")["gpa"].mean()
    schools = pd.DataFrame({"admit_rate": admit_rate, "avg_gpa_admitted": avg_gpa_admitted})
    schools["d_admit"] = 1.0 - schools["admit_rate"]
    gpa_col = schools["avg_gpa_admitted"]
    schools["d_gpa"] = (
        (gpa_col - gpa_col.min()) / (gpa_col.max() - gpa_col.min())
        if gpa_col.max() > gpa_col.min()
        else 0
    )
    schools["d_raw"] = (schools["d_admit"] + schools["d_gpa"]) / 2
    d_min, d_max = schools["d_raw"].min(), schools["d_raw"].max()
    schools["difficulty"] = (
        (schools["d_raw"] - d_min) / (d_max - d_min) if d_max > d_min else schools["d_raw"]
    )

    SCHOOL_DIFFICULTY_CACHE = schools["difficulty"].to_dict()
    _logger.info(
        "compute_school_difficulty: computed for %d schools (cache miss)",
        len(SCHOOL_DIFFICULTY_CACHE),
    )
    return SCHOOL_DIFFICULTY_CACHE


def get_tier_thresholds(difficulty: float) -> tuple[float, float]:
    safety = _TIER_DIFF_SAFETY_BASE + _TIER_DIFF_SAFETY_COEFF * difficulty
    target = _TIER_DIFF_TARGET_BASE + _TIER_DIFF_TARGET_COEFF * difficulty
    return safety, target


def tier_color_for(label: str) -> str:
    return _TIER_COLORS.get(_TIER_LABEL_NORMALISE.get(label, label), "#64748b")


@lru_cache(maxsize=64)
def logo_b64(school: str) -> str | None:
    p = Path(get_logo_path(school))
    if not p.exists() or p.name == "product_logo.png":
        return None
    return base64.b64encode(p.read_bytes()).decode()


def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("'", "&#39;")
    )
