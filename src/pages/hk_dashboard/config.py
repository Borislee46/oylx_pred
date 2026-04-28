"""HK Dashboard — column name constants, file paths, and bonus tier tables."""

import os

# ── Data paths ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "hk")

FILES = {
    "kehu_ziyuan": os.path.join(DATA_DIR, "客服资源.csv"),
    "tmk": os.path.join(DATA_DIR, "客服TMK资源处理.csv"),
    "qianyue": os.path.join(DATA_DIR, "客服签约列表.csv"),
    "class_master": os.path.join(DATA_DIR, "班级维表.csv"),
    "roster": os.path.join(DATA_DIR, "班级花名册.csv"),
    "revenue": os.path.join(DATA_DIR, "收入人次.csv"),
    "deferred_revenue": os.path.join(DATA_DIR, "结转收入.csv"),
}

# ── Column name constants ───────────────────────────────────
RESOURCE_ID = "资源id"
WORK_ORDER_ID = "工单id"
CONTRACT_ID = "签约单id"
STUDENT_ID = "学员编号"
CLASS_CODE = "班级编码"
TEACHER_CODE = "教师编码"

# ── Colour palette ──────────────────────────────────────────
PRIMARY = "#008a6c"
ACCENT = "#10b981"
TEXT_DARK = "#1f2937"
TEXT_MID = "#374151"
TEXT_LIGHT = "#4b5563"
BG_LIGHT = "#f8fffe"
PALETTE = ["#008a6c", "#f7ab00", "#73c0de", "#a9a9a9", "#006a60", "#f39800"]

# ── Renewal bonus tiers ─────────────────────────────────────
# (rate_low, rate_high, [(size_low, size_high, hkd_per_hour), ...])
BONUS_TIERS = [
    (0.50, 0.75, [(1, 24, 10), (25, 49, 20), (50, float("inf"), 30)]),
    (0.75, 0.85, [(1, 24, 20), (25, 49, 30), (50, float("inf"), 40)]),
    (0.85, 1.01, [(1, 24, 30), (25, 49, 40), (50, float("inf"), 50)]),
]

def lookup_bonus_rate(renewal_rate: float, student_count: int) -> int:
    for r_low, r_high, brackets in BONUS_TIERS:
        if r_low <= renewal_rate < r_high:
            for s_low, s_high, rate in brackets:
                if s_low <= student_count <= s_high:
                    return rate
    return 0
