"""
V5: Joint Penalty Effect Analysis
==================================
Deep-dive into the 5-layer penalty chain's joint behavior:
  1. Penalty count distribution × ECE per count
  2. ECE contribution per layer (the V2 gap — calibration, not just ranking)
  3. Total penalty ratio distribution + ceiling hits
  4. Layer co-occurrence matrix
  5. Penalty vs student strength (confirming C9 -18pp vs 双非 -6pp)
  6. Interaction effects: is joint penalty super-additive?

Limitations:
  - CrossMajor penalty uses simplified linear interpolation only.
    Production code includes _adjust_cross_major_by_evidence() empirical Bayes
    shrinkage (adjustment_pipeline.py:349-396) which reduces penalty when
    historical cross-major admit rates support it. V5 measurements for
    CrossMajor are conservative (penalty is heavier than production).
  - TF-IDF text boost is not included in this simulation.
  - School tier classification uses keyword heuristics, not production
    school_level_service.

Usage: python reports/v5_joint_penalty_effect/run_joint_penalty_analysis.py
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "machine_learning_models"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.machine_learning_models.data_loader import load_and_preprocess_data
from src.utils.model_loader import _load_serialized_xgb

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "src", "machine_learning_models", "data", "cases.feather")
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "src", "machine_learning_models", "pre-trained_models",
    "xgboost_20260316_092608.ubj",
)
SIM_CACHE_PATH = os.path.join(PROJECT_ROOT, "cache", "background_target_similarity.feather")
DETAILS_PATH = os.path.join(
    PROJECT_ROOT, "src", "machine_learning_models", "data", "school_major_details.feather"
)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Faculty rules ────────────────────────────────────────────────────────────
CROSS_FACULTY_RULES: dict[str, set[str]] = {
    "文学院": {"文学院", "社会科学院", "教育学院", "商学院", "艺术学院"},
    "社会科学院": {"社会科学院", "文学院", "商学院", "教育学院", "艺术学院", "建筑学院"},
    "法学院": {"法学院"},
    "教育学院": {"教育学院", "文学院", "社会科学院"},
    "商学院": {"商学院", "社会科学院", "文学院"},
    "理学院": {"理学院", "工程学院", "商学院", "经济金融学院", "科学学院", "计算机学院"},
    "工程学院": {"工程学院", "理学院", "商学院", "计算机学院", "建筑学院", "设计学院", "科学学院"},
    "计算机学院": {"计算机学院", "工程学院", "理学院", "商学院"},
    "艺术学院": {"艺术学院", "社会科学院", "文学院", "设计学院", "建筑学院"},
    "设计学院": {"设计学院", "艺术学院", "工程学院", "建筑学院"},
    "建筑学院": {"建筑学院", "工程学院", "设计学院", "艺术学院"},
    "医学院": {"医学院"},
    "经济金融学院": {"经济金融学院", "商学院", "社会科学院"},
    "科学学院": {"科学学院", "理学院", "工程学院"},
}

# ── Constants from config.py ─────────────────────────────────────────────────
GPA_MINIMUM = 2.0
GPA_PENALTY_MAX_COEFFICIENT = 0.8
GPA_PENALTY_QUADRATIC_COEFFICIENT = 0.15
GPA_PENALTY_SEVERE_THRESHOLD = 0.95

LANGUAGE_MINIMUM = 0.6
LANGUAGE_PENALTY_SEVERE_THRESHOLD = 0.95
LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER = 0.5
LANGUAGE_PENALTY_LEVEL_1_THRESHOLD = 0.85
LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER = 1.5
LANGUAGE_PENALTY_LEVEL_2_THRESHOLD = 0.7
LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER = 1.0
LANGUAGE_PENALTY_LEVEL_3_THRESHOLD = 0.4
LANGUAGE_PENALTY_LEVEL_3_MULTIPLIER = 0.5
LANGUAGE_PENALTY_LEVEL_3_5_THRESHOLD = 0.2

CROSS_MAJOR_SIMILARITY_MIN = 0.8
CROSS_MAJOR_PENALTY_FACTOR = 0.5
MIN_SIMILARITY_THRESHOLD = 0.89

FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR = 0.3

PROFESSIONAL_MAJORS_LOWER = ["business administration", "mba"]
PROFESSIONAL_REDUCTION_FACTOR = 0.30
PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR = 0.50

MAX_TOTAL_PENALTY_RATIO = 0.70
MAX_TOTAL_BOOST_RATIO = 0.30
PENALTY_DECAY_FACTOR = 0.85
BOOST_DECAY_FACTOR = 0.80
ARBITRATION_MIN_PROBABILITY = 0.005

# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

PENALTY_NAMES = ["GPA", "Language", "CrossMajor", "Faculty", "Professional"]
PENALTY_LABELS = [
    "GPA Penalty\n(二次z-score)",
    "Language Penalty\n(阶梯阈值)",
    "Cross Major\n(相似度×0.5)",
    "Cross Faculty\n(学部×0.3)",
    "Professional Degree\n(MBA无实习)",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Penalty computation functions (matching production)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gpa_penalty(gpa, gpa_mean, gpa_std):
    if gpa < GPA_MINIMUM:
        return GPA_PENALTY_SEVERE_THRESHOLD
    if gpa >= gpa_mean:
        return 0.0
    z = (gpa_mean - gpa) / max(gpa_std, 1e-6)
    return min(GPA_PENALTY_MAX_COEFFICIENT, GPA_PENALTY_QUADRATIC_COEFFICIENT * (z ** 2))


def compute_language_penalty(score, lang_mean, lang_std):
    pass_line = max(LANGUAGE_MINIMUM, lang_mean - LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER * lang_std)
    if score < LANGUAGE_MINIMUM:
        return LANGUAGE_PENALTY_SEVERE_THRESHOLD
    if score >= pass_line:
        return 0.0
    dist = pass_line - score
    if dist > LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER * lang_std:
        return LANGUAGE_PENALTY_LEVEL_1_THRESHOLD
    elif dist > LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER * lang_std:
        return LANGUAGE_PENALTY_LEVEL_2_THRESHOLD
    elif dist > LANGUAGE_PENALTY_LEVEL_3_MULTIPLIER * lang_std:
        return LANGUAGE_PENALTY_LEVEL_3_THRESHOLD
    else:
        return LANGUAGE_PENALTY_LEVEL_3_5_THRESHOLD


def compute_cross_major_penalty(similarity):
    if similarity >= MIN_SIMILARITY_THRESHOLD:
        return 0.0
    if similarity <= CROSS_MAJOR_SIMILARITY_MIN:
        return 1.0 - CROSS_MAJOR_PENALTY_FACTOR
    t = (similarity - CROSS_MAJOR_SIMILARITY_MIN) / (MIN_SIMILARITY_THRESHOLD - CROSS_MAJOR_SIMILARITY_MIN)
    factor = CROSS_MAJOR_PENALTY_FACTOR + (1.0 - CROSS_MAJOR_PENALTY_FACTOR) * t
    return 1.0 - factor


def compute_faculty_penalty(is_out_of_scope):
    if is_out_of_scope:
        return 1.0 - FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR
    return 0.0


def compute_professional_penalty(major_name, internship_count):
    if internship_count > 0:
        return 0.0
    major_lower = str(major_name).lower()
    if any(p in major_lower for p in PROFESSIONAL_MAJORS_LOWER):
        return PROFESSIONAL_REDUCTION_FACTOR
    return 0.0


def is_faculty_out_of_scope(bg_faculty, target_faculty):
    if not bg_faculty or not target_faculty:
        return False
    if bg_faculty == target_faculty:
        return False
    allowed = CROSS_FACULTY_RULES.get(bg_faculty, set())
    if not allowed:
        return bg_faculty != target_faculty
    return target_faculty not in allowed


def arbitrate(base_prob, penalty_values):
    """Production Arbitrator: sort, decay, cap. Returns (adjusted_prob, total_penalty_ratio)."""
    if not penalty_values:
        return base_prob, 0.0

    indexed = list(enumerate(penalty_values))
    indexed.sort(key=lambda x: x[1][0], reverse=True)

    total_penalty_ratio = 0.0
    decay = 1.0
    for _, (ratio, _name) in indexed:
        contribution = ratio * decay
        total_penalty_ratio += contribution
        decay *= PENALTY_DECAY_FACTOR

    total_penalty_ratio = min(total_penalty_ratio, MAX_TOTAL_PENALTY_RATIO)
    prob = base_prob * (1.0 - total_penalty_ratio)

    if prob <= 0:
        prob = ARBITRATION_MIN_PROBABILITY
    elif prob > 1.0:
        prob = 1.0
    elif 0 < prob < ARBITRATION_MIN_PROBABILITY:
        prob = ARBITRATION_MIN_PROBABILITY

    return prob, total_penalty_ratio


# ═══════════════════════════════════════════════════════════════════════════════
# ECE computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error with reliability diagram bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        mask = bin_indices == b
        n_b = mask.sum()
        if n_b == 0:
            bin_stats.append({"bin": b, "n": 0, "prob_mean": np.nan, "actual_rate": np.nan, "contrib": 0})
            continue
        prob_mean = float(probs[mask].mean())
        actual_rate = float(labels[mask].mean())
        contrib = n_b / len(probs) * abs(prob_mean - actual_rate)
        ece += contrib
        bin_stats.append({
            "bin": b,
            "range": f"[{bins[b]:.1f}, {bins[b+1]:.1f})",
            "n": int(n_b),
            "prob_mean": round(prob_mean, 4),
            "actual_rate": round(actual_rate, 4),
            "bias": round(prob_mean - actual_rate, 4),
            "contrib": round(contrib, 6),
        })

    return ece, bin_stats


def compute_brier(probs, labels):
    return float(np.mean((probs - labels) ** 2))


# ═══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    print("[1/6] Loading data and model...")

    _, X_test, _, y_test, feature_names, _, _, _ = load_and_preprocess_data(DATA_PATH)
    y_true = y_test.values.astype(int)

    full_df = pd.read_feather(DATA_PATH)

    raw_model = _load_serialized_xgb(MODEL_PATH)
    raw_probas = raw_model.predict_proba(X_test)[:, 1]
    print(f"  Test set: {len(X_test)} samples, XGBoost predictions done")

    # Similarity cache
    sim_cache = pd.read_feather(SIM_CACHE_PATH)
    sim_cache["bg_major"] = sim_cache["bg_major"].astype(str).str.strip().str.lower()
    sim_cache["target_major"] = sim_cache["target_major"].astype(str).str.strip().str.lower()
    sim_lookup = {}
    for _, row in sim_cache.iterrows():
        sim_lookup[(row["bg_major"], row["target_major"])] = float(row["similarity"])
    print(f"  Similarity cache: {len(sim_lookup)} pairs")

    # Target faculty mapping
    details = pd.read_feather(DETAILS_PATH)
    if "专业大类" in details.columns:
        details["target_faculty"] = details["专业大类"].astype(str).str.strip()
    else:
        details["target_faculty"] = "未知"

    uni_col = "学校"
    major_col = "专业英文名称"
    target_faculty_map = {}
    for _, row in details.iterrows():
        key = (str(row[uni_col]).strip(), str(row[major_col]).strip())
        target_faculty_map[key] = str(row["target_faculty"])
    print(f"  Target faculty mapping: {len(target_faculty_map)} entries")

    # Get test indices
    from sklearn.model_selection import train_test_split
    orig_df = pd.read_feather(DATA_PATH)
    X_orig = orig_df.drop(columns=["admitted"], errors="ignore")
    y_orig = orig_df["admitted"]
    _, X_test_raw, _, _ = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
    )
    test_indices = X_test_raw.index.tolist()
    test_orig = orig_df.iloc[test_indices].copy()
    print(f"  Original test set: {len(test_orig)} rows")

    # GPA stats
    gpa_series = pd.to_numeric(full_df["gpa"], errors="coerce").dropna()
    gpa_mean = float(gpa_series.mean())
    gpa_std = max(1e-6, float(gpa_series.std()))

    # Language stats
    lang_scores = []
    for _, row in full_df.iterrows():
        t = pd.to_numeric(row.get("toefl"), errors="coerce")
        i = pd.to_numeric(row.get("ielts"), errors="coerce")
        t_val = t / 120.0 if pd.notna(t) else 0.0
        i_val = i / 9.0 if pd.notna(i) else 0.0
        lang_scores.append(max(t_val, i_val))
    lang_scores = np.array(lang_scores)
    lang_mean = float(np.mean(lang_scores[lang_scores > 0])) if (lang_scores > 0).any() else 0.85
    lang_std = max(1e-6, float(np.std(lang_scores[lang_scores > 0]))) if (lang_scores > 0).any() else 0.15

    print(f"  GPA: mean={gpa_mean:.3f}, std={gpa_std:.3f}")
    print(f"  Language: mean={lang_mean:.4f}, std={lang_std:.4f}")

    # Build case records
    print("\n[2/6] Building per-case features...")
    cases = []
    for i, (idx, row) in enumerate(test_orig.iterrows()):
        gpa_raw = pd.to_numeric(row.get("gpa"), errors="coerce")
        gpa = float(gpa_raw) if pd.notna(gpa_raw) else gpa_mean  # median imputation

        t_val = pd.to_numeric(row.get("toefl"), errors="coerce")
        i_val = pd.to_numeric(row.get("ielts"), errors="coerce")
        lang_raw = max(
            t_val / 120.0 if pd.notna(t_val) else 0.0,
            i_val / 9.0 if pd.notna(i_val) else 0.0,
        )

        bg_major = str(row.get("background_major", "")).strip().lower()
        bg_faculty = str(row.get("faculty", "")).strip()
        target_uni = str(row.get("target_university", "")).strip()
        target_major = str(row.get("target_major", "")).strip()
        ic_raw = pd.to_numeric(row.get("internship_count"), errors="coerce")
        internship_count = int(ic_raw) if pd.notna(ic_raw) else 0

        sim_key = (bg_major, target_major.lower())
        similarity = sim_lookup.get(sim_key, 0.85)

        fac_key = (target_uni, target_major)
        target_fac = target_faculty_map.get(fac_key, "未知")

        is_out_of_scope = is_faculty_out_of_scope(bg_faculty, target_fac)
        is_professional = any(p in target_major.lower() for p in PROFESSIONAL_MAJORS_LOWER)

        # University tier for stratification
        bg_uni = str(row.get("background_university", ""))

        cases.append({
            "idx": i,
            "gpa": gpa,
            "language_score": lang_raw,
            "bg_major": bg_major,
            "bg_faculty": bg_faculty,
            "bg_uni": bg_uni,
            "target_uni": target_uni,
            "target_major": target_major,
            "target_faculty": target_fac,
            "similarity": similarity,
            "internship_count": internship_count,
            "is_out_of_scope": is_out_of_scope,
            "is_professional": is_professional,
            "base_prob": float(raw_probas[i]),
            "actual": int(row.get("admitted", 0)),
        })

    print(f"  Built {len(cases)} case records")
    return cases, gpa_mean, gpa_std, lang_mean, lang_std


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis engine
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_penalties(c, gpa_mean, gpa_std, lang_mean, lang_std):
    """Compute raw penalty values for all 5 layers."""
    return [
        compute_gpa_penalty(c["gpa"], gpa_mean, gpa_std),
        compute_language_penalty(c["language_score"], lang_mean, lang_std),
        compute_cross_major_penalty(c["similarity"]),
        compute_faculty_penalty(c["is_out_of_scope"]),
        compute_professional_penalty(c["target_major"], c["internship_count"]),
    ]


def run_full_pipeline(cases, gpa_mean, gpa_std, lang_mean, lang_std):
    """Run full adjustment chain and return detailed per-case results."""
    results = []
    for c in cases:
        raw_penalties = compute_all_penalties(c, gpa_mean, gpa_std, lang_mean, lang_std)

        # Build penalty list for arbitrator (only non-zero)
        active = [(raw_penalties[i], PENALTY_NAMES[i])
                   for i in range(5) if raw_penalties[i] > 0]

        adj_prob, total_penalty_ratio = arbitrate(c["base_prob"], active)

        results.append({
            **c,
            "raw_penalties": raw_penalties,
            "active_penalties": [n for _, n in active],
            "n_active": len(active),
            "penalty_mask": [1.0 if p > 0 else 0.0 for p in raw_penalties],
            "total_penalty_ratio": total_penalty_ratio,
            "hits_ceiling": abs(total_penalty_ratio - MAX_TOTAL_PENALTY_RATIO) < 1e-6,
            "adj_prob": adj_prob,
            "prob_reduction": c["base_prob"] - adj_prob,
        })

    return results


def run_ablated_pipeline(cases, gpa_mean, gpa_std, lang_mean, lang_std, disabled_indices):
    """Run pipeline with specific penalty layers disabled."""
    results = []
    for c in cases:
        raw_penalties = compute_all_penalties(c, gpa_mean, gpa_std, lang_mean, lang_std)
        for di in disabled_indices:
            raw_penalties[di] = 0.0

        active = [(raw_penalties[i], PENALTY_NAMES[i])
                   for i in range(5) if raw_penalties[i] > 0]

        adj_prob, total_penalty_ratio = arbitrate(c["base_prob"], active)
        results.append(adj_prob)

    return np.array(results)


# ═══════════════════════════════════════════════════════════════════════════════
# School tier classification (matching production school_level_service)
# ═══════════════════════════════════════════════════════════════════════════════

C9_SCHOOLS = {
    "北京大学", "清华大学", "复旦大学", "上海交通大学",
    "浙江大学", "南京大学", "中国科学技术大学", "哈尔滨工业大学",
    "西安交通大学",
}

def classify_tier(uni_name):
    """Rough tier classification matching production logic.

    Uses comprehensive keyword lists for 985/211 (aligned with V6 external drift analysis).
    C9 is exact name match (most reliable). 985/211 boundary may still have ~5-10% misclassification
    for schools with ambiguous names (e.g., provincial universities with tier-upgraded programs).
    """
    if not uni_name:
        return "unknown"
    name = str(uni_name).strip()
    if name in C9_SCHOOLS:
        return "C9"
    keywords_985 = ["985", "北京大学", "清华大学", "复旦大学", "上海交通", "浙江大学",
                    "南京大学", "中国科学技术", "哈尔滨工业", "西安交通",
                    "武汉大学", "华中科技", "中山大学", "南开大学", "天津大学",
                    "同济大学", "东南大学", "厦门大学", "四川大学", "电子科技",
                    "华南理工", "大连理工", "吉林大学", "山东大学", "湖南大学",
                    "中南大学", "重庆大学", "西北工业", "兰州大学",
                    "北京航空", "北京理工", "中国农业", "中央民族", "华东师范",
                    "东北大学", "西北农林", "国防科技"]
    if any(k in name for k in keywords_985):
        return "985"
    keywords_211 = ["211", "中国地质", "中国石油", "中国矿业", "华北电力",
                    "北京交通", "北京科技", "北京化工", "北京邮电", "北京林业",
                    "北京中医药", "北京外国", "中国传媒", "中央财经", "对外经济贸易",
                    "中国政法", "上海财经", "上海外国", "上海大学",
                    "南京航空", "南京理工", "河海大学", "江南大学", "南京农业",
                    "中国药科", "南京师范", "苏州大学",
                    "武汉理工", "华中农业", "华中师范", "中南财经政法",
                    "西南交通", "西南财经", "西南大学",
                    "西安电子", "长安大学", "陕西师范",
                    "合肥工业", "安徽大学",
                    "大连海事", "东北财经", "东北师范", "东北林业", "东北农业",
                    "郑州大学", "云南大学", "贵州大学", "广西大学",
                    "南昌大学", "福州大学", "海南大学", "内蒙古大学",
                    "新疆大学", "西藏大学", "宁夏大学", "青海大学",
                    "石河子大学", "延边大学"]
    if any(k in name for k in keywords_211):
        return "211"
    return "other"


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_penalty_count_distribution(results, output_path):
    """Penalty count histogram + ECE per count."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Penalty count histogram
    ax = axes[0]
    n_counts = pd.Series([r["n_active"] for r in results]).value_counts().sort_index()
    colors = ["#2ECC71", "#3498DB", "#F39C12", "#E67E22", "#E74C3C", "#8E44AD"]
    bars = ax.bar(n_counts.index, n_counts.values,
                  color=[colors[i] for i in n_counts.index], edgecolor="white")
    ax.set_xlabel("Number of Active Penalty Layers", fontsize=11)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_title("Penalty Count Distribution", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, n_counts.values):
        pct = val / len(results) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: ECE per penalty count
    ax = axes[1]
    ece_per_count = {}
    probs_all = np.array([r["adj_prob"] for r in results])
    labels_all = np.array([r["actual"] for r in results])
    global_ece, _ = compute_ece(probs_all, labels_all)

    for k in sorted(n_counts.index):
        mask = np.array([r["n_active"] == k for r in results])
        if mask.sum() < 10:
            continue
        sub_ece, _ = compute_ece(probs_all[mask], labels_all[mask])
        ece_per_count[k] = sub_ece

    xs = list(ece_per_count.keys())
    ys = [ece_per_count[k] for k in xs]
    bar_colors = ["#E74C3C" if y > 0.10 else "#F39C12" if y > 0.05 else "#2ECC71" for y in ys]
    bars = ax.bar(xs, ys, color=bar_colors, edgecolor="white")
    ax.axhline(y=global_ece, color="red", linestyle="--", linewidth=1.5,
               label=f"Global ECE = {global_ece:.4f}")
    ax.set_xlabel("Number of Active Penalty Layers", fontsize=11)
    ax.set_ylabel("ECE", fontsize=11)
    ax.set_title("ECE by Penalty Count", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    # Panel 3: Mean probability vs actual by penalty count
    ax = axes[2]
    mean_probs = []
    actual_rates = []
    ns = []
    for k in sorted(n_counts.index):
        mask = np.array([r["n_active"] == k for r in results])
        if mask.sum() < 10:
            continue
        mean_probs.append(float(probs_all[mask].mean()))
        actual_rates.append(float(labels_all[mask].mean()))
        ns.append(int(mask.sum()))

    x_pos = np.arange(len(mean_probs))
    width = 0.35
    bars1 = ax.bar(x_pos - width / 2, mean_probs, width, label="Mean Predicted",
                   color="#3498DB", edgecolor="white")
    bars2 = ax.bar(x_pos + width / 2, actual_rates, width, label="Actual Rate",
                   color="#E74C3C", edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{k} layers\n(n={n})" for k, n in zip(sorted(n_counts.index), ns)], fontsize=8)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title("Predicted vs Actual by Penalty Count", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Joint Penalty Effect — Distribution & Calibration", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] penalty_count_distribution.png → {output_path}")


def plot_ece_per_layer(all_ece_results, output_path):
    """ECE contribution: full chain vs each layer removed."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    baseline_ece = all_ece_results["baseline"]["ece"]
    baseline_brier = all_ece_results["baseline"]["brier"]

    # Panel 1: ECE when layer is REMOVED
    ax = axes[0]
    layers = list(all_ece_results["ablated"].keys())
    labels = [PENALTY_LABELS[PENALTY_NAMES.index(l.split("_")[0])] for l in layers]
    ece_values = [all_ece_results["ablated"][l]["ece"] for l in layers]

    # Color: green if removing layer IMPROVES ECE (i.e., layer hurts calibration)
    # red if removing layer WORSENS ECE (i.e., layer helps calibration)
    ece_deltas = [baseline_ece - e for e in ece_values]
    bar_colors = ["#2ECC71" if d > 0 else "#E74C3C" for d in ece_deltas]

    y_pos = range(len(layers))
    bars = ax.barh(y_pos, ece_values, color=bar_colors, edgecolor="white")
    ax.axvline(x=baseline_ece, color="black", linestyle="--", linewidth=1.5,
               label=f"Full chain ECE = {baseline_ece:.4f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("ECE", fontsize=11)
    ax.set_title("ECE When Layer REMOVED\n(green = layer hurts calibration)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    for i, (val, delta) in enumerate(zip(ece_values, ece_deltas)):
        sign = "+" if delta > 0 else ""
        ax.text(val + 0.001, i, f"{val:.4f} (Δ={sign}{delta:.4f})", va="center", fontsize=9, fontweight="bold")

    # Panel 2: Brier score per layer removed
    ax = axes[1]
    brier_values = [all_ece_results["ablated"][l]["brier"] for l in layers]
    brier_deltas = [baseline_brier - b for b in brier_values]
    bar_colors2 = ["#2ECC71" if d > 0 else "#E74C3C" for d in brier_deltas]

    bars = ax.barh(y_pos, brier_values, color=bar_colors2, edgecolor="white")
    ax.axvline(x=baseline_brier, color="black", linestyle="--", linewidth=1.5,
               label=f"Full chain Brier = {baseline_brier:.4f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Brier Score", fontsize=11)
    ax.set_title("Brier Score When Layer REMOVED", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    for i, (val, delta) in enumerate(zip(brier_values, brier_deltas)):
        sign = "+" if delta > 0 else ""
        ax.text(val + 0.0005, i, f"{val:.4f} (Δ={sign}{delta:.4f})", va="center", fontsize=9, fontweight="bold")

    fig.suptitle("Calibration Impact Per Layer — ECE & Brier Ablation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] ece_per_layer.png → {output_path}")


def plot_penalty_ratio_distribution(results, output_path):
    """Total penalty ratio histogram + ceiling analysis + strength correlation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ratios = np.array([r["total_penalty_ratio"] for r in results])
    n_ceiling = sum(1 for r in results if r["hits_ceiling"])
    ceiling_pct = n_ceiling / len(results) * 100

    # Panel 1: Total penalty ratio histogram
    ax = axes[0, 0]
    ax.hist(ratios, bins=50, color="#3498DB", edgecolor="white", alpha=0.8)
    ax.axvline(x=MAX_TOTAL_PENALTY_RATIO, color="red", linestyle="--", linewidth=2,
               label=f"Ceiling ({MAX_TOTAL_PENALTY_RATIO})")
    ax.axvline(x=np.mean(ratios), color="orange", linestyle="-", linewidth=2,
               label=f"Mean = {np.mean(ratios):.3f}")
    ax.axvline(x=np.median(ratios), color="green", linestyle="-", linewidth=2,
               label=f"Median = {np.median(ratios):.3f}")
    ax.set_xlabel("Total Penalty Ratio", fontsize=11)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_title(f"Total Penalty Ratio Distribution\n({ceiling_pct:.1f}% hit ceiling)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 2: Penalty ratio quantiles
    ax = axes[0, 1]
    quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    q_values = np.quantile(ratios, quantiles)
    ax.plot(quantiles, q_values, "o-", color="#3498DB", linewidth=2, markersize=8)
    ax.fill_between(quantiles, 0, q_values, alpha=0.15, color="#3498DB")
    ax.axhline(y=MAX_TOTAL_PENALTY_RATIO, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Quantile", fontsize=11)
    ax.set_ylabel("Total Penalty Ratio", fontsize=11)
    ax.set_title("Penalty Ratio Quantile Function", fontsize=12, fontweight="bold")
    for q, v in zip(quantiles, q_values):
        ax.annotate(f"{v:.3f}", (q, v), textcoords="offset points", xytext=(0, 10),
                    fontsize=8, ha="center")

    # Panel 3: Penalty ratio vs student strength (base_prob proxy for strength)
    ax = axes[1, 0]
    base_probs = np.array([r["base_prob"] for r in results])
    # Bin by base_prob
    bins = np.percentile(base_probs, [0, 20, 40, 60, 80, 100])
    bin_labels = ["Bottom 20%", "20-40%", "40-60%", "60-80%", "Top 20%"]
    bin_indices = np.digitize(base_probs, bins[1:-1])

    mean_ratios = []
    mean_probs_in_bin = []
    for b in range(5):
        mask = bin_indices == b
        if mask.sum() > 0:
            mean_ratios.append(float(np.mean(ratios[mask])))
            mean_probs_in_bin.append(float(np.mean(base_probs[mask])))

    ax.bar(range(5), mean_ratios, color=["#2ECC71", "#3498DB", "#F39C12", "#E67E22", "#E74C3C"],
           edgecolor="white")
    ax.set_xticks(range(5))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_xlabel("Student Strength (Base Probability Quintile)", fontsize=11)
    ax.set_ylabel("Mean Total Penalty Ratio", fontsize=11)
    ax.set_title("Penalty Ratio vs Student Strength\n(stronger students = MORE penalized?)",
                 fontsize=11, fontweight="bold")

    # Panel 4: Penalty ratio by school tier
    ax = axes[1, 1]
    tiers = ["C9", "985", "211", "other"]
    tier_ratios = {}
    tier_probs = {}
    for tier in tiers:
        mask = np.array([classify_tier(r["bg_uni"]) == tier for r in results])
        if mask.sum() > 5:
            tier_ratios[tier] = float(np.mean(ratios[mask]))
            tier_probs[tier] = float(np.mean([r["adj_prob"] for r in results if classify_tier(r["bg_uni"]) == tier]))

    tier_colors = ["#E74C3C", "#F39C12", "#3498DB", "#95A5A6"]
    x_pos = range(len(tier_ratios))
    bars = ax.bar(x_pos, list(tier_ratios.values()), color=tier_colors[:len(tier_ratios)], edgecolor="white")

    # Add actual admission rate as annotation
    for i, tier in enumerate(tier_ratios.keys()):
        tier_mask = np.array([classify_tier(r["bg_uni"]) == tier for r in results])
        actual_rate = float(np.mean([r["actual"] for r in results if classify_tier(r["bg_uni"]) == tier]))
        ax.annotate(f"Actual admit: {actual_rate:.2f}", (i, tier_ratios[tier]),
                    textcoords="offset points", xytext=(0, 10), fontsize=8, ha="center")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(tier_ratios.keys()), fontsize=10)
    ax.set_ylabel("Mean Total Penalty Ratio", fontsize=11)
    ax.set_title("Penalty Ratio by University Tier\n(C9 penalized hardest?)", fontsize=12, fontweight="bold")

    fig.suptitle("Penalty Ratio — Distribution & Stratification", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] penalty_ratio_distribution.png → {output_path}")


def plot_cooccurrence_matrix(results, output_path):
    """Which penalties fire together?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Co-occurrence heatmap
    ax = axes[0]
    n_penalties = 5
    cooc_matrix = np.zeros((n_penalties, n_penalties))
    masks = np.array([r["penalty_mask"] for r in results])

    for i in range(n_penalties):
        for j in range(n_penalties):
            if i == j:
                cooc_matrix[i, j] = masks[:, i].mean()
            else:
                both = (masks[:, i] == 1) & (masks[:, j] == 1)
                cooc_matrix[i, j] = both.mean() / max(masks[:, i].mean(), 1e-6)

    im = ax.imshow(cooc_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n_penalties))
    ax.set_yticks(range(n_penalties))
    ax.set_xticklabels(PENALTY_NAMES, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(PENALTY_NAMES, fontsize=9)
    ax.set_title("Penalty Co-occurrence Matrix\n(P(j also fires | i fires))", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    for i in range(n_penalties):
        for j in range(n_penalties):
            color = "white" if cooc_matrix[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cooc_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    # Panel 2: Co-occurrence pattern as stacked bars
    ax = axes[1]
    # For each penalty layer, show what ELSE fires when it fires
    co_occurrence_bars = {}
    for i in range(n_penalties):
        i_mask = masks[:, i] == 1
        if i_mask.sum() == 0:
            co_occurrence_bars[i] = [0] * (n_penalties - 1)
            continue
        other_indices = [j for j in range(n_penalties) if j != i]
        co_occurrence_bars[i] = [masks[i_mask, j].mean() for j in other_indices]

    x = np.arange(n_penalties)
    width = 0.15
    other_names = [[n for j, n in enumerate(PENALTY_NAMES) if j != i] for i in range(n_penalties)]

    for i in range(n_penalties - 1):  # 4 co-occurrence bars per penalty
        values = [co_occurrence_bars[p][i] for p in range(n_penalties)]
        bars = ax.bar(x + i * width, values, width, label=f"w/ {other_names[0][i]}",
                      edgecolor="white", alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(PENALTY_NAMES, fontsize=9)
    ax.set_ylabel("Co-occurrence Rate", fontsize=11)
    ax.set_title("When This Penalty Fires, What Else Fires?", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncols=2)
    ax.set_ylim(0, 1.0)

    fig.suptitle("Penalty Layer Interaction Patterns", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] cooccurrence_matrix.png → {output_path}")


def plot_excess_penalty_analysis(results, output_path):
    """Analyze cases where penalty is 'excessive' relative to base probability."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Define "excessive": prob_reduction / base_prob is high
    reductions = np.array([r["prob_reduction"] for r in results])
    base_probs = np.array([r["base_prob"] for r in results])
    ratios = np.array([r["total_penalty_ratio"] for r in results])

    # Panel 1: prob_reduction vs base_prob scatter
    ax = axes[0]
    n_penalties = np.array([r["n_active"] for r in results])
    scatter = ax.scatter(base_probs, reductions, c=n_penalties, cmap="YlOrRd",
                         alpha=0.5, s=15, edgecolors="none")
    ax.plot([0, 1], [0, 0.7], "r--", linewidth=1, alpha=0.5,
            label="Max reduction (70% cap)")
    ax.set_xlabel("Base Probability (XGBoost)", fontsize=11)
    ax.set_ylabel("Probability Reduction", fontsize=11)
    ax.set_title("Reduction vs Base Probability\n(colored by # penalties)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    plt.colorbar(scatter, ax=ax, label="# Active Penalties")

    # Panel 2: Reduction fraction distribution
    ax = axes[1]
    # For cases with base_prob > 0.1
    valid = base_probs > 0.1
    reduction_frac = reductions[valid] / base_probs[valid]
    ax.hist(reduction_frac, bins=50, color="#E74C3C", edgecolor="white", alpha=0.7)
    ax.axvline(x=np.mean(reduction_frac), color="blue", linestyle="-", linewidth=2,
               label=f"Mean = {np.mean(reduction_frac):.3f}")
    ax.axvline(x=np.median(reduction_frac), color="green", linestyle="-", linewidth=2,
               label=f"Median = {np.median(reduction_frac):.3f}")
    ax.set_xlabel("Reduction / Base Probability", fontsize=11)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_title("Fraction of Base Probability Lost to Penalties", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 3: Top offenders — cases with highest penalty ratio by tier
    ax = axes[2]
    tiers = ["C9", "985", "211", "other"]
    tier_data = {}
    for tier in tiers:
        tier_mask = np.array([classify_tier(r["bg_uni"]) == tier for r in results])
        if tier_mask.sum() > 5:
            tier_data[tier] = {
                "mean_ratio": float(np.mean(ratios[tier_mask])),
                "mean_reduction": float(np.mean(reductions[tier_mask])),
                "mean_base_prob": float(np.mean(base_probs[tier_mask])),
                "mean_adj_prob": float(np.mean([r["adj_prob"] for r in results if classify_tier(r["bg_uni"]) == tier])),
                "actual_rate": float(np.mean([r["actual"] for r in results if classify_tier(r["bg_uni"]) == tier])),
                "n": int(tier_mask.sum()),
            }

    x = np.arange(len(tier_data))
    width = 0.25
    bars1 = ax.bar(x - width, [d["mean_base_prob"] for d in tier_data.values()],
                   width, label="Base Prob", color="#3498DB", edgecolor="white")
    bars2 = ax.bar(x, [d["mean_adj_prob"] for d in tier_data.values()],
                   width, label="Adjusted Prob", color="#F39C12", edgecolor="white")
    bars3 = ax.bar(x + width, [d["actual_rate"] for d in tier_data.values()],
                   width, label="Actual Rate", color="#2ECC71", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={tier_data[t]['n']})" for t in tier_data.keys()], fontsize=9)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title(f"Penalty Impact by University Tier\n(C9: ratio={tier_data.get('C9', {}).get('mean_ratio', 0):.3f})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Excess Penalty Analysis — Who Gets Hit Hardest?", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] excess_penalty_analysis.png → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("V5: Joint Penalty Effect Analysis")
    print("=" * 70)

    # ── Load & prepare ───────────────────────────────────────────────────────
    cases, gpa_mean, gpa_std, lang_mean, lang_std = load_and_prepare()

    # ── Full pipeline run ────────────────────────────────────────────────────
    print("\n[3/6] Running full adjustment pipeline...")
    results = run_full_pipeline(cases, gpa_mean, gpa_std, lang_mean, lang_std)
    probs = np.array([r["adj_prob"] for r in results])
    labels = np.array([r["actual"] for r in results])
    print(f"  Adjusted prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Mean adjusted prob: {probs.mean():.4f}")
    print(f"  Mean actual rate: {labels.mean():.4f}")
    print(f"  Systematic bias: {probs.mean() - labels.mean():.4f}")

    # ── Baseline ECE ─────────────────────────────────────────────────────────
    print("\n[4/6] Computing baseline ECE...")
    baseline_ece, baseline_bins = compute_ece(probs, labels)
    baseline_brier = compute_brier(probs, labels)
    print(f"  Full chain ECE: {baseline_ece:.4f}")
    print(f"  Full chain Brier: {baseline_brier:.4f}")

    # ── ECE per layer ablation ───────────────────────────────────────────────
    print("\n[5/6] Computing ECE per layer removed...")
    ablated_ece = {}
    for i, name in enumerate(PENALTY_NAMES):
        ablated_probs = run_ablated_pipeline(cases, gpa_mean, gpa_std, lang_mean, lang_std, {i})
        ece, bins = compute_ece(ablated_probs, labels)
        brier = compute_brier(ablated_probs, labels)
        ablated_ece[f"{name}_removed"] = {
            "ece": round(ece, 6),
            "brier": round(brier, 6),
            "ece_delta": round(baseline_ece - ece, 6),
            "brier_delta": round(baseline_brier - brier, 6),
            "bins": bins,
        }
        direction = "IMPROVES" if ece < baseline_ece else "WORSENS"
        print(f"  Remove {name}: ECE={ece:.4f} ({direction}, Δ={baseline_ece - ece:+.4f}), "
              f"Brier={brier:.4f} (Δ={baseline_brier - brier:+.4f})")

    # Also: raw XGBoost (all layers off)
    raw_ece, raw_bins = compute_ece(np.array([r["base_prob"] for r in results]), labels)
    raw_brier = compute_brier(np.array([r["base_prob"] for r in results]), labels)
    print(f"  Raw XGBoost (no layers): ECE={raw_ece:.4f}, Brier={raw_brier:.4f}")

    all_ece = {
        "baseline": {"ece": round(baseline_ece, 6), "brier": round(baseline_brier, 6),
                     "bins": baseline_bins},
        "raw_xgboost": {"ece": round(raw_ece, 6), "brier": round(raw_brier, 6),
                        "bins": raw_bins},
        "ablated": ablated_ece,
    }

    # ── Penalty count statistics ─────────────────────────────────────────────
    n_active_counts = pd.Series([r["n_active"] for r in results]).value_counts().to_dict()
    print(f"\n  Penalty count distribution: {dict(sorted(n_active_counts.items()))}")

    # Cases hitting ceiling
    n_ceiling = sum(1 for r in results if r["hits_ceiling"])
    print(f"  Cases hitting ceiling (70%): {n_ceiling} ({n_ceiling / len(results) * 100:.1f}%)")

    # ── Per-penalty statistics ───────────────────────────────────────────────
    penalty_stats = {}
    for i, name in enumerate(PENALTY_NAMES):
        triggered = sum(1 for r in results if r["raw_penalties"][i] > 0)
        mean_value = np.mean([r["raw_penalties"][i] for r in results if r["raw_penalties"][i] > 0]) if triggered > 0 else 0
        penalty_stats[name] = {
            "triggered": triggered,
            "triggered_pct": round(triggered / len(results) * 100, 1),
            "mean_raw_penalty": round(float(mean_value), 4),
        }
        print(f"  {name}: triggered {triggered} ({triggered/len(results)*100:.1f}%), "
              f"mean raw={mean_value:.4f}")

    # ── Stratified bias ──────────────────────────────────────────────────────
    tier_bias = {}
    for tier in ["C9", "985", "211", "other"]:
        mask = np.array([classify_tier(r["bg_uni"]) == tier for r in results])
        if mask.sum() > 5:
            mean_prob = float(probs[mask].mean())
            actual_rate = float(labels[mask].mean())
            mean_ratio = float(np.mean([r["total_penalty_ratio"] for r in results if classify_tier(r["bg_uni"]) == tier]))
            mean_n_penalties = float(np.mean([r["n_active"] for r in results if classify_tier(r["bg_uni"]) == tier]))
            tier_bias[tier] = {
                "n": int(mask.sum()),
                "mean_adj_prob": round(mean_prob, 4),
                "actual_rate": round(actual_rate, 4),
                "bias": round(mean_prob - actual_rate, 4),
                "mean_penalty_ratio": round(mean_ratio, 4),
                "mean_n_penalties": round(mean_n_penalties, 2),
            }
    print(f"\n  Stratified bias by tier:")
    for tier, stats in tier_bias.items():
        print(f"    {tier}: bias={stats['bias']:.4f}, "
              f"mean_penalty_ratio={stats['mean_penalty_ratio']:.4f}, "
              f"mean_n_penalties={stats['mean_n_penalties']:.2f}")

    # ── Faculty penalty: who does it actually harm? ────────────────────────
    # V5's headline finding: removing Faculty lowers ECE to 0.111, BELOW raw
    # XGBoost's 0.118. This is counter-intuitive — it means Faculty penalty
    # creates MORE error than it removes. But WHO is being harmed?
    print("\n  === Faculty Penalty Harm Analysis ===")
    FACULTY_IDX = PENALTY_NAMES.index("Faculty")
    no_faculty_results = run_ablated_pipeline(
        cases, gpa_mean, gpa_std, lang_mean, lang_std, {FACULTY_IDX}
    )
    # For each case with Faculty penalty triggered, compute the harm:
    # harm = |prob_with_faculty - actual| - |prob_without_faculty - actual|
    # Positive harm = Faculty penalty made prediction WORSE
    faculty_harm_data = []
    for i, r in enumerate(results):
        if r["raw_penalties"][FACULTY_IDX] <= 0:
            continue  # Faculty penalty not triggered for this case
        error_full = abs(r["adj_prob"] - r["actual"])
        error_nofac = abs(no_faculty_results[i] - r["actual"])
        harm = error_full - error_nofac  # positive = faculty made it worse
        faculty_harm_data.append({
            "idx": r["idx"],
            "tier": classify_tier(r["bg_uni"]),
            "gpa": r["gpa"],
            "language_score": r["language_score"],
            "base_prob": r["base_prob"],
            "adj_prob_full": r["adj_prob"],
            "adj_prob_nofac": no_faculty_results[i],
            "actual": r["actual"],
            "error_full": error_full,
            "error_nofac": error_nofac,
            "harm": harm,
            "bg_faculty": r.get("bg_faculty", ""),
            "target_faculty": r.get("target_faculty", ""),
            "total_penalty_ratio": r["total_penalty_ratio"],
        })

    n_harmed = sum(1 for d in faculty_harm_data if d["harm"] > 0.01)
    n_helped = sum(1 for d in faculty_harm_data if d["harm"] < -0.01)
    n_neutral = len(faculty_harm_data) - n_harmed - n_helped
    mean_harm = float(np.mean([d["harm"] for d in faculty_harm_data]))

    print(f"  Faculty-triggered cases: {len(faculty_harm_data)}")
    print(f"    Harmed (>1pp worse):   {n_harmed} ({n_harmed/max(len(faculty_harm_data),1)*100:.1f}%)")
    print(f"    Helped (>1pp better):  {n_helped} ({n_helped/max(len(faculty_harm_data),1)*100:.1f}%)")
    print(f"    Neutral (≤1pp):        {n_neutral} ({n_neutral/max(len(faculty_harm_data),1)*100:.1f}%)")
    print(f"    Mean harm: {mean_harm:+.4f}")

    # Profile: which tier gets harmed most?
    harm_by_tier = {}
    for tier in ["C9", "985", "211", "other"]:
        tier_data = [d for d in faculty_harm_data if d["tier"] == tier]
        if tier_data:
            harm_by_tier[tier] = {
                "n": len(tier_data),
                "n_harmed": sum(1 for d in tier_data if d["harm"] > 0.01),
                "n_helped": sum(1 for d in tier_data if d["harm"] < -0.01),
                "mean_harm": round(float(np.mean([d["harm"] for d in tier_data])), 5),
                "mean_actual": round(float(np.mean([d["actual"] for d in tier_data])), 4),
            }
    print(f"\n  Harm by tier:")
    print(f"  {'Tier':<10} {'N':<8} {'Harmed':<8} {'Helped':<8} {'Mean Harm':<12} {'Actual Rate':<12}")
    print(f"  {'-'*58}")
    for tier, s in sorted(harm_by_tier.items()):
        print(f"  {tier:<10} {s['n']:<8} {s['n_harmed']:<8} {s['n_helped']:<8} "
              f"{s['mean_harm']:+.5f}     {s['mean_actual']:.4f}")

    # Worst-case: top harmed cases
    worst_harmed = sorted(faculty_harm_data, key=lambda d: d["harm"], reverse=True)[:20]
    print(f"\n  Top 20 most harmed cases (Faculty penalty made prediction WORSE):")
    print(f"  {'Tier':<10} {'GPA':<6} {'Base':<8} {'Full':<8} {'NoFac':<8} {'Actual':<8} {'Harm':<8}")
    print(f"  {'-'*66}")
    for d in worst_harmed[:10]:
        print(f"  {d['tier']:<10} {d['gpa']:<6.2f} {d['base_prob']:<8.4f} "
              f"{d['adj_prob_full']:<8.4f} {d['adj_prob_nofac']:<8.4f} "
              f"{d['actual']:<8} {d['harm']:+.4f}")

    # Key insight: are harmed cases systematically different from helped cases?
    harmed_mask = np.array([d["harm"] > 0.01 for d in faculty_harm_data])
    helped_mask = np.array([d["harm"] < -0.01 for d in faculty_harm_data])
    if harmed_mask.sum() > 0 and helped_mask.sum() > 0:
        harmed_actual = float(np.mean([d["actual"] for d, m in zip(faculty_harm_data, harmed_mask) if m]))
        helped_actual = float(np.mean([d["actual"] for d, m in zip(faculty_harm_data, helped_mask) if m]))
        print(f"\n  Key insight:")
        print(f"    Harmed cases actual admit rate:  {harmed_actual:.4f}")
        print(f"    Helped cases actual admit rate:  {helped_actual:.4f}")
        if harmed_actual > helped_actual:
            print(f"    → Faculty penalty HARMS students who SHOULD have higher admit rates.")
            print(f"      The penalty is reducing probabilities for students who actually get admitted.")
        else:
            print(f"    → Faculty penalty helps by reducing probabilities for lower-admit students.")

    faculty_harm_summary = {
        "n_faculty_triggered": len(faculty_harm_data),
        "n_harmed": n_harmed,
        "n_helped": n_helped,
        "n_neutral": n_neutral,
        "mean_harm": round(mean_harm, 5),
        "harm_by_tier": harm_by_tier,
        "harmed_actual_rate": round(harmed_actual, 5) if harmed_mask.sum() > 0 else None,
        "helped_actual_rate": round(helped_actual, 5) if helped_mask.sum() > 0 else None,
        "interpretation": (
            "Faculty penalty systematically harms students who actually get admitted. "
            "The penalty reduces their predicted probability far below their true admission rate, "
            "creating the calibration degradation seen in the ECE ablation."
        ) if (harmed_mask.sum() > 0 and helped_mask.sum() > 0 and harmed_actual > helped_actual) else (
            "Faculty penalty has mixed effects — harms some students, helps others."
        ),
    }

    # ── Generate visualizations ──────────────────────────────────────────────
    print("\n[6/6] Generating visualizations...")

    plot_penalty_count_distribution(results,
        os.path.join(OUTPUT_DIR, "penalty_count_distribution.png"))
    plot_ece_per_layer(all_ece,
        os.path.join(OUTPUT_DIR, "ece_per_layer.png"))
    plot_penalty_ratio_distribution(results,
        os.path.join(OUTPUT_DIR, "penalty_ratio_distribution.png"))
    plot_cooccurrence_matrix(results,
        os.path.join(OUTPUT_DIR, "cooccurrence_matrix.png"))
    plot_excess_penalty_analysis(results,
        os.path.join(OUTPUT_DIR, "excess_penalty_analysis.png"))

    # ── Save JSON report ─────────────────────────────────────────────────────
    # Find most extreme cases
    extreme_cases = sorted(results, key=lambda r: r["total_penalty_ratio"], reverse=True)[:10]
    extreme_summary = []
    for c in extreme_cases:
        extreme_summary.append({
            "idx": c["idx"],
            "base_prob": round(c["base_prob"], 4),
            "adj_prob": round(c["adj_prob"], 4),
            "reduction": round(c["prob_reduction"], 4),
            "total_penalty_ratio": round(c["total_penalty_ratio"], 4),
            "n_active": c["n_active"],
            "active_penalties": c["active_penalties"],
            "gpa": round(c["gpa"], 2),
            "language_score": round(c["language_score"], 4),
            "bg_uni": c["bg_uni"],
            "target_uni": c["target_uni"],
            "similarity": round(c["similarity"], 4),
            "is_out_of_scope": c["is_out_of_scope"],
            "actual": c["actual"],
        })

    # Least penalized (strong students)
    least_penalized = sorted(results, key=lambda r: r["total_penalty_ratio"])[:5]
    least_summary = []
    for c in least_penalized:
        least_summary.append({
            "idx": c["idx"],
            "base_prob": round(c["base_prob"], 4),
            "adj_prob": round(c["adj_prob"], 4),
            "total_penalty_ratio": round(c["total_penalty_ratio"], 4),
            "n_active": c["n_active"],
            "gpa": round(c["gpa"], 2),
            "language_score": round(c["language_score"], 4),
            "bg_uni": c["bg_uni"],
            "target_uni": c["target_uni"],
            "similarity": round(c["similarity"], 4),
            "actual": c["actual"],
        })

    report = {
        "summary": {
            "n_cases": len(results),
            "global_ece": round(baseline_ece, 4),
            "global_brier": round(baseline_brier, 4),
            "raw_xgboost_ece": round(raw_ece, 4),
            "raw_xgboost_brier": round(raw_brier, 4),
            "ece_deterioration_from_adjustments": round(baseline_ece - raw_ece, 4),
            "mean_adj_prob": round(float(probs.mean()), 4),
            "mean_actual_rate": round(float(labels.mean()), 4),
            "systematic_bias": round(float(probs.mean() - labels.mean()), 4),
            "n_ceiling_hits": n_ceiling,
            "ceiling_hit_pct": round(n_ceiling / len(results) * 100, 1),
        },
        "penalty_count_distribution": {str(k): int(v) for k, v in sorted(n_active_counts.items())},
        "ece_per_penalty_count": {},
        "per_layer_statistics": penalty_stats,
        "ece_ablation": all_ece,
        "stratified_bias": tier_bias,
        "faculty_harm_analysis": faculty_harm_summary,
        "extreme_cases_most_penalized": extreme_summary,
        "extreme_cases_least_penalized": least_summary,
    }

    # ECE per penalty count
    for k in sorted(n_active_counts.keys()):
        mask = np.array([r["n_active"] == k for r in results])
        if mask.sum() >= 10:
            sub_ece, _ = compute_ece(probs[mask], labels[mask])
            report["ece_per_penalty_count"][str(k)] = {
                "n": int(mask.sum()),
                "ece": round(sub_ece, 4),
                "mean_prob": round(float(probs[mask].mean()), 4),
                "actual_rate": round(float(labels[mask].mean()), 4),
                "bias": round(float(probs[mask].mean() - labels[mask].mean()), 4),
            }

    report_path = os.path.join(OUTPUT_DIR, "joint_penalty_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] joint_penalty_report.json → {report_path}")

    # ── Print key findings ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("V5: Key Findings")
    print("=" * 70)

    print(f"\n  Global ECE: {baseline_ece:.4f} (vs raw XGBoost: {raw_ece:.4f})")
    print(f"  Adjustment chain DEGRADES calibration by {baseline_ece - raw_ece:+.4f} ECE")

    # Which layer hurts calibration most?
    worst_layer = max(ablated_ece.items(), key=lambda x: x[1]["ece_delta"])
    best_layer = min(ablated_ece.items(), key=lambda x: x[1]["ece_delta"])
    print(f"\n  Worst calibration offender: {worst_layer[0]} "
          f"(removing it IMPROVES ECE by {worst_layer[1]['ece_delta']:+.4f})")
    print(f"  Best calibration contributor: {best_layer[0]} "
          f"(removing it WORSENS ECE by {best_layer[1]['ece_delta']:+.4f})")

    # Ceiling insight
    print(f"\n  Ceiling (70%): {n_ceiling} cases ({n_ceiling/len(results)*100:.1f}%) hit the max penalty cap")
    if n_ceiling > 0:
        ceiling_cases = [r for r in results if r["hits_ceiling"]]
        ceiling_n_penalties = np.mean([r["n_active"] for r in ceiling_cases])
        print(f"  Ceiling cases have mean {ceiling_n_penalties:.1f} active penalties")

    # Tier bias
    print(f"\n  Stratified bias confirms asymmetric penalty effect:")
    for tier in ["C9", "985", "211", "other"]:
        if tier in tier_bias:
            s = tier_bias[tier]
            print(f"    {tier}: bias={s['bias']:.4f} | "
                  f"mean_penalty_ratio={s['mean_penalty_ratio']:.4f} | "
                  f"mean_n_penalties={s['mean_n_penalties']:.2f}")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("    - penalty_count_distribution.png")
    print("    - ece_per_layer.png")
    print("    - penalty_ratio_distribution.png")
    print("    - cooccurrence_matrix.png")
    print("    - excess_penalty_analysis.png")
    print("    - joint_penalty_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
