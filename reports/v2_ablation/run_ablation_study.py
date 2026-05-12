"""
V2: 五层概率调整链 Ablation Study
====================================
逐层关掉，计算 Kendall tau + top-K 重叠率 + 每层影响最大的 case。
产出 ablation 矩阵图 + 极端 case trace。

运行方式: python reports/v2_ablation/run_ablation_study.py
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
from src.utils.model_loader import _load_serialized_xgb, _safe_json_loads

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "src", "machine_learning_models", "data", "cases.feather")
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "src", "machine_learning_models", "pre-trained_models",
    "xgboost_20260316_092608.ubj",
)
SIM_CACHE_PATH = os.path.join(PROJECT_ROOT, "cache", "background_target_similarity.feather")
DETAILS_PATH = os.path.join(PROJECT_ROOT, "src", "machine_learning_models", "data",
                           "school_major_details.feather")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Faculty rules (matching production faculty_filters.py) ─────────────────
# These define which target faculties are IN-SCOPE given a background faculty.
# If the target faculty is NOT in the allowed set → cross-faculty penalty ×0.3.
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


def is_faculty_out_of_scope(bg_faculty, target_faculty):
    """Check if target faculty is out of scope for background faculty.
    Matches production is_faculty_out_of_scope() logic.
    """
    if not bg_faculty or not target_faculty:
        return False
    if bg_faculty == target_faculty:
        return False
    allowed = CROSS_FACULTY_RULES.get(bg_faculty, set())
    if not allowed:
        # Unknown faculty → check if same as background, otherwise conservative
        return bg_faculty != target_faculty
    return target_faculty not in allowed

# ── constants from config.py ───────────────────────────────────────────────
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

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

LAYER_NAMES = [
    "L1_GPA_Penalty",
    "L2_Language_Penalty",
    "L3_Cross_Major_Penalty",
    "L4_Faculty_Penalty",
    "L5_Professional_Penalty",
]
LAYER_LABELS = [
    "GPA惩罚\n(GPA Penalty)",
    "语言惩罚\n(Language Penalty)",
    "跨专业惩罚\n(Cross-Major ×0.5)",
    "跨学部惩罚\n(Faculty ×0.3)",
    "职业学位惩罚\n(Professional Penalty)",
]


# ═══════════════════════════════════════════════════════════════════════════
# Layer implementations (matching production code formulas)
# ═══════════════════════════════════════════════════════════════════════════

def compute_gpa_penalty(gpa, gpa_mean, gpa_std):
    """Layer 1: quadratic penalty, production formula."""
    if gpa < GPA_MINIMUM:
        return GPA_PENALTY_SEVERE_THRESHOLD
    if gpa >= gpa_mean:
        return 0.0
    z = (gpa_mean - gpa) / max(gpa_std, 1e-6)
    return min(GPA_PENALTY_MAX_COEFFICIENT, GPA_PENALTY_QUADRATIC_COEFFICIENT * (z ** 2))


def compute_language_penalty(score, lang_mean, lang_std):
    """Layer 2: tiered threshold penalty, production formula."""
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
    """Layer 3: linear interpolation, production formula from utils.py."""
    if similarity >= MIN_SIMILARITY_THRESHOLD:
        return 0.0
    if similarity <= CROSS_MAJOR_SIMILARITY_MIN:
        return 1.0 - CROSS_MAJOR_PENALTY_FACTOR  # = 0.5
    t = (similarity - CROSS_MAJOR_SIMILARITY_MIN) / (MIN_SIMILARITY_THRESHOLD - CROSS_MAJOR_SIMILARITY_MIN)
    factor = CROSS_MAJOR_PENALTY_FACTOR + (1.0 - CROSS_MAJOR_PENALTY_FACTOR) * t
    return 1.0 - factor


def compute_faculty_penalty(is_out_of_scope):
    """Layer 4: hard ×0.3 when out of scope."""
    if is_out_of_scope:
        return 1.0 - FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR  # = 0.7
    return 0.0


def compute_professional_penalty(major_name, internship_count, is_user_specified=False):
    """Layer 5: internship-required professional degrees."""
    if internship_count > 0:
        return 0.0
    major_lower = str(major_name).lower()
    if any(p in major_lower for p in PROFESSIONAL_MAJORS_LOWER):
        factor = (PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR if is_user_specified
                  else PROFESSIONAL_REDUCTION_FACTOR)
        return factor  # 0.50 or 0.30
    return 0.0


def arbitrate(base_prob, penalty_values, penalty_names):
    """Production Arbitrator logic: sort, decay, cap.

    penalty_values: list of (penalty_ratio, name)
    Returns: adjusted probability
    """
    if not penalty_values:
        return base_prob

    # Sort by severity (largest first)
    indexed = list(enumerate(penalty_values))
    indexed.sort(key=lambda x: x[1][0], reverse=True)

    total_penalty_ratio = 0.0
    decay = 1.0
    for _, (ratio, name) in indexed:
        contribution = ratio * decay
        total_penalty_ratio += contribution
        decay *= PENALTY_DECAY_FACTOR

    total_penalty_ratio = min(total_penalty_ratio, MAX_TOTAL_PENALTY_RATIO)
    prob = base_prob * (1.0 - total_penalty_ratio)

    # Normalize
    if prob <= 0:
        prob = ARBITRATION_MIN_PROBABILITY
    elif prob > 1.0:
        prob = 1.0
    elif 0 < prob < ARBITRATION_MIN_PROBABILITY:
        prob = ARBITRATION_MIN_PROBABILITY

    return prob


# ═══════════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    """Load data, model, similarity cache, and build per-case features."""
    print("[1/5] 加载数据和模型...")

    # Data
    _, X_test, _, y_test, feature_names, _, _, _ = load_and_preprocess_data(DATA_PATH)
    y_true = y_test.values.astype(int)

    # Full dataset for GPA/language stats
    full_df = pd.read_feather(DATA_PATH)

    # Model
    raw_model = _load_serialized_xgb(MODEL_PATH)
    raw_probas = raw_model.predict_proba(X_test)[:, 1]
    print(f"  测试集: {len(X_test)} 样本, XGBoost 预测完成")

    # Similarity cache
    sim_cache = pd.read_feather(SIM_CACHE_PATH)
    sim_cache["bg_major"] = sim_cache["bg_major"].astype(str).str.strip().str.lower()
    sim_cache["target_major"] = sim_cache["target_major"].astype(str).str.strip().str.lower()
    sim_lookup = {}
    for _, row in sim_cache.iterrows():
        sim_lookup[(row["bg_major"], row["target_major"])] = float(row["similarity"])
    print(f"  相似度缓存: {len(sim_lookup)} 对")

    # Target faculty mapping from school_major_details
    details = pd.read_feather(DETAILS_PATH)
    # '专业大类' ≈ faculty category
    if "专业大类" in details.columns:
        details["target_faculty"] = details["专业大类"].astype(str).str.strip()
    else:
        details["target_faculty"] = "未知"

    # Build (university, major) → faculty lookup
    uni_col = "学校"
    major_col = "专业英文名称"
    target_faculty_map = {}
    for _, row in details.iterrows():
        key = (str(row[uni_col]).strip(), str(row[major_col]).strip())
        target_faculty_map[key] = str(row["target_faculty"])

    print(f"  目标专业学部映射: {len(target_faculty_map)} 条")

    # Build per-case data
    print("\n[2/5] 构建逐 case 特征...")
    orig_df = pd.read_feather(DATA_PATH)
    # Get test indices from the original split
    from sklearn.model_selection import train_test_split
    X_orig = orig_df.drop(columns=["admitted"], errors="ignore")
    y_orig = orig_df["admitted"]
    _, X_test_raw, _, _ = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
    )
    test_indices = X_test_raw.index.tolist()
    test_orig = orig_df.iloc[test_indices].copy()
    print(f"  原始测试集: {len(test_orig)} 行")

    # GPA stats from FULL dataset
    gpa_series = pd.to_numeric(full_df["gpa"], errors="coerce").dropna()
    gpa_mean = float(gpa_series.mean())
    gpa_std = max(1e-6, float(gpa_series.std()))

    # Language stats from FULL dataset
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
    cases = []
    for i, (idx, row) in enumerate(test_orig.iterrows()):
        gpa_raw = pd.to_numeric(row.get("gpa"), errors="coerce")
        gpa = float(gpa_raw) if pd.notna(gpa_raw) else 3.0

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

        # Lookup similarity
        sim_key = (bg_major, target_major.lower())
        similarity = sim_lookup.get(sim_key, 0.85)  # default mid-range

        # Lookup target faculty
        fac_key = (target_uni, target_major)
        target_fac = target_faculty_map.get(fac_key, "未知")

        # Faculty out-of-scope: use production CROSS_FACULTY_RULES
        # This allows related-faculty transitions (e.g., 理学院→工程学院 is OK)
        is_out_of_scope = is_faculty_out_of_scope(bg_faculty, target_fac)

        # Professional major check
        is_professional = any(p in target_major.lower() for p in PROFESSIONAL_MAJORS_LOWER)

        cases.append({
            "idx": i,
            "gpa": gpa,
            "language_score": lang_raw,
            "bg_major": bg_major,
            "bg_faculty": bg_faculty,
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

    print(f"  构建 {len(cases)} 条 case 记录")
    print(f"  跨学部比例: {sum(1 for c in cases if c['is_out_of_scope']) / len(cases):.1%}")
    print(f"  职业学位比例: {sum(1 for c in cases if c['is_professional']) / len(cases):.1%}")
    print(f"  跨专业惩罚触发: {sum(1 for c in cases if c['similarity'] < MIN_SIMILARITY_THRESHOLD) / len(cases):.1%}")

    return cases, gpa_mean, gpa_std, lang_mean, lang_std


# ═══════════════════════════════════════════════════════════════════════════
# Ablation engine
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(cases, gpa_mean, gpa_std, lang_mean, lang_std, disabled_layers=None):
    """Run adjustment pipeline, optionally disabling specified layers."""
    if disabled_layers is None:
        disabled_layers = set()

    results = []
    for c in cases:
        penalties = []

        if "L1" not in disabled_layers:
            p = compute_gpa_penalty(c["gpa"], gpa_mean, gpa_std)
            if p > 0:
                penalties.append((p, "GPA Penalty"))

        if "L2" not in disabled_layers:
            p = compute_language_penalty(c["language_score"], lang_mean, lang_std)
            if p > 0:
                penalties.append((p, "Language Penalty"))

        if "L3" not in disabled_layers:
            p = compute_cross_major_penalty(c["similarity"])
            if p > 0:
                penalties.append((p, "Cross Major Penalty"))

        if "L4" not in disabled_layers:
            p = compute_faculty_penalty(c["is_out_of_scope"])
            if p > 0:
                penalties.append((p, "Faculty Penalty"))

        if "L5" not in disabled_layers:
            p = compute_professional_penalty(c["target_major"], c["internship_count"])
            if p > 0:
                penalties.append((p, "Professional Penalty"))

        adj_prob = arbitrate(c["base_prob"], penalties, [n for _, n in penalties])
        results.append({
            **c,
            "adj_prob": adj_prob,
            "penalties_applied": [n for _, n in penalties],
            "n_penalties": len(penalties),
        })

    return results


def compute_kendall_tau(ranking_a, ranking_b):
    """Kendall's tau-b between two rankings.

    Uses the standard formula:
        tau_b = (C - D) / sqrt((C + D + T_A) * (C + D + T_B))
    where T_A = pairs tied in A only, T_B = pairs tied in B only.
    Pairs tied in BOTH are excluded from both T_A and T_B.
    """
    n = len(ranking_a)
    if n < 2:
        return 1.0

    concordant = 0
    discordant = 0
    ties_a_only = 0
    ties_b_only = 0

    for i in range(n):
        for j in range(i + 1, n):
            a_diff = ranking_a[i] - ranking_a[j]
            b_diff = ranking_b[i] - ranking_b[j]

            if a_diff == 0 and b_diff == 0:
                # Tied in both — excluded from denominator
                pass
            elif a_diff == 0:
                ties_a_only += 1
            elif b_diff == 0:
                ties_b_only += 1
            elif (a_diff > 0 and b_diff > 0) or (a_diff < 0 and b_diff < 0):
                concordant += 1
            else:
                discordant += 1

    denom = np.sqrt(
        (concordant + discordant + ties_a_only) *
        (concordant + discordant + ties_b_only)
    )
    if denom == 0:
        return 1.0
    return (concordant - discordant) / denom


def top_k_overlap(ranking_a, ranking_b, k):
    """Fraction of overlap in top-K between two rankings."""
    set_a = set(ranking_a[:k])
    set_b = set(ranking_b[:k])
    return len(set_a & set_b) / k


def compare_rankings(baseline_results, ablated_results, layer_name, k_values=(10, 20, 50, 100)):
    """Compare rankings between baseline and ablated pipeline output."""
    # Sort by adjusted probability descending
    baseline_sorted = sorted(baseline_results, key=lambda x: x["adj_prob"], reverse=True)
    ablated_sorted = sorted(ablated_results, key=lambda x: x["adj_prob"], reverse=True)

    # Get idx-based rankings
    baseline_order = [r["idx"] for r in baseline_sorted]
    ablated_order = [r["idx"] for r in ablated_sorted]

    tau = compute_kendall_tau(baseline_order, ablated_order)

    overlaps = {}
    for k in k_values:
        overlaps[f"top_{k}"] = top_k_overlap(baseline_order, ablated_order, k)

    # Mean absolute probability change
    prob_deltas = []
    base_prob_map = {r["idx"]: r["adj_prob"] for r in baseline_results}
    ablated_prob_map = {r["idx"]: r["adj_prob"] for r in ablated_results}
    for idx in base_prob_map:
        delta = abs(base_prob_map[idx] - ablated_prob_map.get(idx, base_prob_map[idx]))
        prob_deltas.append(delta)
    mean_abs_delta = float(np.mean(prob_deltas))

    # Find most affected cases (largest absolute prob changes)
    case_deltas = sorted(
        [(idx, abs(base_prob_map[idx] - ablated_prob_map.get(idx, base_prob_map[idx])))
         for idx in base_prob_map],
        key=lambda x: x[1],
        reverse=True,
    )

    # Map layer name to the penalty name used in penalties_applied
    layer_to_penalty_name = {
        "L1_GPA_Penalty": "GPA Penalty",
        "L2_Language_Penalty": "Language Penalty",
        "L3_Cross_Major_Penalty": "Cross Major Penalty",
        "L4_Faculty_Penalty": "Faculty Penalty",
        "L5_Professional_Penalty": "Professional Penalty",
    }
    target_penalty_name = layer_to_penalty_name.get(layer_name, layer_name)
    cases_with_penalty = sum(
        1 for r in baseline_results
        if target_penalty_name in r["penalties_applied"]
    )

    return {
        "layer": layer_name,
        "kendall_tau": round(tau, 4),
        "top_k_overlap": {k: round(v, 4) for k, v in overlaps.items()},
        "mean_abs_prob_delta": round(mean_abs_delta, 6),
        "cases_affected": cases_with_penalty,
        "cases_affected_pct": round(cases_with_penalty / len(baseline_results) * 100, 1),
        "most_affected_cases": [
            {
                "case_idx": idx,
                "prob_delta": round(delta, 4),
            }
            for idx, delta in case_deltas[:5]
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_ablation_matrix(all_comparisons, output_path):
    """Heatmap-style ablation matrix."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    layers = [c["layer"] for c in all_comparisons]
    labels = [LAYER_LABELS[LAYER_NAMES.index(l)] for l in layers]

    # Panel 1: Kendall's tau
    ax = axes[0]
    taus = [c["kendall_tau"] for c in all_comparisons]
    colors = ["#E74C3C" if t < 0.95 else "#F39C12" if t < 0.99 else "#2ECC71" for t in taus]
    bars = ax.barh(range(len(layers)), [1.0 - t for t in taus], color=colors, edgecolor="white")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("1 - Kendall τ (排序偏离度)", fontsize=11)
    ax.set_title("排序一致性 (Ranking Consistency)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    for i, (t, d) in enumerate(zip(taus, [1.0 - t for t in taus])):
        ax.text(d + 0.001, i, f"τ={t:.4f}", va="center", fontsize=9, fontweight="bold")

    # Panel 2: Mean |Δprob|
    ax = axes[1]
    deltas = [c["mean_abs_prob_delta"] for c in all_comparisons]
    bars = ax.barh(range(len(layers)), deltas, color="#3498DB", edgecolor="white")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |Δ Probability|", fontsize=11)
    ax.set_title("概率变化幅度 (Probability Impact)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    for i, d in enumerate(deltas):
        ax.text(d + 0.0002, i, f"{d:.4f}", va="center", fontsize=9)

    # Panel 3: Cases affected %
    ax = axes[2]
    pcts = [c["cases_affected_pct"] for c in all_comparisons]
    bars = ax.barh(range(len(layers)), pcts, color="#9B59B6", edgecolor="white")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("% of Cases Affected", fontsize=11)
    ax.set_title("影响范围 (Coverage)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    for i, p in enumerate(pcts):
        ax.text(p + 0.5, i, f"{p:.1f}%", va="center", fontsize=9)

    fig.suptitle("Ablation Study — 逐层关掉对排序和概率的影响", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] ablation_matrix.png saved → {output_path}")


def plot_topk_overlap(all_comparisons, output_path):
    """Top-K overlap across layers."""
    fig, ax = plt.subplots(figsize=(10, 5))
    k_values = ["top_10", "top_20", "top_50", "top_100"]
    k_labels = ["Top 10", "Top 20", "Top 50", "Top 100"]
    colors = ["#2ECC71", "#3498DB", "#F39C12", "#E74C3C"]

    x = np.arange(len(all_comparisons))
    width = 0.2

    for i, (k, label, color) in enumerate(zip(k_values, k_labels, colors)):
        values = [c["top_k_overlap"][k] for c in all_comparisons]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, edgecolor="white")
        for bar, val in zip(bars, values):
            if val < 1.0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", fontsize=7, rotation=90)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([LAYER_LABELS[LAYER_NAMES.index(c["layer"])] for c in all_comparisons],
                       fontsize=9)
    ax.set_ylabel("Overlap Rate", fontsize=11)
    ax.set_title("Top-K 推荐重叠率 (Recommendation Overlap)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncols=2)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] topk_overlap.png saved → {output_path}")


def plot_penalty_distribution(baseline_results, output_path):
    """Distribution of how many penalties each case receives."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    n_penalties = [r["n_penalties"] for r in baseline_results]
    ax = axes[0]
    counts = pd.Series(n_penalties).value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color="#3498DB", edgecolor="white")
    ax.set_xlabel("Number of Active Penalty Layers", fontsize=11)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_title("每个Case触发多少层惩罚", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(val), ha="center", fontsize=11, fontweight="bold")

    penalty_counts = {}
    for name in ["GPA Penalty", "Language Penalty", "Cross Major Penalty",
                 "Faculty Penalty", "Professional Penalty"]:
        count = sum(1 for r in baseline_results if name in " ".join(r["penalties_applied"]))
        penalty_counts[name] = count

    ax = axes[1]
    names = list(penalty_counts.keys())
    counts_list = list(penalty_counts.values())
    colors = ["#E74C3C", "#F39C12", "#3498DB", "#9B59B6", "#1ABC9C"]
    bars = ax.barh(range(len(names)), counts_list, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Cases Affected", fontsize=11)
    ax.set_title("各层触发数量", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, counts_list):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{val} ({val / len(baseline_results) * 100:.1f}%)",
                va="center", fontsize=9)

    fig.suptitle("惩罚层触发分布", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] penalty_distribution.png saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("V2: 五层概率调整链 Ablation Study")
    print("=" * 70)

    # ── Load & prepare ─────────────────────────────────────────────────────
    cases, gpa_mean, gpa_std, lang_mean, lang_std = load_and_prepare()

    # ── Baseline: all layers ON ─────────────────────────────────────────────
    print("\n[3/5] 运行 Baseline (全层开启)...")
    baseline = run_pipeline(cases, gpa_mean, gpa_std, lang_mean, lang_std)
    baseline_probs = [r["adj_prob"] for r in baseline]
    print(f"  Baseline 概率范围: [{min(baseline_probs):.4f}, {max(baseline_probs):.4f}]")
    print(f"  Baseline 均值: {np.mean(baseline_probs):.4f}")
    print(f"  触发惩罚的 case: {sum(1 for r in baseline if r['n_penalties'] > 0)} / {len(baseline)}")

    # ── Ablation: disable each layer ────────────────────────────────────────
    print("\n[4/5] 逐层关掉并比较...")
    all_comparisons = []
    for layer_name in LAYER_NAMES:
        disabled = {layer_name.split("_")[0]}
        ablated = run_pipeline(cases, gpa_mean, gpa_std, lang_mean, lang_std, disabled)
        comparison = compare_rankings(baseline, ablated, layer_name)
        all_comparisons.append(comparison)
        print(f"  {layer_name}: τ={comparison['kendall_tau']:.4f}, "
              f"|Δprob|={comparison['mean_abs_prob_delta']:.4f}, "
              f"affected={comparison['cases_affected_pct']:.1f}%")

    # ── Raw XGBoost (no layers) ────────────────────────────────────────────
    raw_results = [{
        **c,
        "adj_prob": c["base_prob"],
        "penalties_applied": [],
        "n_penalties": 0,
    } for c in cases]
    raw_comparison = compare_rankings(baseline, raw_results, "ALL_LAYERS_OFF")
    print(f"  ALL_LAYERS_OFF: τ={raw_comparison['kendall_tau']:.4f}, "
          f"|Δprob|={raw_comparison['mean_abs_prob_delta']:.4f}")

    # ── Generate outputs ────────────────────────────────────────────────────
    print("\n[5/5] 生成图表和报告...")

    # Ablation matrix
    matrix_path = os.path.join(OUTPUT_DIR, "ablation_matrix.png")
    plot_ablation_matrix(all_comparisons, matrix_path)

    # Top-K overlap
    overlap_path = os.path.join(OUTPUT_DIR, "topk_overlap.png")
    plot_topk_overlap(all_comparisons, overlap_path)

    # Penalty distribution
    dist_path = os.path.join(OUTPUT_DIR, "penalty_distribution.png")
    plot_penalty_distribution(baseline, dist_path)

    # Extreme cases: most affected per layer
    extreme_cases = {}
    for comp in all_comparisons:
        layer = comp["layer"]
        # Find cases that had this layer's penalty in baseline but not in ablation
        base_map = {r["idx"]: r for r in baseline}
        ablated = run_pipeline(cases, gpa_mean, gpa_std, lang_mean, lang_std,
                               {layer.split("_")[0]})
        ablated_map = {r["idx"]: r for r in ablated}

        case_impacts = []
        for idx in base_map:
            delta = abs(base_map[idx]["adj_prob"] - ablated_map[idx]["adj_prob"])
            if delta > 0.001:
                case_impacts.append({
                    "case_idx": idx,
                    "prob_delta": round(delta, 4),
                    "base_prob": round(base_map[idx]["base_prob"], 4),
                    "baseline_adj": round(base_map[idx]["adj_prob"], 4),
                    "ablated_adj": round(ablated_map[idx]["adj_prob"], 4),
                    "gpa": base_map[idx]["gpa"],
                    "language_score": round(base_map[idx]["language_score"], 4),
                    "target_uni": base_map[idx]["target_uni"],
                    "target_major": base_map[idx]["target_major"],
                    "similarity": round(base_map[idx]["similarity"], 4),
                    "is_out_of_scope": base_map[idx]["is_out_of_scope"],
                    "is_professional": base_map[idx]["is_professional"],
                    "internship_count": base_map[idx]["internship_count"],
                })

        case_impacts.sort(key=lambda x: x["prob_delta"], reverse=True)
        extreme_cases[layer] = case_impacts[:3]

    extreme_path = os.path.join(OUTPUT_DIR, "extreme_cases.json")
    with open(extreme_path, "w", encoding="utf-8") as f:
        json.dump(extreme_cases, f, ensure_ascii=False, indent=2)
    print(f"[OK] extreme_cases.json saved → {extreme_path}")

    # Full ablation report
    report = {
        "baseline": {
            "n_cases": len(baseline),
            "mean_adj_prob": round(float(np.mean([r["adj_prob"] for r in baseline])), 4),
            "mean_base_prob": round(float(np.mean([r["base_prob"] for r in baseline])), 4),
            "penalty_distribution": {
                str(k): int(v) for k, v in
                pd.Series([r["n_penalties"] for r in baseline]).value_counts().to_dict().items()
            },
        },
        "layer_comparisons": all_comparisons,
        "all_layers_off": raw_comparison,
        "extreme_cases_top3": extreme_cases,
    }
    report_path = os.path.join(OUTPUT_DIR, "ablation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] ablation_report.json saved → {report_path}")

    # ── Print summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Ablation 结果摘要")
    print("=" * 70)
    print(f"\n{'Layer':<30} {'Kendall τ':<12} {'|Δprob|':<12} {'Affected %':<12}")
    print("-" * 66)
    for comp in all_comparisons:
        print(f"{comp['layer']:<30} {comp['kendall_tau']:<12} "
              f"{comp['mean_abs_prob_delta']:<12} {comp['cases_affected_pct']:<12}")
    print(f"{'ALL LAYERS OFF':<30} {raw_comparison['kendall_tau']:<12} "
          f"{raw_comparison['mean_abs_prob_delta']:<12} {'100.0':<12}")

    # Key insights
    print(f"\n关键发现:")
    taus = [(c["layer"], c["kendall_tau"]) for c in all_comparisons]
    taus.sort(key=lambda x: x[1])
    worst = taus[0]

    most_affected = max(all_comparisons, key=lambda x: x["cases_affected_pct"])
    highest_impact = max(all_comparisons, key=lambda x: x["mean_abs_prob_delta"])

    print(f"  1. 影响排序最大的层: {worst[0]} (τ={worst[1]:.4f})")
    print(f"  2. 影响范围最广的层: {most_affected['layer']} ({most_affected['cases_affected_pct']}% cases)")
    print(f"  3. 平均概率变化最大的层: {highest_impact['layer']} "
          f"(|Δprob|={highest_impact['mean_abs_prob_delta']:.4f})")
    print(f"  4. 五层全关 vs 全开: τ={raw_comparison['kendall_tau']:.4f}")
    print(f"\n产物目录: {OUTPUT_DIR}")
    print("  - ablation_matrix.png     (放 portfolio)")
    print("  - topk_overlap.png        (Top-K 推荐稳定性)")
    print("  - penalty_distribution.png (惩罚触发分布)")
    print("  - ablation_report.json    (完整数据)")
    print("  - extreme_cases.json      (极端 case trace)")
    print("=" * 70)


if __name__ == "__main__":
    main()
