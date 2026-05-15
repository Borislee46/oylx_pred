"""
V8: Text Uplift Calibration Measurement
========================================
Measure the calibration impact of TF-IDF text uplift (+0~15% boost).
This is the one adjustment component never included in ECE ablation (V5 only
covered the 5-layer penalty chain).

Key questions:
  - Does text uplift improve or degrade ECE?
  - What is the uplift distribution (mean, range)?
  - Do cases WITH text have better calibration than cases WITHOUT?
  - How many cases pass the similarity gate?

Usage: python reports/v8_text_uplift_calibration/run_text_uplift_analysis.py
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
import numpy as np
import pandas as pd

from src.machine_learning_models.data_loader import load_and_preprocess_data
from src.utils.model_loader import _load_serialized_xgb
from sklearn.metrics import roc_auc_score

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

# Text uplift model paths (from config)
VECTORIZER_PATH = os.path.join(
    PROJECT_ROOT, "src", "machine_learning_models", "pre-trained_models", "tfidf_vectorizer.joblib"
)
CENTROIDS_PATH = os.path.join(
    PROJECT_ROOT, "src", "machine_learning_models", "pre-trained_models", "tfidf_centroids.npz"
)
WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT, "src", "machine_learning_models", "pre-trained_models", "text_uplift_weights.json"
)

# ── Faculty rules (same as V5) ───────────────────────────────────────────────
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

# ── Constants ─────────────────────────────────────────────────────────────────
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

MAX_TOTAL_PENALTY_RATIO = 0.70
PENALTY_DECAY_FACTOR = 0.85
ARBITRATION_MIN_PROBABILITY = 0.005

PENALTY_NAMES = ["GPA", "Language", "CrossMajor", "Faculty", "Professional"]

# Text uplift config
MAX_TOTAL_BOOST = 0.15
SIM_GATE_SUM_MIN = 0.10
SIM_GATE_MAX_MIN = 0.08
UPLIFT_SMOOTHING = 0.7
CAP_MIN_FACTOR = 0.10
CAP_QUALITY_GAMMA = 1.2
QUALITY_SCORE_MAX_WEIGHT = 0.7
QUALITY_SCORE_MEAN_WEIGHT = 0.3

# Canonical keys match LogitUpliftProvider (used for model centroids/weights).
# DataFrame columns use singular form (research_detail, not research_details).
CANONICAL_TEXT_KEYS = ("research_details", "award_details", "internship_details", "paper_details")
CANONICAL_COUNT_KEYS = ("research_count", "award_count", "internship_count", "paper_count")

# Actual column names in cases.feather
DF_TEXT_COLS = ("research_detail", "award_detail", "internship_detail", "paper_detail")
DF_COUNT_COLS = ("research_count", "award_count", "internship_count", "paper_count")

# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════════════════
# Penalty functions (same as V5)
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
# ECE / Brier
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        prob_mean = float(probs[mask].mean())
        actual_rate = float(labels[mask].mean())
        ece += n_b / len(probs) * abs(prob_mean - actual_rate)
    return ece


def compute_brier(probs, labels):
    return float(np.mean((probs - labels) ** 2))


# ═══════════════════════════════════════════════════════════════════════════════
# Text uplift (simplified production pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def _fast_entropy(text: str) -> float:
    if not text:
        return 0.0
    try:
        b = text.encode("utf-8")
    except UnicodeEncodeError:
        return 0.0
    if len(b) < 10:
        return 0.0
    counts = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(b)
    entropy = -np.sum(probs * np.log2(probs))
    byte_rich = float(np.clip(entropy / 5.0, 0.0, 1.0))
    n = len(text)
    if n >= 12:
        span = max(12.0, float(n ** 0.55))
        char_f = float(np.clip(len(set(text)) / span, 0.0, 1.0))
        byte_rich *= 0.35 + 0.65 * char_f
    return float(np.clip(byte_rich, 0.0, 1.0))


def _clip_probability(val: float) -> float:
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _bounded_fuse(base: float, bonus: float) -> float:
    base = _clip_probability(base)
    bonus = _clip_probability(bonus)
    return 1.0 - (1.0 - base) * (1.0 - bonus)


def load_text_uplift_models():
    """Load TF-IDF vectorizer, centroids, and weights."""
    import joblib
    vec = joblib.load(VECTORIZER_PATH)
    data = np.load(CENTROIDS_PATH)
    centroids = {}
    for k in data.files:
        v = np.asarray(data[k], dtype=np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        centroids[k] = v
    with open(WEIGHTS_PATH, encoding="utf-8") as f:
        weights_raw = json.load(f) or {}
    weights = (
        float(weights_raw.get("b", 0.0)),
        float(weights_raw.get("w_r", 0.0)),
        float(weights_raw.get("w_a", 0.0)),
        float(weights_raw.get("w_i", 0.0)),
        float(weights_raw.get("w_p", 0.0)),
        float(weights_raw.get("u_r", 0.0)),
        float(weights_raw.get("u_a", 0.0)),
        float(weights_raw.get("u_i", 0.0)),
        float(weights_raw.get("u_p", 0.0)),
    )
    return vec, centroids, weights


def compute_text_uplift(prob, text_fields, vec, centroids, weights):
    """Compute text uplift for a single case. Returns (boosted_prob, delta_logit, uplift_amount).

    text_fields uses canonical keys (research_details, etc.) matching the TF-IDF model.
    If text is empty or gate fails, returns (prob, 0.0, 0.0).
    """
    texts = {k: str(text_fields.get(k, "")).strip() for k in CANONICAL_TEXT_KEYS}
    counts = {}
    for k in CANONICAL_COUNT_KEYS:
        try:
            counts[k] = int(float(text_fields.get(k, 0) or 0))
        except (ValueError, TypeError):
            counts[k] = 0

    if all(not t for t in texts.values()):
        return prob, 0.0, 0.0

    # Vectorize and compute similarities
    text_list = [texts[k] for k in CANONICAL_TEXT_KEYS]
    X = vec.transform(text_list)
    sims = {}
    for idx, k in enumerate(CANONICAL_TEXT_KEYS):
        row = X.getrow(idx)
        if row.nnz == 0:
            sims[k] = 0.0
            continue
        centroid = centroids.get(k)
        if centroid is None or centroid.size == 0:
            sims[k] = 0.0
            continue
        dot_val = row.dot(centroid)
        sims[k] = _clip_probability(float(np.asarray(dot_val).flat[0]))

    # Gate
    sim_values = list(sims.values())
    sim_sum = sum(sim_values)
    sim_max = max(sim_values) if sim_values else 0.0
    if sim_sum < SIM_GATE_SUM_MIN or sim_max < SIM_GATE_MAX_MIN:
        return prob, 0.0, 0.0

    # Delta logit
    b = weights[0]
    text_w = weights[1:5]
    inter_w = weights[5:9]
    delta = float(b)

    for i, k in enumerate(CANONICAL_TEXT_KEYS):
        s = sims[k]
        if s <= 0:
            continue
        richness = _fast_entropy(texts[k])
        s_adj = float(s * richness)
        delta += text_w[i] * s_adj
        cnt = counts[CANONICAL_COUNT_KEYS[i]]
        if cnt > 0:
            delta += inter_w[i] * s_adj * np.log1p(cnt * richness)

    if delta <= 0:
        return prob, 0.0, 0.0

    effective_delta = delta * UPLIFT_SMOOTHING

    # Probability applier
    s_arr = np.array(sim_values)
    q_raw = QUALITY_SCORE_MAX_WEIGHT * np.max(s_arr) + QUALITY_SCORE_MEAN_WEIGHT * np.mean(s_arr)
    q_adj = q_raw ** max(1.0, CAP_QUALITY_GAMMA)
    cap_factor = min(1.0, max(CAP_MIN_FACTOR, q_adj))

    if prob < 0.1 or prob > 0.9:
        return prob, delta, 0.0

    logit_p = np.log(prob / (1.0 - prob))
    new_p = 1.0 / (1.0 + np.exp(-(logit_p + effective_delta)))

    scale = 1.0 - 2.0 * abs(prob - 0.5)
    cap = prob * (1.0 + MAX_TOTAL_BOOST * cap_factor * scale)

    boosted = min(new_p, cap, 1.0)
    uplift = boosted - prob

    return boosted, delta, uplift


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading (same split as V5)
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    print("[1/5] Loading data and model...")

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

    # Build case records with text fields
    print("\n[2/5] Building per-case features (with text)...")
    cases = []
    for i, (idx, row) in enumerate(test_orig.iterrows()):
        gpa_raw = pd.to_numeric(row.get("gpa"), errors="coerce")
        gpa = float(gpa_raw) if pd.notna(gpa_raw) else gpa_mean

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

        # Extract text fields (map from dataframe columns to canonical keys)
        text_fields = {}
        for ck, df_col in zip(CANONICAL_TEXT_KEYS, DF_TEXT_COLS):
            text_fields[ck] = str(row.get(df_col, ""))
        for ck, df_col in zip(CANONICAL_COUNT_KEYS, DF_COUNT_COLS):
            try:
                text_fields[ck] = float(row.get(df_col, 0) or 0)
            except (ValueError, TypeError):
                text_fields[ck] = 0.0

        has_text = any(
            str(row.get(df_col, "")).strip()
            and str(row.get(df_col, "")).strip().lower() not in ("无", "none", "nan", "", "暂无")
            for df_col in DF_TEXT_COLS
        )

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
            "base_prob": float(raw_probas[i]),
            "actual": int(row.get("admitted", 0)),
            "text_fields": text_fields,
            "has_text": has_text,
        })

    n_with_text = sum(1 for c in cases if c["has_text"])
    print(f"  Built {len(cases)} case records ({n_with_text} with text, "
          f"{len(cases) - n_with_text} without)")

    return cases, gpa_mean, gpa_std, lang_mean, lang_std


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_penalty_chain(cases, gpa_mean, gpa_std, lang_mean, lang_std):
    """Run 5-layer penalty chain (same as V5, without text uplift)."""
    results = []
    for c in cases:
        raw_penalties = [
            compute_gpa_penalty(c["gpa"], gpa_mean, gpa_std),
            compute_language_penalty(c["language_score"], lang_mean, lang_std),
            compute_cross_major_penalty(c["similarity"]),
            compute_faculty_penalty(c["is_out_of_scope"]),
            compute_professional_penalty(c["target_major"], c["internship_count"]),
        ]
        active = [(raw_penalties[i], PENALTY_NAMES[i]) for i in range(5) if raw_penalties[i] > 0]
        adj_prob, _ = arbitrate(c["base_prob"], active)
        results.append(adj_prob)
    return np.array(results)


def apply_text_uplift(cases, probs_no_text, vec, centroids, weights):
    """Apply text uplift on top of penalty-adjusted probabilities."""
    probs_with_text = probs_no_text.copy()
    uplifts = np.zeros(len(cases))
    deltas = np.zeros(len(cases))
    gated_pass = np.zeros(len(cases), dtype=bool)

    for i, c in enumerate(cases):
        if not c["has_text"]:
            continue
        boosted, delta, uplift = compute_text_uplift(
            probs_no_text[i], c["text_fields"], vec, centroids, weights
        )
        probs_with_text[i] = boosted
        uplifts[i] = uplift
        deltas[i] = delta
        gated_pass[i] = delta > 0

    return probs_with_text, uplifts, deltas, gated_pass


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_uplift_distribution(results_df, output_path):
    """Uplift distribution and calibration comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    has_text = results_df["has_text"]
    gated = results_df["gated_pass"]

    # Panel 1: Uplift amount histogram (only cases that passed gate)
    ax = axes[0, 0]
    uplift_vals = results_df.loc[gated, "uplift"]
    if len(uplift_vals) > 0:
        ax.hist(uplift_vals * 100, bins=40, color="#3498DB", edgecolor="white")
        ax.axvline(x=uplift_vals.mean() * 100, color="red", linestyle="--",
                   label=f"Mean = {uplift_vals.mean()*100:.2f}%")
    ax.set_xlabel("Uplift (percentage points)", fontsize=11)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_title(f"Text Uplift Distribution\n({gated.sum()} cases passed gate)", fontsize=12, fontweight="bold")
    if len(uplift_vals) > 0:
        ax.legend(fontsize=8)

    # Panel 2: ECE comparison — with text vs without text
    ax = axes[1, 0]
    labels_all = results_df["actual"].values

    # Build category list dynamically (handle edge case of zero text cases)
    cat_data = [("All Cases", np.ones(len(results_df), dtype=bool))]
    if has_text.sum() > 0:
        cat_data.append(("Has Text", has_text.values))
    cat_data.append(("No Text", ~has_text.values))

    ece_no_text = []
    ece_with_text = []
    categories = []
    for label, mask in cat_data:
        if mask.sum() > 0:
            categories.append(label)
            ece_no_text.append(compute_ece(
                results_df.loc[mask, "prob_no_text"].values, labels_all[mask]))
            ece_with_text.append(compute_ece(
                results_df.loc[mask, "prob_with_text"].values, labels_all[mask]))

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, ece_no_text, width, label="Without Text Uplift", color="#E74C3C", edgecolor="white")
    ax.bar(x + width/2, ece_with_text, width, label="With Text Uplift", color="#2ECC71", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("ECE", fontsize=11)
    ax.set_title("ECE: With vs Without Text Uplift", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    for i, (no, wi) in enumerate(zip(ece_no_text, ece_with_text)):
        ax.text(i - width/2, no + 0.001, f"{no:.4f}", ha="center", fontsize=9)
        ax.text(i + width/2, wi + 0.001, f"{wi:.4f}", ha="center", fontsize=9)

    # Panel 3: Gate statistics
    ax = axes[0, 1]
    n_total = len(results_df)
    n_has_text = has_text.sum()
    n_gated = gated.sum()
    n_no_text = n_total - n_has_text
    n_has_text_no_gate = n_has_text - n_gated
    gate_data = [n_gated, n_has_text_no_gate, n_no_text]
    gate_labels = [
        f"Passed Gate\n({n_gated}, {n_gated/n_total*100:.1f}%)",
        f"Has Text, Failed Gate\n({n_has_text_no_gate}, {n_has_text_no_gate/n_total*100:.1f}%)",
        f"No Text\n({n_no_text}, {n_no_text/n_total*100:.1f}%)",
    ]
    colors = ["#2ECC71", "#F39C12", "#95A5A6"]
    ax.pie(gate_data, labels=gate_labels, colors=colors, autopct="%1.1f%%",
           startangle=90, explode=(0.05, 0, 0))
    ax.set_title("Text Uplift Gate Coverage", fontsize=12, fontweight="bold")

    # Panel 4: Uplift vs base probability scatter
    ax = axes[1, 1]
    gated_mask = gated.values
    if gated_mask.sum() > 0:
        ax.scatter(
            results_df.loc[gated_mask, "prob_no_text"],
            results_df.loc[gated_mask, "uplift"] * 100,
            c=results_df.loc[gated_mask, "actual"].map({0: "#E74C3C", 1: "#2ECC71"}),
            alpha=0.5, s=15, edgecolors="none"
        )
    ax.set_xlabel("Probability (before uplift)", fontsize=11)
    ax.set_ylabel("Uplift (percentage points)", fontsize=11)
    ax.set_title("Uplift vs Base Probability\n(green=admitted, red=rejected)", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Text Uplift Calibration Analysis (V8)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] text_uplift_distribution.png → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("V8: Text Uplift Calibration Measurement")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    cases, gpa_mean, gpa_std, lang_mean, lang_std = load_and_prepare()

    # ── Run penalty chain (same as V5, no text) ──────────────────────────────
    print("\n[3/5] Running 5-layer penalty chain (without text uplift)...")
    probs_no_text = run_penalty_chain(cases, gpa_mean, gpa_std, lang_mean, lang_std)
    labels = np.array([c["actual"] for c in cases])

    ece_no_text = compute_ece(probs_no_text, labels)
    brier_no_text = compute_brier(probs_no_text, labels)
    print(f"  Without text uplift: ECE={ece_no_text:.4f}, Brier={brier_no_text:.4f}")
    print(f"  Mean prob: {probs_no_text.mean():.4f}, Actual rate: {labels.mean():.4f}")

    # ── Load text uplift models ──────────────────────────────────────────────
    print("\n[4/5] Loading text uplift models...")
    try:
        vec, centroids, weights = load_text_uplift_models()
        print(f"  TF-IDF vectorizer loaded ({len(centroids)} centroids, {len(weights)} weights)")
        models_loaded = True
    except (FileNotFoundError, OSError) as e:
        print(f"  WARNING: Text uplift models not found: {e}")
        print(f"  Vectorizer: {VECTORIZER_PATH}")
        print(f"  Centroids: {CENTROIDS_PATH}")
        print(f"  Weights: {WEIGHTS_PATH}")
        print("  → Text uplift measurement SKIPPED. Train models first with scripts/train_text_tfidf.py")
        models_loaded = False

    if models_loaded:
        # ── Apply text uplift ─────────────────────────────────────────────────
        print("\n  Applying text uplift to cases with text data...")
        probs_with_text, uplifts, deltas, gated_pass = apply_text_uplift(
            cases, probs_no_text, vec, centroids, weights
        )

        ece_with_text = compute_ece(probs_with_text, labels)
        brier_with_text = compute_brier(probs_with_text, labels)
        delta_ece = ece_no_text - ece_with_text
        print(f"  With text uplift:    ECE={ece_with_text:.4f}, Brier={brier_with_text:.4f}")
        print(f"  ΔECE (improvement):  {delta_ece:+.4f} "
              f"({'IMPROVES' if delta_ece > 0 else 'WORSENS'} calibration)")

        n_gated = gated_pass.sum()
        n_has_text = sum(1 for c in cases if c["has_text"])
        print(f"  Cases with text: {n_has_text} → passed gate: {n_gated} "
              f"({n_gated/max(n_has_text,1)*100:.1f}%)")

        if n_gated > 0:
            uplift_pct = uplifts[gated_pass] * 100
            print(f"  Uplift range: [{uplift_pct.min():.2f}%, {uplift_pct.max():.2f}%]")
            print(f"  Uplift mean: {uplift_pct.mean():.2f}%")

        # ── Stratified by text availability ───────────────────────────────────
        print(f"\n  Stratified ECE:")
        has_text_mask = np.array([c["has_text"] for c in cases])
        for label, mask in [("Has Text", has_text_mask), ("No Text", ~has_text_mask)]:
            if mask.sum() > 0:
                ece_no = compute_ece(probs_no_text[mask], labels[mask])
                ece_with = compute_ece(probs_with_text[mask], labels[mask])
                print(f"    {label} (n={mask.sum()}): "
                      f"ECE {ece_no:.4f} → {ece_with:.4f} (Δ={ece_no - ece_with:+.4f})")

        # ── AUC: discrimination comparison ─────────────────────────────────
        # Text uplift is designed to improve discrimination (ranking), not calibration.
        # ECE alone is insufficient — must also measure AUC.
        from sklearn.metrics import roc_auc_score
        auc_no_text = roc_auc_score(labels, probs_no_text)
        auc_with_text = roc_auc_score(labels, probs_with_text)
        delta_auc = auc_with_text - auc_no_text
        print(f"\n  Discrimination (AUC):")
        print(f"    Without text uplift: AUC={auc_no_text:.4f}")
        print(f"    With text uplift:    AUC={auc_with_text:.4f}")
        print(f"    ΔAUC: {delta_auc:+.4f} "
              f"({'IMPROVES' if delta_auc > 0.001 else 'WORSENS' if delta_auc < -0.001 else 'NEGLIGIBLE'} "
              f"discrimination)")

        # Brier skill score decomposition
        # BSS = 1 - Brier/Brier_baseline
        brier_baseline = float(np.mean((labels.mean() - labels) ** 2))
        bss_no = 1 - brier_no_text / brier_baseline
        bss_with = 1 - brier_with_text / brier_baseline
        print(f"    Brier Skill Score: {bss_no:.4f} → {bss_with:.4f} "
              f"(Δ={bss_with - bss_no:+.4f})")

        # Stratified AUC
        for label, mask in [("Has Text", has_text_mask), ("No Text", ~has_text_mask)]:
            if mask.sum() > 1:
                auc_no = roc_auc_score(labels[mask], probs_no_text[mask])
                auc_with = roc_auc_score(labels[mask], probs_with_text[mask])
                print(f"    {label} (n={mask.sum()}): AUC {auc_no:.4f} → {auc_with:.4f} "
                      f"(Δ={auc_with - auc_no:+.4f})")

        # ── Build results dataframe for visualization ─────────────────────────
        results_df = pd.DataFrame({
            "prob_no_text": probs_no_text,
            "prob_with_text": probs_with_text,
            "uplift": uplifts,
            "delta_logit": deltas,
            "has_text": has_text_mask,
            "gated_pass": gated_pass,
            "actual": labels,
        })

        # ── Visualizations ────────────────────────────────────────────────────
        print("\n[5/5] Generating visualizations...")
        plot_uplift_distribution(
            results_df,
            os.path.join(OUTPUT_DIR, "text_uplift_distribution.png")
        )

        # ── Save report ───────────────────────────────────────────────────────
        report = {
            "summary": {
                "n_cases": len(cases),
                "n_has_text": int(n_has_text),
                "n_gated_pass": int(n_gated),
                "ece_no_text": round(ece_no_text, 6),
                "ece_with_text": round(ece_with_text, 6),
                "delta_ece": round(delta_ece, 6),
                "brier_no_text": round(brier_no_text, 6),
                "brier_with_text": round(brier_with_text, 6),
                "auc_no_text": round(auc_no_text, 6),
                "auc_with_text": round(auc_with_text, 6),
                "delta_auc": round(delta_auc, 6),
                "brier_skill_no_text": round(bss_no, 6),
                "brier_skill_with_text": round(bss_with, 6),
                "uplift_mean_pct": round(float(uplifts[gated_pass].mean() * 100), 3) if n_gated > 0 else 0,
                "uplift_max_pct": round(float(uplifts.max() * 100), 3),
                "decision": (
                    "Text uplift IMPROVES both calibration and discrimination"
                    if delta_ece > 0.003 and delta_auc > 0.001
                    else "Text uplift IMPROVES calibration but not discrimination"
                    if delta_ece > 0.003 and abs(delta_auc) <= 0.001
                    else "Text uplift IMPROVES discrimination but not calibration"
                    if abs(delta_ece) <= 0.003 and delta_auc > 0.001
                    else "Text uplift has NEGLIGIBLE impact on both calibration and discrimination"
                    if abs(delta_ece) <= 0.003 and abs(delta_auc) <= 0.001
                    else "Text uplift WORSENS calibration"
                ),
            },
            "stratified": {},
        }

        for label, mask in [("Has Text", has_text_mask), ("No Text", ~has_text_mask)]:
            if mask.sum() > 0:
                report["stratified"][label] = {
                    "n": int(mask.sum()),
                    "ece_no_text": round(compute_ece(probs_no_text[mask], labels[mask]), 6),
                    "ece_with_text": round(compute_ece(probs_with_text[mask], labels[mask]), 6),
                    "actual_rate": round(float(labels[mask].mean()), 4),
                }

        report_path = os.path.join(OUTPUT_DIR, "text_uplift_ece.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[OK] text_uplift_ece.json → {report_path}")

        # ── Key findings ──────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("V8: Key Findings")
        print("=" * 70)
        direction = "IMPROVES" if delta_ece > 0 else "WORSENS"
        significance = (
            "SIGNIFICANT (|ΔECE| > 0.005)" if abs(delta_ece) > 0.005
            else "NEGLIGIBLE (|ΔECE| ≤ 0.005)"
        )
        auc_dir = "IMPROVES" if delta_auc > 0.001 else "WORSENS" if delta_auc < -0.001 else "UNCHANGED"
        print(f"""
  CALIBRATION IMPACT:
    Without text uplift: ECE={ece_no_text:.4f}, Brier={brier_no_text:.4f}
    With text uplift:    ECE={ece_with_text:.4f}, Brier={brier_with_text:.4f}
    ΔECE: {delta_ece:+.4f} → Text uplift {direction} calibration ({significance})

  DISCRIMINATION IMPACT:
    Without text uplift: AUC={auc_no_text:.4f}, BSS={bss_no:.4f}
    With text uplift:    AUC={auc_with_text:.4f}, BSS={bss_with:.4f}
    ΔAUC: {delta_auc:+.4f} → Text uplift {auc_dir} discrimination

  COVERAGE:
    Total cases: {len(cases)}
    Has text: {n_has_text} ({n_has_text/len(cases)*100:.1f}%)
    Passed gate: {n_gated} ({n_gated/max(n_has_text,1)*100:.1f}% of text cases)
    Mean uplift: {uplift_pct.mean():.2f}% (n={n_gated})""" if n_gated > 0 else "    No cases passed gate")

    else:
        # Models not available
        report = {
            "summary": {
                "n_cases": len(cases),
                "ece_no_text": round(ece_no_text, 6),
                "brier_no_text": round(brier_no_text, 6),
                "error": "Text uplift models not found. Train with scripts/train_text_tfidf.py first.",
            },
        }
        report_path = os.path.join(OUTPUT_DIR, "text_uplift_ece.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[OK] text_uplift_ece.json → {report_path} (models not available)")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("    - text_uplift_distribution.png")
    print("    - text_uplift_ece.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
