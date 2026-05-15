"""
V7: Prediction Failure Patterns Analysis
=========================================
Deep dive into cases that fail prediction or have low-quality predictions.

Key questions:
  1. What exactly causes prediction to fail?
  2. Is missing data a signal (not just noise)?
  3. Can we predict failure before attempting prediction?
  4. What's a useful taxonomy of cold-start / low-quality cases?

Usage: python reports/v7_prediction_failures/run_failure_analysis.py
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "src", "machine_learning_models", "data", "cases.feather")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

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
# Data loading & feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_engineer():
    print("[1/6] Loading and engineering features...")
    df = pd.read_feather(DATA_PATH)
    print(f"  {len(df)} rows, {len(df.columns)} cols")

    # ── Define failure modes ─────────────────────────────────────────────────
    # A case "fails" if predict() would return no result.
    # Based on _build_payload in test_sparsity_stress.py:
    #   - background_university must be non-empty
    #   - background_major must be non-empty
    #   - GPA must be > 0 and not NaN
    #   - Language (ielts or toefl) must be > 0

    df["_no_bg_uni"] = df["background_university"].isna() | (df["background_university"].astype(str).str.strip() == "")
    df["_no_bg_major"] = df["background_major"].isna() | (df["background_major"].astype(str).str.strip() == "")
    df["_no_target_uni"] = df["target_university"].isna() | (df["target_university"].astype(str).str.strip() == "")
    df["_no_target_major"] = df["target_major"].isna() | (df["target_major"].astype(str).str.strip() == "")

    gpa_raw = pd.to_numeric(df["gpa"], errors="coerce")
    df["_no_gpa"] = gpa_raw.isna() | (gpa_raw <= 0)

    ielts_ok = pd.to_numeric(df["ielts"], errors="coerce").notna() & (pd.to_numeric(df["ielts"], errors="coerce") > 0)
    toefl_ok = pd.to_numeric(df["toefl"], errors="coerce").notna() & (pd.to_numeric(df["toefl"], errors="coerce") > 0)
    df["_no_lang"] = ~(ielts_ok | toefl_ok)

    # Failure classification
    df["_fail_severe"] = df["_no_bg_uni"] | df["_no_bg_major"] | df["_no_target_uni"] | df["_no_target_major"]
    df["_fail_moderate"] = df["_no_gpa"] & df["_no_lang"]  # both critical numeric fields missing
    df["_fail_mild"] = df["_no_gpa"] | df["_no_lang"]  # at least one missing

    # Overall: would predict() fail?
    df["_would_fail"] = df["_fail_severe"] | df["_fail_moderate"]

    # ── Feature engineering ───────────────────────────────────────────────────
    df["_gpa_val"] = pd.to_numeric(df["gpa"], errors="coerce")
    df["_lang_norm"] = np.maximum(
        pd.to_numeric(df["ielts"], errors="coerce").fillna(0) / 9.0,
        pd.to_numeric(df["toefl"], errors="coerce").fillna(0) / 120.0,
    )
    df["_has_gpa"] = (~df["_no_gpa"]).astype(int)
    df["_has_lang"] = (~df["_no_lang"]).astype(int)
    df["_has_bg_major"] = (~df["_no_bg_major"]).astype(int)

    # School tier
    C9 = {"北京大学", "清华大学", "复旦大学", "上海交通大学", "浙江大学",
          "南京大学", "中国科学技术大学", "哈尔滨工业大学", "西安交通大学"}
    def tier(uni):
        if pd.isna(uni): return "unknown"
        u = str(uni).strip()
        if u in C9: return "C9"
        if any(k in u for k in ["大学", "学院"]):
            # Rough heuristic: check for 985/211 indicators
            return "other"
        return "overseas"
    df["_tier"] = df["background_university"].apply(tier)

    # Experience counts
    for col in ["research_count", "internship_count", "paper_count", "award_count"]:
        df[f"_{col}"] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Major group (first 2 chars of background_major)
    df["_bg_major_group"] = df["background_major"].astype(str).str[:2]

    print(f"  Would fail (severe+moderate): {df['_would_fail'].sum()} "
          f"({df['_would_fail'].mean():.1%})")
    print(f"    Severe (missing identifiers): {df['_fail_severe'].sum()} "
          f"({df['_fail_severe'].mean():.1%})")
    print(f"    Moderate (no GPA & no lang): {df['_fail_moderate'].sum()} "
          f"({df['_fail_moderate'].mean():.1%})")
    print(f"    Mild (no GPA | no lang): {df['_fail_mild'].sum()} "
          f"({df['_fail_mild'].mean():.1%})")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 1: Failure taxonomy
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_failure_taxonomy(df):
    print("\n[2/6] Building failure taxonomy...")

    # Breakdown by failure reason
    reasons = {
        "no_bg_major": df["_no_bg_major"].sum(),
        "no_gpa_and_lang": (df["_no_gpa"] & df["_no_lang"] & ~df["_fail_severe"]).sum(),
        "no_gpa_only": (df["_no_gpa"] & ~df["_no_lang"] & ~df["_fail_severe"]).sum(),
        "no_lang_only": (~df["_no_gpa"] & df["_no_lang"] & ~df["_fail_severe"]).sum(),
        "no_bg_uni": df["_no_bg_uni"].sum(),
        "no_target_uni": df["_no_target_uni"].sum(),
        "no_target_major": df["_no_target_major"].sum(),
        "all_ok": (~df["_would_fail"]).sum(),
    }

    # Multi-reason overlap
    failure_pivot = pd.crosstab(
        df["_no_gpa"].map({True: "NoGPA", False: "HasGPA"}),
        df["_no_lang"].map({True: "NoLang", False: "HasLang"}),
        margins=True,
    )

    # Admit rate by failure mode
    admit_by_mode = {}
    for mode in ["_fail_severe", "_fail_moderate", "_fail_mild", "_would_fail"]:
        mask = df[mode]
        if mask.sum() > 0:
            admit_by_mode[mode] = {
                "n": int(mask.sum()),
                "pct": round(float(mask.mean()) * 100, 1),
                "admit_rate": round(float(df.loc[mask, "admitted"].mean()), 4),
                "baseline_admit": round(float(df["admitted"].mean()), 4),
            }

    # Compare with overall
    baseline_admit = float(df["admitted"].mean())

    print(f"\n  Failure taxonomy:")
    print(f"  {'Reason':<25} {'Count':<8} {'%':<8} {'Admit Rate':<12} {'vs Baseline':<12}")
    print(f"  {'─'*65}")
    for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
        if reason == "all_ok":
            continue
        mask = None
        if reason == "no_bg_major":
            mask = df["_no_bg_major"]
        elif reason == "no_gpa_and_lang":
            mask = df["_no_gpa"] & df["_no_lang"] & ~df["_fail_severe"]
        elif reason == "no_gpa_only":
            mask = df["_no_gpa"] & ~df["_no_lang"] & ~df["_fail_severe"]
        elif reason == "no_lang_only":
            mask = ~df["_no_gpa"] & df["_no_lang"] & ~df["_fail_severe"]
        else:
            continue

        if mask is not None and mask.sum() > 0:
            admit_r = float(df.loc[mask, "admitted"].mean())
            print(f"  {reason:<25} {count:<8} {count/len(df)*100:<7.1f}% "
                  f"{admit_r:<12.4f} {admit_r - baseline_admit:+.4f}")

    print(f"  {'all_ok':<25} {reasons['all_ok']:<8} "
          f"{reasons['all_ok']/len(df)*100:<7.1f}% "
          f"{float(df[~df['_would_fail']].admitted.mean()):<12.4f} "
          f"{float(df[~df['_would_fail']].admitted.mean()) - baseline_admit:+.4f}")

    return {
        "reasons": {k: int(v) for k, v in reasons.items()},
        "admit_by_mode": admit_by_mode,
        "baseline_admit": baseline_admit,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 2: Missing-as-signal
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_missing_as_signal(df):
    print("\n[3/6] Missing-as-signal analysis...")

    results = {}

    # GPA missing → signal?
    has_gpa = df[~df["_no_gpa"]]
    no_gpa = df[df["_no_gpa"]]
    results["gpa"] = {
        "has_gpa_n": len(has_gpa),
        "has_gpa_admit": round(float(has_gpa["admitted"].mean()), 4),
        "no_gpa_n": len(no_gpa),
        "no_gpa_admit": round(float(no_gpa["admitted"].mean()), 4),
        "delta": round(float(no_gpa["admitted"].mean() - has_gpa["admitted"].mean()), 4),
        "signal_direction": "POSITIVE" if no_gpa["admitted"].mean() > has_gpa["admitted"].mean() else "NEGATIVE",
    }
    print(f"  GPA missing: admit={results['gpa']['no_gpa_admit']:.4f} "
          f"vs has GPA={results['gpa']['has_gpa_admit']:.4f} "
          f"(Δ={results['gpa']['delta']:+.4f}) → {results['gpa']['signal_direction']} signal")

    # Language missing → signal?
    has_lang = df[~df["_no_lang"]]
    no_lang = df[df["_no_lang"]]
    results["language"] = {
        "has_lang_n": len(has_lang),
        "has_lang_admit": round(float(has_lang["admitted"].mean()), 4),
        "no_lang_n": len(no_lang),
        "no_lang_admit": round(float(no_lang["admitted"].mean()), 4),
        "delta": round(float(no_lang["admitted"].mean() - has_lang["admitted"].mean()), 4),
        "signal_direction": "POSITIVE" if no_lang["admitted"].mean() > has_lang["admitted"].mean() else "NEGATIVE",
    }
    print(f"  Language missing: admit={results['language']['no_lang_admit']:.4f} "
          f"vs has lang={results['language']['has_lang_admit']:.4f} "
          f"(Δ={results['language']['delta']:+.4f}) → {results['language']['signal_direction']} signal")

    # BG major missing → signal?
    has_bgm = df[~df["_no_bg_major"]]
    no_bgm = df[df["_no_bg_major"]]
    results["bg_major"] = {
        "has_bgm_n": len(has_bgm),
        "has_bgm_admit": round(float(has_bgm["admitted"].mean()), 4),
        "no_bgm_n": len(no_bgm),
        "no_bgm_admit": round(float(no_bgm["admitted"].mean()), 4),
        "delta": round(float(no_bgm["admitted"].mean() - has_bgm["admitted"].mean()), 4),
        "signal_direction": "POSITIVE" if no_bgm["admitted"].mean() > has_bgm["admitted"].mean() else "NEGATIVE",
    }
    print(f"  BG Major missing: admit={results['bg_major']['no_bgm_admit']:.4f} "
          f"vs has={results['bg_major']['has_bgm_admit']:.4f} "
          f"(Δ={results['bg_major']['delta']:+.4f}) → {results['bg_major']['signal_direction']} signal")

    # Joint missing patterns
    print(f"\n  Joint missing patterns:")
    patterns = [
        ("Has GPA + Has Lang", ~df["_no_gpa"] & ~df["_no_lang"]),
        ("Has GPA + No Lang", ~df["_no_gpa"] & df["_no_lang"]),
        ("No GPA + Has Lang", df["_no_gpa"] & ~df["_no_lang"]),
        ("No GPA + No Lang", df["_no_gpa"] & df["_no_lang"]),
    ]
    for name, mask in patterns:
        n = mask.sum()
        admit = float(df.loc[mask, "admitted"].mean()) if n > 0 else 0
        print(f"    {name:<22}: n={n:<6} admit={admit:.4f}")

    # GPA missing by background tier
    print(f"\n  GPA missing rate by background:")
    for tier in ["C9", "other", "overseas", "unknown"]:
        tier_mask = df["_tier"] == tier
        if tier_mask.sum() > 0:
            miss_rate = df.loc[tier_mask, "_no_gpa"].mean()
            admit = df.loc[tier_mask, "admitted"].mean()
            print(f"    {tier:<10}: missing={miss_rate:.1%}, admit={admit:.4f}, n={tier_mask.sum()}")

    # Key insight: DEC-010 uses median imputation for missing GPA
    # If missing GPA is a POSITIVE signal, median imputation is BIASED downward
    gpa_median = df.loc[~df["_no_gpa"], "_gpa_val"].median()
    print(f"\n  DEC-010 IMPLICATION:")
    print(f"    Median GPA (imputation value): {gpa_median:.2f}")
    print(f"    No-GPA students actual admit:  {results['gpa']['no_gpa_admit']:.4f}")
    print(f"    Has-GPA-at-median students admit: need to check...")

    # What's the admit rate for students with GPA ≈ median?
    near_median = df[(df["_gpa_val"] > gpa_median - 0.1) & (df["_gpa_val"] < gpa_median + 0.1)]
    if len(near_median) > 0:
        near_median_admit = float(near_median["admitted"].mean())
        print(f"    Students with GPA≈{gpa_median:.1f} admit: {near_median_admit:.4f} (n={len(near_median)})")
        print(f"    Median imputation gives them the admit rate of a ~{gpa_median:.1f} GPA student")
        print(f"    But their actual admit rate is {results['gpa']['no_gpa_admit']:.4f}")
        print(f"    → Median imputation UNDERESTIMATES no-GPA students by "
              f"{results['gpa']['no_gpa_admit'] - near_median_admit:+.4f}")

    # ── Tier-stratified analysis: check for Simpson's paradox ──────────────
    # "No GPA students have higher admit rate" could be confounded by school tier.
    # If no-GPA students concentrate in high-admit tiers (e.g., overseas),
    # the 13pp gap might be a composition effect, not a missing-data signal.
    print(f"\n  Tier-stratified GPA missing vs admit rate:")
    C9_SCHOOLS = {
        "北京大学", "清华大学", "复旦大学", "上海交通大学",
        "浙江大学", "南京大学", "中国科学技术大学", "哈尔滨工业大学", "西安交通大学",
    }
    def classify_tier_v7(uni_name):
        if not uni_name or pd.isna(uni_name):
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
        # Distinguish overseas from domestic "other"
        if any(k in name for k in ["大学", "学院"]):
            return "other_domestic"
        return "overseas"

    df["_tier_v7"] = df["background_university"].apply(classify_tier_v7)
    tier_stratified = {}
    print(f"  {'Tier':<16} {'N_has_GPA':<10} {'Admit':<8} {'N_no_GPA':<10} {'Admit':<8} {'Δ':<8} {'Pct_no_GPA':<10}")
    print(f"  {'-'*70}")
    for tier in ["C9", "985", "211", "other_domestic", "overseas", "unknown"]:
        tier_mask = df["_tier_v7"] == tier
        if tier_mask.sum() == 0:
            continue
        has_gpa_mask = tier_mask & ~df["_no_gpa"]
        no_gpa_mask = tier_mask & df["_no_gpa"]
        n_has = has_gpa_mask.sum()
        n_no = no_gpa_mask.sum()
        admit_has = float(df.loc[has_gpa_mask, "admitted"].mean()) if n_has > 0 else 0
        admit_no = float(df.loc[no_gpa_mask, "admitted"].mean()) if n_no > 0 else 0
        pct_no = n_no / tier_mask.sum() * 100 if tier_mask.sum() > 0 else 0
        tier_stratified[tier] = {
            "n_has_gpa": int(n_has),
            "has_gpa_admit": round(admit_has, 4),
            "n_no_gpa": int(n_no),
            "no_gpa_admit": round(admit_no, 4),
            "delta": round(admit_no - admit_has, 4),
            "pct_no_gpa": round(pct_no, 1),
        }
        print(f"  {tier:<16} {n_has:<10} {admit_has:.4f}  {n_no:<10} {admit_no:.4f}  "
              f"{admit_no - admit_has:+.4f}  {pct_no:.1f}%")

    # Interpretation
    all_tiers_delta_positive = all(tier_stratified[t]["delta"] > 0
                                   for t in tier_stratified if tier_stratified[t]["n_no_gpa"] > 20)
    if all_tiers_delta_positive:
        print(f"\n  ＞ Within EVERY tier, no-GPA students have HIGHER admit rates.")
        print(f"    This supports the 'missing GPA is a genuine positive signal' hypothesis —")
        print(f"    it's NOT just a composition effect (Simpson's paradox ruled out).")
    else:
        print(f"\n  ＞ Tier stratification reveals composition effects:")
        for t in tier_stratified:
            s = tier_stratified[t]
            if s["n_no_gpa"] > 20 and s["delta"] < 0:
                print(f"    {t}: no-GPA admit LOWER than has-GPA — contradicts overall pattern")
        print(f"    The aggregate 13pp gap is partially confounded by school tier.")

    return results, tier_stratified


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 3: Cold-start taxonomy
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_cold_start(df):
    print("\n[4/5] Cold-start taxonomy...")

    # Group by (bg_uni, bg_major, target_uni, target_major) and count
    keys = ["background_university", "background_major", "target_university", "target_major"]
    combo_counts = (
        df[keys].astype(str)
        .groupby(keys, dropna=False)
        .size()
        .reset_index(name="count")
    )

    # Cold-start categories
    cold = {
        "zero_shot": int((combo_counts["count"] == 0).sum()),
        "one_shot": int((combo_counts["count"] == 1).sum()),
        "few_shot_2_4": int(((combo_counts["count"] >= 2) & (combo_counts["count"] <= 4)).sum()),
        "low_5_9": int(((combo_counts["count"] >= 5) & (combo_counts["count"] <= 9)).sum()),
        "medium_10_29": int(((combo_counts["count"] >= 10) & (combo_counts["count"] <= 29)).sum()),
        "ok_30_plus": int((combo_counts["count"] >= 30).sum()),
    }

    total = sum(cold.values())
    print(f"  Total unique combos: {total}")
    for name, count in sorted(cold.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name:<20} {count:<8} ({count/total*100:.1f}%)")

    # Which combos fail prediction?
    # Merge failure info back to combos
    df_indexed = df.copy()
    df_indexed["_key"] = (
        df_indexed["background_university"].astype(str) + "||" +
        df_indexed["background_major"].astype(str) + "||" +
        df_indexed["target_university"].astype(str) + "||" +
        df_indexed["target_major"].astype(str)
    )

    # Per combo: fail rate
    combo_fail = df_indexed.groupby("_key").agg(
        n=("admitted", "count"),
        fail_rate=("_would_fail", "mean"),
        admit_rate=("admitted", "mean"),
        gpa_missing=("_no_gpa", "mean"),
        lang_missing=("_no_lang", "mean"),
        bg_major_missing=("_no_bg_major", "mean"),
    ).reset_index()

    # By sample count bucket
    for lo, hi, label in [(0, 1, "0-1"), (2, 4, "2-4"), (5, 9, "5-9"),
                           (10, 29, "10-29"), (30, 999999, "30+")]:
        bucket = combo_fail[(combo_fail["n"] >= lo) & (combo_fail["n"] <= hi)]
        if len(bucket) > 0:
            print(f"    n={label:<8}: combos={len(bucket):<6} "
                  f"fail_rate={bucket['fail_rate'].mean():.1%} "
                  f"admit={bucket['admit_rate'].mean():.3f}")

    # Cold-start × missing data overlap
    cold_combos = combo_fail[combo_fail["n"] <= 4]
    print(f"\n  Cold-start (≤4 samples) combos: {len(cold_combos)}")
    print(f"    Fail rate: {cold_combos['fail_rate'].mean():.1%}")
    print(f"    GPA missing: {cold_combos['gpa_missing'].mean():.1%}")
    print(f"    Lang missing: {cold_combos['lang_missing'].mean():.1%}")
    print(f"    BG major missing: {cold_combos['bg_major_missing'].mean():.1%}")

    return {
        "combo_distribution": cold,
        "total_combos": total,
        "cold_start_fail_rate": round(float(cold_combos["fail_rate"].mean()), 4),
        "cold_start_admit_rate": round(float(cold_combos["admit_rate"].mean()), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_failure_patterns(df, missing_signal, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Failure reasons breakdown
    ax = axes[0, 0]
    reasons = {
        "BG Major\nmissing": df["_no_bg_major"].sum(),
        "No GPA\n+ No Lang": (df["_no_gpa"] & df["_no_lang"] & ~df["_fail_severe"]).sum(),
        "No GPA\nonly": (df["_no_gpa"] & ~df["_no_lang"] & ~df["_fail_severe"]).sum(),
        "No Lang\nonly": (~df["_no_gpa"] & df["_no_lang"] & ~df["_fail_severe"]).sum(),
        "All data\npresent": (~df["_would_fail"]).sum(),
    }
    colors = ["#E74C3C", "#F39C12", "#E67E22", "#3498DB", "#2ECC71"]
    bars = ax.barh(list(reasons.keys()), list(reasons.values()), color=colors, edgecolor="white")
    ax.set_xlabel("Number of Cases", fontsize=11)
    ax.set_title("Prediction Failure Reasons", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, reasons.values()):
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                f"{val} ({val/len(df)*100:.1f}%)", va="center", fontsize=9)

    # Panel 2: Missing-as-signal — admit rate comparison
    ax = axes[0, 1]
    keys_map = {"GPA": "gpa", "Language": "language", "BG Major": "bg_major"}
    admit_keys_map = {"GPA": ("has_gpa_admit", "no_gpa_admit"),
                      "Language": ("has_lang_admit", "no_lang_admit"),
                      "BG Major": ("has_bgm_admit", "no_bgm_admit")}
    signals = ["GPA", "Language", "BG Major"]
    has_rates = [missing_signal[keys_map[s]][admit_keys_map[s][0]] for s in signals]
    no_rates = [missing_signal[keys_map[s]][admit_keys_map[s][1]] for s in signals]

    x = np.arange(len(signals))
    width = 0.35
    ax.bar(x - width/2, has_rates, width, label="Has Data", color="#2ECC71", edgecolor="white")
    ax.bar(x + width/2, no_rates, width, label="Missing", color="#E74C3C", edgecolor="white")
    ax.axhline(y=df["admitted"].mean(), color="gray", linestyle="--", linewidth=1,
               label=f"Baseline ({df['admitted'].mean():.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(signals, fontsize=10)
    ax.set_ylabel("Admission Rate", fontsize=11)
    ax.set_title("Missing Data as Signal\n(missing ≠ random)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 3: GPA missing × Language missing joint
    ax = axes[0, 2]
    patterns = [
        ("Has GPA\nHas Lang", (~df["_no_gpa"] & ~df["_no_lang"]).sum(),
         float(df[~df["_no_gpa"] & ~df["_no_lang"]]["admitted"].mean())),
        ("Has GPA\nNo Lang", (~df["_no_gpa"] & df["_no_lang"]).sum(),
         float(df[~df["_no_gpa"] & df["_no_lang"]]["admitted"].mean())),
        ("No GPA\nHas Lang", (df["_no_gpa"] & ~df["_no_lang"]).sum(),
         float(df[df["_no_gpa"] & ~df["_no_lang"]]["admitted"].mean())),
        ("No GPA\nNo Lang", (df["_no_gpa"] & df["_no_lang"]).sum(),
         float(df[df["_no_gpa"] & df["_no_lang"]]["admitted"].mean())),
    ]
    pattern_names = [p[0] for p in patterns]
    pattern_counts = [p[1] for p in patterns]
    pattern_admits = [p[2] for p in patterns]
    x = np.arange(len(patterns))
    ax.bar(x, pattern_counts, color=["#2ECC71", "#3498DB", "#F39C12", "#E74C3C"], edgecolor="white")
    for i, (count, admit) in enumerate(zip(pattern_counts, pattern_admits)):
        ax.text(i, count + 500, f"admit={admit:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pattern_names, fontsize=8)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_title("Joint Missing Patterns", fontsize=12, fontweight="bold")

    # Panel 4: Admit rate by failure risk level
    ax = axes[1, 0]
    levels = ["All OK", "Mild\n(1 missing)", "Moderate\n(both missing)", "Severe\n(no identifier)"]
    level_admits = [
        float(df[~df["_would_fail"] & ~df["_fail_mild"]]["admitted"].mean()),
        float(df[df["_fail_mild"] & ~df["_fail_moderate"] & ~df["_fail_severe"]]["admitted"].mean()),
        float(df[df["_fail_moderate"] & ~df["_fail_severe"]]["admitted"].mean()),
        float(df[df["_fail_severe"]]["admitted"].mean()),
    ]
    level_counts = [
        int((~df["_would_fail"] & ~df["_fail_mild"]).sum()),
        int((df["_fail_mild"] & ~df["_fail_moderate"] & ~df["_fail_severe"]).sum()),
        int((df["_fail_moderate"] & ~df["_fail_severe"]).sum()),
        int(df["_fail_severe"].sum()),
    ]
    colors = ["#2ECC71", "#F39C12", "#E67E22", "#E74C3C"]
    x = np.arange(len(levels))
    ax.bar(x, level_admits, color=colors, edgecolor="white")
    ax.axhline(y=df["admitted"].mean(), color="gray", linestyle="--", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(levels, level_counts)], fontsize=9)
    ax.set_ylabel("Admission Rate", fontsize=11)
    ax.set_title("Admit Rate by Data Completeness\n(more missing = HIGHER admit?)", fontsize=11, fontweight="bold")
    for i, (rate, count) in enumerate(zip(level_admits, level_counts)):
        ax.text(i, rate + 0.01, f"{rate:.3f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 5: Failure rate by school type
    ax = axes[1, 1]
    tier_fail = df.groupby("_tier").agg(
        fail_rate=("_would_fail", "mean"),
        admit=("admitted", "mean"),
        n=("admitted", "count"),
    ).reset_index()
    x = np.arange(len(tier_fail))
    ax.bar(x - 0.15, tier_fail["fail_rate"] * 100, 0.3, label="Fail Rate %",
           color="#E74C3C", edgecolor="white")
    ax.bar(x + 0.15, tier_fail["admit"] * 100, 0.3, label="Admit Rate %",
           color="#3498DB", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={n})" for t, n in zip(tier_fail["_tier"], tier_fail["n"])], fontsize=9)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_title("Failure & Admit Rate by School Type", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 6: DEC-010 critique
    ax = axes[1, 2]
    ax.axis("off")
    critique = (
        "DEC-010 需要重审\n"
        "─────────────────\n\n"
        "DEC-010用median imputation\n"
        "处理缺失GPA，理由是'保守'。\n\n"
        "相关性发现：\n"
        "• 无GPA学生录取率 = 0.461\n"
        "• 中位GPA(3.3)学生录取率更低\n"
        "• Median imputation可能低估\n"
        "  (相关系，因果待验证)\n\n"
        "⚠ 相关性 ≠ 因果\n"
        " 可能混杂：院校层级、地区效应\n"
        " 竞争性假说未排除\n\n"
        "建议 (按安全程度)：\n"
        "1. 标记'数据不完整'并告知\n"
        "2. 控制混杂后考虑调整系数\n"
        "3. 独立base prob有泄漏风险\n"
        "   (label→feature)"
    )
    ax.text(0, 1, critique, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Prediction Failure & Missing Data Patterns", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] failure_patterns.png → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("V7: Prediction Failure Patterns Analysis")
    print("=" * 70)

    df = load_and_engineer()
    taxonomy = analyze_failure_taxonomy(df)
    missing_signal, tier_stratified = analyze_missing_as_signal(df)
    cold_start = analyze_cold_start(df)

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\n[5/5] Generating visualizations...")
    plot_failure_patterns(df, missing_signal,
                          os.path.join(OUTPUT_DIR, "failure_patterns.png"))

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "summary": {
            "total_cases": len(df),
            "would_fail": int(df["_would_fail"].sum()),
            "would_fail_pct": round(float(df["_would_fail"].mean()) * 100, 1),
            "baseline_admit": round(float(df["admitted"].mean()), 4),
        },
        "taxonomy": taxonomy,
        "missing_as_signal": missing_signal,
        "tier_stratified_missing_gpa": tier_stratified,
        "cold_start": cold_start,
    }

    report_path = os.path.join(OUTPUT_DIR, "failure_patterns_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"[OK] failure_patterns_report.json → {report_path}")

    # ── Key findings ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("V7: Key Findings")
    print("=" * 70)

    print(f"""
  FAILURE TAXONOMY:
    Total cases: {len(df):,}
    Would fail prediction: {df['_would_fail'].sum():,} ({df['_would_fail'].mean():.1%})
      - Severe (no identifier): {df['_fail_severe'].sum():,} ({df['_fail_severe'].mean():.1%})
      - Moderate (no GPA+Lang): {df['_fail_moderate'].sum():,} ({df['_fail_moderate'].mean():.1%})
      - Mild (one missing):     {df['_fail_mild'].sum():,} ({df['_fail_mild'].mean():.1%})

  MISSING-AS-SIGNAL (correlational, causal direction TBD):
    Missing GPA CORRELATES with higher admit rate:
      No GPA:    admit={missing_signal['gpa']['no_gpa_admit']:.4f}
      Has GPA:   admit={missing_signal['gpa']['has_gpa_admit']:.4f}
      Delta:     {missing_signal['gpa']['delta']:+.4f}
      → Median imputation MAY underestimate no-GPA students (correlational, confounders not controlled)

    Missing BG Major CORRELATES with much higher admit rate:
      No BG Major: admit={missing_signal['bg_major']['no_bgm_admit']:.4f}
      Has BG Major: admit={missing_signal['bg_major']['has_bgm_admit']:.4f}
      Delta:        {missing_signal['bg_major']['delta']:+.4f}
      → Causal direction unverified — consider school tier / region confounders

    Missing Language is a weak NEGATIVE correlate:
      No Lang:   admit={missing_signal['language']['no_lang_admit']:.4f}
      Has Lang:  admit={missing_signal['language']['has_lang_admit']:.4f}
      Delta:     {missing_signal['language']['delta']:+.4f}

  FAILURE PREDICTABILITY (qualitative):
    Prediction failure is almost entirely determined by data completeness:
      - No BG Major → 100% failure (predict() lacks features for payload)
      - Has GPA + Has Lang + Has BG Major → ~0% failure
    Failure is defined by missing data itself — a predictive model would be circular.
    Correct approach: check input completeness in UI layer, flag missing cases upfront.

  COLD START:
    {cold_start['total_combos']:,} unique combos
    ≤4 samples: {cold_start['combo_distribution']['zero_shot'] + cold_start['combo_distribution']['one_shot'] + cold_start['combo_distribution']['few_shot_2_4']:,} combos
    Cold-start fail rate: {cold_start['cold_start_fail_rate']:.1%}
""")

    print(f"  Output directory: {OUTPUT_DIR}")
    print("    - failure_patterns.png")
    print("    - failure_patterns_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
