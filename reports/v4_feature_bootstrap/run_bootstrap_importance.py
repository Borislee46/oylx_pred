"""
V4: 特征重要性 Bootstrap 分析
================================
1000 轮 bootstrap 采样训练，画特征排名箱线图 + 95% CI。
验证 target_major 的 top-1 地位是否稳定。

运行方式: python reports/v4_feature_bootstrap/run_bootstrap_importance.py
"""

import json
import os
import sys
import time
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
from xgboost import XGBClassifier

from src.machine_learning_models.data_config import (
    CATEGORICAL_COLUMNS,
    MONOTONE_INCREASING_WHITELIST,
    MONOTONE_DECREASING_WHITELIST,
    TARGET_COLUMN,
)
from src.machine_learning_models.data_loader import load_and_preprocess_data

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "src", "machine_learning_models", "data", "cases.feather")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── bootstrap config ──────────────────────────────────────────────────────
N_BOOTSTRAP = 1000
SAMPLE_FRAC = 0.8  # each bootstrap sample uses 80% of training data
N_ESTIMATORS = 150  # simplified for speed (production uses 375)
MAX_DEPTH = 6       # simplified (production uses 10)

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# Feature name mapping for display
FEATURE_LABELS = {
    "target_major": "目标专业\n(Target Major)",
    "background_university": "本科院校\n(Background Univ)",
    "language_score": "语言成绩\n(Language)",
    "gpa": "GPA",
    "award_count": "获奖数量\n(Awards)",
    "target_university": "目标院校\n(Target Univ)",
    "background_major": "本科专业\n(Background Major)",
    "research_count": "科研数量\n(Research)",
    "paper_count": "论文数量\n(Papers)",
    "internship_count": "实习数量\n(Internship)",
}


def build_monotone_constraints(feature_names):
    """Build monotonic constraints matching production config."""
    categorical_features = [col for col in CATEGORICAL_COLUMNS if col in feature_names]
    constraints = []
    for f in feature_names:
        if f in categorical_features:
            constraints.append(0)
        elif f in MONOTONE_INCREASING_WHITELIST:
            constraints.append(1)
        elif f in MONOTONE_DECREASING_WHITELIST:
            constraints.append(-1)
        else:
            constraints.append(0)
    return tuple(constraints)


def train_and_get_importance(X, y, feature_names, scale_pos_weight):
    """Train a simplified XGBoost and return feature importance dict."""
    monotone = build_monotone_constraints(feature_names)
    model = XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        enable_categorical=True,
        tree_method="hist",
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=0.1,
        subsample=0.87,
        colsample_bytree=0.92,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone,
        verbosity=0,
    )
    model.fit(X, y)
    importances = model.feature_importances_
    return {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}


def bootstrap_feature_importance(X_train, y_train, feature_names, scale_pos_weight,
                                 n_bootstrap=N_BOOTSTRAP, sample_frac=SAMPLE_FRAC):
    """Run bootstrap, return per-round feature importance and ranks."""
    n_samples = int(len(X_train) * sample_frac)
    all_importances = []  # list of {feature: importance}
    all_ranks = []        # list of {feature: rank}

    print(f"  训练集: {len(X_train)} 样本, bootstrap: {n_bootstrap} 轮 × {n_samples} 样本")
    start_time = time.time()

    for i in range(n_bootstrap):
        indices = np.random.choice(len(X_train), size=n_samples, replace=True)
        X_boot = X_train.iloc[indices]
        y_boot = y_train.iloc[indices]

        imp = train_and_get_importance(X_boot, y_boot, feature_names, scale_pos_weight)
        all_importances.append(imp)

        # Rank: 1 = most important
        sorted_features = sorted(imp, key=imp.get, reverse=True)
        rank = {f: r + 1 for r, f in enumerate(sorted_features)}
        all_ranks.append(rank)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (n_bootstrap - i - 1)
            print(f"  [{i + 1}/{n_bootstrap}] elapsed={elapsed:.0f}s, eta={eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"  完成! 总耗时 {elapsed:.0f}s ({elapsed / n_bootstrap:.2f}s/轮)")

    return all_importances, all_ranks


def compute_bootstrap_stats(all_importances, all_ranks, feature_names):
    """Compute summary statistics from bootstrap distributions."""
    stats = {}
    for f in feature_names:
        imps = [r[f] for r in all_importances]
        ranks = [r[f] for r in all_ranks]
        stats[f] = {
            "importance_mean": float(np.mean(imps)),
            "importance_std": float(np.std(imps, ddof=1)),
            "importance_p2_5": float(np.percentile(imps, 2.5)),
            "importance_p97_5": float(np.percentile(imps, 97.5)),
            "rank_mean": float(np.mean(ranks)),
            "rank_std": float(np.std(ranks, ddof=1)),
            "rank_median": float(np.median(ranks)),
            "rank_p2_5": float(np.percentile(ranks, 2.5)),
            "rank_p97_5": float(np.percentile(ranks, 97.5)),
        }

    # Sort by mean importance descending
    return dict(sorted(stats.items(), key=lambda x: x[1]["importance_mean"], reverse=True))


def detect_overlapping_features(stats, rank_ci_overlap_threshold=0.5):
    """Detect feature pairs where rank CIs overlap → statistically indistinguishable ranking."""
    features = list(stats.keys())
    overlapping_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            ci1_low = stats[f1]["rank_p2_5"]
            ci1_high = stats[f1]["rank_p97_5"]
            ci2_low = stats[f2]["rank_p2_5"]
            ci2_high = stats[f2]["rank_p97_5"]

            # Check overlap: only flag if overlap fraction exceeds threshold
            overlap = min(ci1_high, ci2_high) - max(ci1_low, ci2_low)
            span = max(ci1_high, ci2_high) - min(ci1_low, ci2_low)
            if span > 0 and overlap / span > rank_ci_overlap_threshold:
                overlapping_pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "rank_ci_1": f"[{ci1_low:.1f}, {ci1_high:.1f}]",
                    "rank_ci_2": f"[{ci2_low:.1f}, {ci2_high:.1f}]",
                    "overlap": round(overlap / span, 3),
                })

    overlapping_pairs.sort(key=lambda x: x["overlap"], reverse=True)
    return overlapping_pairs


def plot_rank_boxplot(stats, feature_names, output_path):
    """Feature rank boxplot (lower rank = more important)."""
    # Order features by median rank
    ordered = sorted(feature_names, key=lambda f: stats[f]["rank_median"])

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [FEATURE_LABELS.get(f, f) for f in ordered]
    colors = plt.cm.RdYlGn_r([stats[f]["rank_median"] / len(feature_names) for f in ordered])

    for i, f in enumerate(ordered):
        s = stats[f]
        # Draw box: P25 to P75
        box_y = i
        ax.barh(box_y, s["rank_p97_5"] - s["rank_p2_5"], height=0.5,
                left=s["rank_p2_5"], color=colors[i], alpha=0.3, edgecolor="none")
        ax.barh(box_y, s["rank_mean"] - s["rank_p2_5"], height=0.3,
                left=s["rank_p2_5"], color=colors[i], alpha=0.6, edgecolor="none")
        # Median marker
        ax.plot(s["rank_median"], box_y, "D", color="black", markersize=8,
                markeredgecolor="white", markeredgewidth=0.5)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Feature Importance Rank (1 = Most Important)", fontsize=12)
    ax.set_title(f"Feature Rank Stability — {N_BOOTSTRAP}-Round Bootstrap",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(len(feature_names) + 0.5, 0.5)
    ax.grid(axis="x", alpha=0.3)

    # Add CI annotations
    for i, f in enumerate(ordered):
        s = stats[f]
        ax.text(s["rank_p97_5"] + 0.3, i,
                f"[{s['rank_p2_5']:.1f}, {s['rank_p97_5']:.1f}]",
                va="center", fontsize=8, color="#555")

    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] feature_rank_box.png saved → {output_path}")


def plot_importance_boxplot(stats, feature_names, output_path):
    """Feature importance magnitude boxplot."""
    ordered = sorted(feature_names, key=lambda f: stats[f]["importance_mean"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [FEATURE_LABELS.get(f, f) for f in ordered]
    colors = plt.cm.Blues([0.3 + 0.7 * stats[f]["importance_mean"] / stats[ordered[0]]["importance_mean"]
                           for f in ordered])

    for i, f in enumerate(ordered):
        s = stats[f]
        ax.barh(i, s["importance_mean"], height=0.6, color=colors[i], edgecolor="white")
        # Error bar
        ax.errorbar(s["importance_mean"], i,
                    xerr=[[s["importance_mean"] - s["importance_p2_5"]],
                          [s["importance_p97_5"] - s["importance_mean"]]],
                    fmt="none", ecolor="black", capsize=3, linewidth=1)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Feature Importance (gain)", fontsize=12)
    ax.set_title(f"Feature Importance Magnitude with 95% CI — {N_BOOTSTRAP}-Round Bootstrap",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Value annotations
    for i, f in enumerate(ordered):
        s = stats[f]
        ax.text(s["importance_mean"] + 0.003, i,
                f"{s['importance_mean']:.3f} ±{s['importance_std']:.3f}",
                va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] feature_importance_box.png saved → {output_path}")


def plot_rank_correlation(all_ranks, feature_names, output_path):
    """Correlation matrix of feature ranks across bootstrap rounds."""
    rank_matrix = np.array([[r[f] for f in feature_names] for r in all_ranks])
    corr = np.corrcoef(rank_matrix.T)

    fig, ax = plt.subplots(figsize=(8, 7))
    labels = [FEATURE_LABELS.get(f, f).replace("\n", " ") for f in feature_names]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Feature Rank Correlation Across Bootstrap Rounds", fontsize=13, fontweight="bold")

    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if abs(corr[i, j]) > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] rank_correlation.png saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("V4: 特征重要性 Bootstrap 分析")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] 加载训练数据...")
    data = load_and_preprocess_data(DATA_PATH)
    X_train, X_test, y_train, y_test, feature_names, _, _, _ = data
    print(f"  训练集: {len(X_train)} 样本, 特征: {len(feature_names)}")

    # Compute scale_pos_weight
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"  scale_pos_weight: {scale_pos_weight:.4f} (neg={n_neg}, pos={n_pos})")

    # ── Bootstrap ──────────────────────────────────────────────────────────
    print(f"\n[2/4] 运行 {N_BOOTSTRAP} 轮 bootstrap...")
    all_importances, all_ranks = bootstrap_feature_importance(
        X_train, y_train, feature_names, scale_pos_weight
    )

    # ── Compute stats ──────────────────────────────────────────────────────
    print("\n[3/4] 计算 bootstrap 统计量...")
    stats = compute_bootstrap_stats(all_importances, all_ranks, feature_names)
    overlapping = detect_overlapping_features(stats)

    # Comparison with production importance
    prod_importance = {
        "target_major": 0.2438, "background_university": 0.1468, "language_score": 0.0926,
        "gpa": 0.0912, "award_count": 0.0778, "target_university": 0.0771,
        "background_major": 0.0759, "research_count": 0.0732, "paper_count": 0.0646,
        "internship_count": 0.0569,
    }

    print(f"\n{'Feature':<25} {'Importance':<14} {'Rank':<15} {'Rank 95% CI':<20} {'vs Prod':<10}")
    print("-" * 84)
    for f in stats:
        s = stats[f]
        prod_val = prod_importance.get(f, 0)
        delta = s["importance_mean"] - prod_val
        print(f"{f:<25} {s['importance_mean']:.4f} ±{s['importance_std']:.4f}  "
              f"{s['rank_median']:.1f} ({s['rank_mean']:.1f})     "
              f"[{s['rank_p2_5']:.1f}, {s['rank_p97_5']:.1f}]       "
              f"{delta:+.4f}")

    print(f"\n  重叠排名对 (不可区分):")
    for pair in overlapping[:5]:
        print(f"    {pair['feature_1']} vs {pair['feature_2']}: "
              f"CI{pair['rank_ci_1']} ∩ CI{pair['rank_ci_2']} "
              f"(overlap={pair['overlap']:.2f})")

    # ── Generate outputs ────────────────────────────────────────────────────
    print("\n[4/4] 生成图表和报告...")

    # Rank boxplot
    rank_path = os.path.join(OUTPUT_DIR, "feature_rank_box.png")
    plot_rank_boxplot(stats, feature_names, rank_path)

    # Importance boxplot
    imp_path = os.path.join(OUTPUT_DIR, "feature_importance_box.png")
    plot_importance_boxplot(stats, feature_names, imp_path)

    # Rank correlation
    corr_path = os.path.join(OUTPUT_DIR, "rank_correlation.png")
    plot_rank_correlation(all_ranks, feature_names, corr_path)

    # Report JSON
    report = {
        "method": f"{N_BOOTSTRAP}-round bootstrap feature importance",
        "bootstrap_config": {
            "n_rounds": N_BOOTSTRAP,
            "sample_frac": SAMPLE_FRAC,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
        },
        "feature_stats": {f: {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in s.items()}
                          for f, s in stats.items()},
        "overlapping_rank_pairs": overlapping,
        "production_comparison": {
            f: {
                "bootstrap_mean": round(stats[f]["importance_mean"], 4),
                "production_value": round(prod_importance.get(f, 0), 4),
                "delta": round(stats[f]["importance_mean"] - prod_importance.get(f, 0), 4),
            }
            for f in feature_names
        },
        "key_findings": {
            "top_feature": max(stats, key=lambda f: stats[f]["importance_mean"]),
            "most_stable_rank": min(stats, key=lambda f: stats[f]["rank_std"]),
            "least_stable_rank": max(stats, key=lambda f: stats[f]["rank_std"]),
            "overlapping_pairs_count": len(overlapping),
        },
    }
    report_path = os.path.join(OUTPUT_DIR, "bootstrap_importance.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] bootstrap_importance.json saved → {report_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("结果摘要")
    print("=" * 70)

    # Top 3
    top3 = sorted(stats, key=lambda f: stats[f]["rank_median"])[:3]
    print(f"\n  Top 3 最稳定特征:")
    for f in top3:
        s = stats[f]
        print(f"    {f}: rank median={s['rank_median']:.1f}, 95% CI [{s['rank_p2_5']:.1f}, {s['rank_p97_5']:.1f}]")

    # Overlapping
    if overlapping:
        print(f"\n  排名不可区分的特征对:")
        for pair in overlapping[:3]:
            print(f"    {pair['feature_1']} ↔ {pair['feature_2']} (重叠率={pair['overlap']:.2f})")
    else:
        print(f"\n  所有特征排名均可区分")
        # Find closest pair
        closest = None
        for i, f1 in enumerate(feature_names):
            for f2 in feature_names[i+1:]:
                diff = abs(stats[f1]["rank_median"] - stats[f2]["rank_median"])
                if closest is None or diff < closest["diff"]:
                    closest = {"f1": f1, "f2": f2, "diff": diff}
        if closest:
            print(f"  最接近的排名: {closest['f1']} vs {closest['f2']} ({closest['diff']:.1f} 位差)")

    # Stability check
    most_stable = min(stats, key=lambda f: stats[f]["rank_std"])
    least_stable = max(stats, key=lambda f: stats[f]["rank_std"])
    print(f"\n  最稳定排名: {most_stable} (std={stats[most_stable]['rank_std']:.2f})")
    print(f"  最不稳定排名: {least_stable} (std={stats[least_stable]['rank_std']:.2f})")

    print(f"\n产物目录: {OUTPUT_DIR}")
    print("  - feature_rank_box.png       (放 portfolio)")
    print("  - feature_importance_box.png (量值 + 95% CI)")
    print("  - rank_correlation.png       (排名相关性)")
    print("  - bootstrap_importance.json  (完整数据)")
    print("=" * 70)


if __name__ == "__main__":
    main()
