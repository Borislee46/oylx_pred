"""
S3: Multiple Testing Demonstration
=====================================
Generate 20 "metrics" with random noise, show how raw p-values
identify false positives, and how Bonferroni / Benjamini-Hochberg correct them.

Self-contained — no project dependencies needed.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def simulate_experiment(n_metrics=20, n_per_group=5000, true_effect_indices=None):
    """Simulate an A/B test with N metrics.

    Most metrics have no true effect (H0 is true).
    `true_effect_indices` specifies which indices have a real effect (>0).
    """
    if true_effect_indices is None:
        true_effect_indices = []

    p_values = []
    is_true_effect = []

    for i in range(n_metrics):
        has_effect = i in true_effect_indices
        is_true_effect.append(has_effect)

        # Control group
        control = np.random.normal(0, 1, n_per_group)
        # Treatment group — some with real effect
        mu = 0.05 if has_effect else 0
        treatment = np.random.normal(mu, 1, n_per_group)

        _, p = stats.ttest_ind(treatment, control)
        p_values.append(p)

    return np.array(p_values), np.array(is_true_effect)


def apply_corrections(p_values, alpha=0.05):
    """Apply Bonferroni and Benjamini-Hochberg corrections."""
    n = len(p_values)

    # Bonferroni: reject if p < alpha / n
    bonf_threshold = alpha / n
    bonf_significant = p_values < bonf_threshold

    # Benjamini-Hochberg
    sorted_indices = np.argsort(p_values)
    bh_significant = np.zeros(n, dtype=bool)
    for rank, idx in enumerate(sorted_indices):
        bh_threshold = alpha * (rank + 1) / n
        if p_values[idx] <= bh_threshold:
            bh_significant[idx] = True
        else:
            break  # BH requires monotonicity — once one fails, all larger p fail

    return bonf_significant, bh_significant, bonf_threshold


def plot_multiple_testing(p_values, is_true_effect, bonf_sig, bh_sig, raw_sig, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    n = len(p_values)
    metric_indices = np.arange(1, n + 1)

    # Panel 1: Raw p-values with thresholds
    ax = axes[0, 0]
    colors = ["#2ECC71" if e else "#E74C3C" for e in is_true_effect]
    ax.bar(metric_indices, p_values, color=colors, edgecolor="white", alpha=0.8)
    ax.axhline(y=0.05, color="#F39C12", linestyle="--", linewidth=1.5,
               label="α = 0.05 (raw)")
    ax.axhline(y=0.05 / n, color="#E74C3C", linestyle=":", linewidth=1.5,
               label=f"Bonferroni: α/n = {0.05/n:.4f}")
    ax.set_xlabel("Metric #", fontsize=11)
    ax.set_ylabel("p-value", fontsize=11)
    ax.set_title("Raw p-values (green=real effect, red=no effect)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xticks(metric_indices)
    ax.grid(axis="y", alpha=0.3)

    # Annotate false positives
    fp_raw = sum(raw_sig & ~is_true_effect)
    fp_bonf = sum(bonf_sig & ~is_true_effect)
    fp_bh = sum(bh_sig & ~is_true_effect)
    ax.text(0.98, 0.95,
            f"False positives:\n"
            f"  Raw: {fp_raw}\n"
            f"  Bonferroni: {fp_bonf}\n"
            f"  BH: {fp_bh}",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    # Panel 2: Significant flags
    ax = axes[0, 1]
    y_positions = {"Raw (α=0.05)": 3, "BH (FDR=0.05)": 2, "Bonferroni": 1}
    for method, sig_array, y in [
        ("Raw (α=0.05)", raw_sig, 3),
        ("BH (FDR=0.05)", bh_sig, 2),
        ("Bonferroni", bonf_sig, 1),
    ]:
        for i, is_sig in enumerate(sig_array):
            if is_sig:
                color = "#2ECC71" if is_true_effect[i] else "#E74C3C"
                marker = "o" if is_true_effect[i] else "x"
                ax.scatter(i + 1, y, marker=marker, s=80 if is_true_effect[i] else 60,
                           color=color, edgecolors="white", linewidth=0.5, zorder=5)

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Bonferroni", "BH (FDR=0.05)", "Raw (α=0.05)"], fontsize=10)
    ax.set_xlabel("Metric #", fontsize=11)
    ax.set_title("Significant Results by Method (●=real, ×=false positive)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(metric_indices)
    ax.set_xlim(0.5, n + 0.5)
    ax.grid(axis="x", alpha=0.3)

    # Panel 3: BH procedure visualization
    ax = axes[1, 0]
    sorted_p = np.sort(p_values)
    ranks = np.arange(1, n + 1)
    bh_line = 0.05 * ranks / n

    ax.plot(ranks, sorted_p, "o-", color="#3498DB", linewidth=2, markersize=6,
            label="Sorted p-values")
    ax.plot(ranks, bh_line, "--", color="#E74C3C", linewidth=2,
            label="BH threshold: (i/n)·α")
    ax.fill_between(ranks, 0, bh_line, alpha=0.05, color="#E74C3C")
    ax.set_xlabel("Rank (i)", fontsize=11)
    ax.set_ylabel("p-value", fontsize=11)
    ax.set_title("Benjamini-Hochberg Procedure", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Find the last significant
    last_sig = 0
    for r, p in enumerate(sorted_p):
        if p <= bh_line[r]:
            last_sig = r + 1
    ax.axvline(x=last_sig, color="#2ECC71", linestyle=":", linewidth=1,
               alpha=0.7, label=f"Last significant: rank {last_sig}")
    ax.legend(fontsize=9)

    # Panel 4: Discovery comparison
    ax = axes[1, 1]
    methods = ["Raw\n(α=0.05)", "Bonferroni", "Benjamini-\nHochberg"]
    true_discoveries = [
        sum(raw_sig & is_true_effect),
        sum(bonf_sig & is_true_effect),
        sum(bh_sig & is_true_effect),
    ]
    false_discoveries = [
        sum(raw_sig & ~is_true_effect),
        sum(bonf_sig & ~is_true_effect),
        sum(bh_sig & ~is_true_effect),
    ]

    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width / 2, true_discoveries, width, color="#2ECC71",
                   edgecolor="white", label="True Discoveries")
    bars2 = ax.bar(x + width / 2, false_discoveries, width, color="#E74C3C",
                   edgecolor="white", label="False Positives")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Discovery Breakdown", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    for bar, val in zip(bars1, true_discoveries):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", fontsize=12, fontweight="bold")
    for bar, val in zip(bars2, false_discoveries):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", fontsize=12, fontweight="bold")

    fig.suptitle("Multiple Testing Correction — 20 Metrics, 3 Real Effects",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] multiple_testing.png saved -> {output_path}")


def main():
    print("=" * 60)
    print("S3: Multiple Testing Demonstration")
    print("=" * 60)

    np.random.seed(42)
    n_metrics = 20
    true_effect_indices = [2, 7, 15]  # 3 out of 20 have real effects

    p_values, is_true_effect = simulate_experiment(
        n_metrics=n_metrics,
        n_per_group=5000,
        true_effect_indices=true_effect_indices,
    )

    raw_sig = p_values < 0.05
    bonf_sig, bh_sig, bonf_threshold = apply_corrections(p_values, alpha=0.05)

    print(f"\nResults for {n_metrics} metrics (3 with real effects):")
    print(f"{'Method':<20} {'True Pos':<12} {'False Pos':<12} {'Precision':<12}")
    print("-" * 56)
    for method, sig in [("Raw (α=0.05)", raw_sig), ("Bonferroni", bonf_sig),
                         ("BH (FDR=0.05)", bh_sig)]:
        tp = sum(sig & is_true_effect)
        fp = sum(sig & ~is_true_effect)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"{method:<20} {tp:<12} {fp:<12} {prec:.3f}")

    output_path = "experiment-simulations/multiple_testing/multiple_testing.png"
    plot_multiple_testing(p_values, is_true_effect, bonf_sig, bh_sig, raw_sig, output_path)

    print(f"\nKey takeaway:")
    print(f"  Raw α=0.05 expects 1 FP by chance (20×0.05=1).")
    print(f"  Bonferroni: strict but low power — may miss real effects.")
    print(f"  BH: balances discovery vs false positives — preferred in practice.")


if __name__ == "__main__":
    main()
