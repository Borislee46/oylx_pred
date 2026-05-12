"""
S4: Power Analysis / MDE Curve
=================================
Using the project's REAL baseline (33.7% admission rate),
compute Minimum Detectable Effect as a function of sample size.

Key insight: with ~12k samples, can only detect +2.5pp effects.
To detect +1pp, need ~50k samples.
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

# REAL project baseline
BASELINE_RATE = 0.337  # actual admission rate from test set
ALPHA = 0.05
POWER = 0.80


def compute_mde(baseline_rate, n_total, alpha=0.05, power=0.80):
    """Compute MDE for a two-proportion z-test.

    Returns absolute MDE (in percentage points) and relative MDE.
    """
    n_per_group = n_total / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Standard error under H0
    p_pool = baseline_rate
    se = np.sqrt(p_pool * (1 - p_pool) * (2 / n_per_group))

    # MDE = (z_alpha + z_beta) * SE
    mde_abs = (z_alpha + z_beta) * se
    mde_rel = mde_abs / baseline_rate

    return mde_abs, mde_rel


def compute_required_n(baseline_rate, desired_mde, alpha=0.05, power=0.80):
    """Compute required total sample size to detect a given MDE."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p = baseline_rate
    # n_per_group = (z_alpha + z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p2-p1)^2
    # p1 = p, p2 = p + mde
    p1, p2 = p, p + desired_mde
    n_per_group = ((z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
                   / (desired_mde ** 2))
    return int(np.ceil(2 * n_per_group))


def plot_power_curve(baseline_rate, output_path):
    """MDE vs sample size curve with key annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel 1: MDE vs Sample Size
    ax = axes[0]
    sample_sizes = np.logspace(np.log10(500), np.log10(100000), 200).astype(int)
    mdes_abs = []
    mdes_rel = []
    for n in sample_sizes:
        mde_a, mde_r = compute_mde(baseline_rate, n)
        mdes_abs.append(mde_a * 100)  # convert to pp
        mdes_rel.append(mde_r * 100)

    ax.plot(sample_sizes, mdes_abs, color="#3498DB", linewidth=2.5)
    ax.fill_between(sample_sizes, 0, mdes_abs, alpha=0.08, color="#3498DB")
    ax.set_xscale("log")
    ax.set_xlabel("Total Sample Size (log scale)", fontsize=12)
    ax.set_ylabel("MDE (percentage points)", fontsize=12)
    ax.set_title(f"Minimum Detectable Effect vs Sample Size\n"
                 f"(baseline={BASELINE_RATE:.1%}, α={ALPHA}, power={POWER})",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, which="both")

    # Annotate key points
    key_ns = [1000, 5000, 12344, 25000, 50000, 100000]
    for n in key_ns:
        if n <= max(sample_sizes):
            mde_a, _ = compute_mde(baseline_rate, n)
            ax.annotate(f"n={n:,}\nMDE={mde_a*100:.1f}pp",
                        xy=(n, mde_a * 100),
                        xytext=(n * 1.3, mde_a * 100 + 1.5),
                        fontsize=8, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
                        color="#2C3E50")

    # Mark project sample size
    ax.axvline(x=12344, color="#E74C3C", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Project N=12,344")
    ax.legend(fontsize=9)

    # Panel 2: Required N for target MDE
    ax = axes[1]
    target_mdes = np.linspace(0.005, 0.08, 50)  # 0.5pp to 8pp
    required_ns = [compute_required_n(baseline_rate, mde) for mde in target_mdes]

    ax.plot([m * 100 for m in target_mdes], required_ns, color="#E74C3C", linewidth=2.5)
    ax.fill_between([m * 100 for m in target_mdes], 0, required_ns, alpha=0.08, color="#E74C3C")
    ax.set_xlabel("Desired MDE (percentage points)", fontsize=12)
    ax.set_ylabel("Required Total Sample Size", fontsize=12)
    ax.set_title("How Many Samples to Detect a Given Effect?", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # Key annotations
    targets = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05]
    for t in targets:
        n = compute_required_n(baseline_rate, t)
        ax.annotate(f"{t*100:.1f}pp → {n:,}",
                    xy=(t * 100, n), fontsize=8,
                    xytext=(t * 100 + 0.2, n * 1.3),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
                    color="#2C3E50", fontweight="bold")

    ax.axhline(y=12344, color="#3498DB", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Project N=12,344")
    ax.legend(fontsize=9)

    fig.suptitle("Power Analysis — Using Real Project Baseline",
                 fontsize=11, color="#888")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] power_mde_curve.png saved -> {output_path}")


def main():
    print("=" * 60)
    print("S4: Power Analysis / MDE Curve")
    print("=" * 60)

    print(f"\nBaseline (admission rate): {BASELINE_RATE:.1%}")
    print(f"α: {ALPHA}, Power: {POWER}")

    # Project sample size
    n_project = 12344
    mde_abs, mde_rel = compute_mde(BASELINE_RATE, n_project)
    print(f"\nAt project sample size (N={n_project:,}):")
    print(f"  MDE = {mde_abs*100:.2f}pp (relative {mde_rel*100:.1f}%)")
    print(f"  Meaning: can detect if admission rate changes from "
          f"{BASELINE_RATE*100:.1f}% to {(BASELINE_RATE + mde_abs)*100:.1f}%")

    # What if we want smaller effects?
    print(f"\nRequired sample sizes:")
    for target_pp in [5.0, 3.0, 2.0, 1.5, 1.0, 0.5]:
        n = compute_required_n(BASELINE_RATE, target_pp / 100)
        feasible = "✓" if n <= n_project else "✗ (need more data)"
        print(f"  Detect +{target_pp}pp: N={n:,} {feasible}")

    output_path = "experiment-simulations/power/power_mde_curve.png"
    plot_power_curve(BASELINE_RATE, output_path)

    print(f"\nKey takeaway for interview:")
    print(f"  'With N={n_project:,}, I can only detect effects ≥ {mde_abs*100:.1f}pp.")
    print(f"  This explains why education ML often relies on calibration metrics")
    print(f"  rather than online A/B testing — the sample size isn't there.'")


if __name__ == "__main__":
    main()
