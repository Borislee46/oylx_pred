"""
S2: Peeking Problem Simulation
================================
Demonstrate how "checking p-value daily" inflates Type I Error from 5% to ~25%.
With subgroup splitting, it can reach 45%+.

This is self-contained — no project dependencies needed.
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

N_SIMULATIONS = 2000
N_TOTAL = 2000
PEEK_INTERVAL = 100


def simulate_no_peeking(n_total, n_sims):
    """Fixed sample size: only check at the end."""
    false_positives = 0
    for _ in range(n_sims):
        group_a = np.random.binomial(1, 0.1, n_total)
        group_b = np.random.binomial(1, 0.1, n_total)  # same distribution
        _, p = stats.ttest_ind(group_a, group_b)
        if p < 0.05:
            false_positives += 1
    return false_positives / n_sims


def simulate_daily_peeking(n_total, n_sims, peek_interval):
    """Check p-value every N samples, stop if p<0.05."""
    false_positives = 0
    for _ in range(n_sims):
        group_a = np.random.binomial(1, 0.1, n_total)
        group_b = np.random.binomial(1, 0.1, n_total)
        stopped = False
        for n in range(peek_interval, n_total + 1, peek_interval):
            _, p = stats.ttest_ind(group_a[:n], group_b[:n])
            if p < 0.05:
                stopped = True
                break
        if stopped:
            false_positives += 1
    return false_positives / n_sims


def simulate_peeking_with_subgroups(n_total, n_sims, peek_interval, n_subgroups=3):
    """Check p-value daily + split into non-overlapping subgroups.

    Each subgroup is a disjoint, pre-defined slice of the data
    (e.g., by user segment: new users / returning users / power users).
    This matches real p-hacking patterns where analysts split by
    demographics, device type, or acquisition channel.
    """
    false_positives = 0
    for _ in range(n_sims):
        group_a = np.random.binomial(1, 0.1, n_total)
        group_b = np.random.binomial(1, 0.1, n_total)
        subgroup_size = n_total // n_subgroups
        stopped = False
        for n in range(peek_interval, n_total + 1, peek_interval):
            for s in range(n_subgroups):
                start = s * subgroup_size
                end = min((s + 1) * subgroup_size, n)  # Fixed: non-overlapping slices
                if end - start < 30:
                    continue
                _, p = stats.ttest_ind(group_a[start:end], group_b[start:end])
                if p < 0.05:
                    stopped = True
                    break
            if stopped:
                break
        if stopped:
            false_positives += 1
    return false_positives / n_sims


def plot_peeking_results(no_peek_rate, daily_peek_rate, subgroup_peek_rate, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Bar comparison
    ax = axes[0]
    categories = ["Fixed Sample\n(No Peeking)", "Daily Peeking", "Daily Peeking\n+ 3 Subgroups"]
    rates = [no_peek_rate, daily_peek_rate, subgroup_peek_rate]
    colors = ["#2ECC71", "#F39C12", "#E74C3C"]
    bars = ax.bar(categories, [r * 100 for r in rates], color=colors, edgecolor="white")
    ax.axhline(y=5, color="black", linestyle="--", linewidth=1.2, alpha=0.5,
               label="Nominal α = 5%")
    ax.set_ylabel("Actual Type I Error Rate (%)", fontsize=12)
    ax.set_title("Peeking Problem: Type I Error Inflation", fontsize=14, fontweight="bold")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate * 100:.1f}%", ha="center", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(rates) * 100 * 1.3)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Cumulative error over time
    ax = axes[1]
    sample_sizes = np.arange(100, N_TOTAL + 1, PEEK_INTERVAL)
    cumulative_errors = []
    peek_points = np.linspace(0, N_TOTAL, 300)

    for n_check in peek_points.astype(int):
        n_check = max(30, n_check)
        fp = 0
        for _ in range(500):
            a = np.random.binomial(1, 0.1, n_check)
            b = np.random.binomial(1, 0.1, n_check)
            _, p = stats.ttest_ind(a, b)
            if p < 0.05:
                fp += 1
        cumulative_errors.append(fp / 500)

    ax.plot(peek_points, [e * 100 for e in cumulative_errors], color="#E74C3C", linewidth=2)
    ax.axhline(y=5, color="black", linestyle="--", linewidth=1.2, alpha=0.5,
               label="α = 5%")
    ax.fill_between(peek_points, 0, [e * 100 for e in cumulative_errors],
                    alpha=0.1, color="#E74C3C")
    ax.set_xlabel("Sample Size (when you stop & check)", fontsize=11)
    ax.set_ylabel("Actual Type I Error Rate (%)", fontsize=11)
    ax.set_title("Error Rate if You Stop at Each Sample Size", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Peeking Problem Simulation — "
                 f"{N_SIMULATIONS} Simulations × N=0-{N_TOTAL}",
                 fontsize=11, color="#888", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] peeking_problem.png saved -> {output_path}")


def main():
    print("=" * 60)
    print("S2: Peeking Problem Simulation")
    print("=" * 60)

    print(f"\nRunning {N_SIMULATIONS} simulations each...")

    print("  Scenario A: Fixed sample (no peeking)...")
    no_peek = simulate_no_peeking(N_TOTAL, N_SIMULATIONS)
    print(f"    Type I Error = {no_peek * 100:.2f}%")

    print("  Scenario B: Daily peeking...")
    daily_peek = simulate_daily_peeking(N_TOTAL, N_SIMULATIONS, PEEK_INTERVAL)
    print(f"    Type I Error = {daily_peek * 100:.2f}%")

    print("  Scenario C: Daily peeking + subgroup splitting...")
    subgroup_peek = simulate_peeking_with_subgroups(N_TOTAL, N_SIMULATIONS, PEEK_INTERVAL)
    print(f"    Type I Error = {subgroup_peek * 100:.2f}%")

    print(f"\nInflation factor:")
    print(f"  Daily peeking:     {daily_peek / no_peek:.1f}x")
    print(f"  + Subgroups:       {subgroup_peek / no_peek:.1f}x")

    output_path = "experiment-simulations/peeking/peeking_problem.png"
    plot_peeking_results(no_peek, daily_peek, subgroup_peek, output_path)

    print(f"\nKey takeaway for interview:")
    print(f"  'I simulated peeking on my own data — with daily checks,")
    print(f"  Type I Error inflates from 5% to {daily_peek*100:.0f}%.")
    print(f"  With subgroup splitting on top, it reaches {subgroup_peek*100:.0f}%.")
    print(f"  This is why I pre-register analysis plans and fix sample sizes.'")


if __name__ == "__main__":
    main()
