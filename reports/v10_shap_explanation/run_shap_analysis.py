"""
V10: SHAP Model Explanation Report — reproducible script.

Generates all PNGs + JSON data using shap.plots.* built-in plotting.
Requires shap_values.npz and shap_summary.json from scripts/compute_shap.py.

Usage:
    python reports/v10_shap_explanation/run_shap_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ml"))

OUT_DIR = Path(__file__).resolve().parent

from reports._common import setup_matplotlib_fonts

# ── Chinese font ──────────────────────────────────────
setup_matplotlib_fonts()

# ── Load data ─────────────────────────────────────────
npz = np.load(PROJECT_ROOT / "reports/shap_values.npz")
shap_vals = npz["values"]  # (N, F)
feature_names = list(npz["feature_names"])  # length F
X_eval = npz["X_eval"]  # (N, F) feature values for coloring
eval_indices = npz["eval_indices"]  # original row indices

with open(PROJECT_ROOT / "reports/shap_summary.json", encoding="utf-8") as f:
    summary = json.load(f)

n_features = len(feature_names)
n_samples = shap_vals.shape[0]

# ── Calibration params (extracted from model booster attributes) ──
import gzip as _gzip_mod
import json as _json_mod

import xgboost as _xgb_mod

_cal_booster = _xgb_mod.Booster()
_model_dir = PROJECT_ROOT / "src" / "ml" / "pre-trained_models"
_candidates = sorted(
    _model_dir.glob("xgboost_*.ubj.gz"), key=lambda p: p.stat().st_mtime, reverse=True
)
if not _candidates:
    _candidates = sorted(
        _model_dir.glob("xgboost_*.ubj"), key=lambda p: p.stat().st_mtime, reverse=True
    )
if _candidates:
    _model_path = str(_candidates[0])
    if _model_path.endswith(".gz"):
        with _gzip_mod.open(_model_path, "rb") as _f:
            _cal_booster.load_model(bytearray(_f.read()))
    else:
        _cal_booster.load_model(_model_path)
    _cal_raw = _cal_booster.attr("calibration_params")
    _cal_parsed = _json_mod.loads(_cal_raw) if _cal_raw else {}
    a_cal = float(_cal_parsed.get("params", {}).get("a", -2.8715))
    b_cal = float(_cal_parsed.get("params", {}).get("b", 1.8980))
    _model_name = Path(_model_path).stem
else:
    a_cal, b_cal = -2.8715, 1.8980
    _model_name = "unknown"
print(f"  Model: {_model_name} | Calibration: a={a_cal:.4f}, b={b_cal:.4f}")

base_logodds = summary["expected_value"]
base_prob = 1 / (1 + np.exp(a_cal * base_logodds + b_cal))

# ── Build shap.Explanation object ─────────────────────
explanation = shap.Explanation(
    values=shap_vals,
    base_values=np.full(n_samples, base_logodds),
    data=X_eval,
    feature_names=feature_names,
)


# ═══════════════════════════════════════════════════════
# Figure 1: SHAP Bar vs XGBoost Gain comparison
# ═══════════════════════════════════════════════════════
def fig1_importance_comparison():
    import xgboost as xgb
    from scipy.stats import spearmanr

    booster = xgb.Booster()
    model_dir = PROJECT_ROOT / "src/ml/pre-trained_models"
    candidates = sorted(
        model_dir.glob("xgboost_*.ubj.gz"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        candidates = sorted(
            model_dir.glob("xgboost_*.ubj"), key=lambda p: p.stat().st_mtime, reverse=True
        )
    model_path = (
        str(candidates[0]) if candidates else str(model_dir / "xgboost_20260316_092608.ubj")
    )
    if model_path.endswith(".gz"):
        import gzip

        with gzip.open(model_path, "rb") as f:
            booster.load_model(bytearray(f.read()))
    else:
        booster.load_model(model_path)
    gain_dict = booster.get_score(importance_type="gain")
    xgb_gain = np.array([gain_dict.get(fn, 0) for fn in feature_names])
    xgb_gain_norm = xgb_gain / xgb_gain.sum() * 100

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs_shap)

    shap_order = np.argsort(-mean_abs_shap)
    gain_order = np.argsort(-xgb_gain)
    rho, p = spearmanr(
        [list(gain_order).index(i) for i in shap_order],
        list(range(n_features)),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    colors = [
        "#E74C3C" if any(k in fn for k in ["university", "major"]) else "#3498DB"
        for fn in feature_names
    ]
    ax1.barh(
        [feature_names[i] for i in order],
        [mean_abs_shap[i] for i in order],
        color=[colors[i] for i in order],
        alpha=0.85,
    )
    ax1.set_xlabel("mean |SHAP| (log-odds)", fontsize=11)
    ax1.set_title("SHAP Feature Importance", fontsize=13, fontweight="bold")
    ax1.axvline(
        x=mean_abs_shap.mean(),
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"mean={mean_abs_shap.mean():.3f}",
    )
    ax1.legend(fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh(
        [feature_names[i] for i in order],
        [xgb_gain_norm[i] for i in order],
        color=[colors[i] for i in order],
        alpha=0.85,
    )
    ax2.set_xlabel("Gain (%)", fontsize=11)
    ax2.set_title("XGBoost Built-in (Gain)", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"SHAP vs XGBoost Gain — Spearman ρ = {rho:.3f} (p={p:.4f})\n"
        f"base prob = {base_prob:.1%} (calibrated), N = {n_samples:,}",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_importance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig1_importance_comparison.png (ρ={rho:.3f})")
    return {"spearman_rho": float(rho), "spearman_p": float(p)}


# ═══════════════════════════════════════════════════════
# Figure 2: SHAP Beeswarm (shap.summary_plot)
# ═══════════════════════════════════════════════════════
def fig2_beeswarm():
    shap.summary_plot(
        shap_vals,
        features=X_eval,
        feature_names=feature_names,
        max_display=10,
        show=False,
        plot_size=(14, 6),
    )
    fig = plt.gcf()
    fig.axes[0].set_title(
        f"SHAP Beeswarm — {n_samples:,} samples\nred = high feature value, blue = low",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(OUT_DIR / "fig2_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig2_beeswarm.png (shap.summary_plot)")


# ═══════════════════════════════════════════════════════
# Figure 3: SHAP Heatmap (shap.plots.heatmap)
# ═══════════════════════════════════════════════════════
def fig3_heatmap():
    # Subsample for heatmap readability
    n_heat = min(1000, n_samples)
    idx = np.random.default_rng(42).choice(n_samples, size=n_heat, replace=False)
    shap.plots.heatmap(
        explanation[idx],
        max_display=10,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.savefig(OUT_DIR / "fig3_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig3_heatmap.png (shap.plots.heatmap)")


# ═══════════════════════════════════════════════════════
# Figure 4: SHAP Bar (shap.plots.bar) — cleaner importance
# ═══════════════════════════════════════════════════════
def fig4_bar():
    shap.plots.bar(explanation, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    fig.axes[0].set_title(
        f"SHAP Bar — mean(|SHAP|) per feature, N={n_samples:,}",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(OUT_DIR / "fig4_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig4_bar.png (shap.plots.bar)")


# ═══════════════════════════════════════════════════════
# Figure 5: Waterfall — best & worst case (shap.plots.waterfall)
# ═══════════════════════════════════════════════════════
def fig5_waterfall():
    shap_sum = shap_vals.sum(axis=1)
    idx_best = int(np.argmax(shap_sum))
    idx_worst = int(np.argmin(shap_sum))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    for ax, idx, label in [
        (ax1, idx_best, "Most Pushed UP"),
        (ax2, idx_worst, "Most Pushed DOWN"),
    ]:
        plt.sca(ax)
        shap.plots.waterfall(explanation[idx], max_display=10, show=False)
        total = shap_sum[idx]
        cal_prob = 1 / (1 + np.exp(a_cal * (base_logodds + total) + b_cal))
        ax.set_title(
            f"{label} — SHAP sum={total:+.2f} → {cal_prob:.2%}",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        f"Waterfall: Extreme Cases (base = {base_prob:.2%})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_waterfall_extremes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig5_waterfall_extremes.png (shap.plots.waterfall ×2)")


# ═══════════════════════════════════════════════════════
# Figure 6: Dependence — top 2 features (shap.dependence_plot)
# ═══════════════════════════════════════════════════════
def fig6_dependence():
    order = np.argsort(-np.abs(shap_vals).mean(axis=0))
    top2 = order[:2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for ax, fi in [(ax1, top2[0]), (ax2, top2[1])]:
        plt.sca(ax)
        shap.dependence_plot(
            int(fi),
            shap_vals,
            features=X_eval,
            feature_names=feature_names,
            interaction_index=4,  # color by gpa
            show=False,
            ax=ax,
        )
        ax.set_title(
            f"{feature_names[fi]} — mean |SHAP| = {np.abs(shap_vals[:, fi]).mean():.3f}",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle("SHAP Dependence: Top 2 Features (colored by GPA)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_dependence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig6_dependence.png (shap.dependence_plot ×2)")


# ═══════════════════════════════════════════════════════
# Figure 7: SHAP Sum Distribution with calibration overlay
# ═══════════════════════════════════════════════════════
def fig7_shap_distribution():
    shap_sum = shap_vals.sum(axis=1)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.hist(shap_sum, bins=80, color="#3498DB", alpha=0.6, edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("SHAP Sum (log-odds)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)

    for p, color in [(25, "#E67E22"), (50, "#E74C3C"), (75, "#27AE60")]:
        val = np.percentile(shap_sum, p)
        ax1.axvline(
            x=val, color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=f"P{p}={val:+.2f}"
        )
    ax1.axvline(x=0, color="gray", linestyle="-", alpha=0.4, linewidth=1)
    ax1.legend(fontsize=9, loc="upper right")

    # Calibrated probability axis
    ax2 = ax1.twiny()
    prob_ticks = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.99]
    logodds_ticks = [(np.log(p / (1 - p)) - b_cal) / a_cal - base_logodds for p in prob_ticks]
    ax2.set_xticks(logodds_ticks)
    ax2.set_xticklabels([f"{p:.0%}" for p in prob_ticks], fontsize=8)
    ax2.set_xlabel("Calibrated Probability", fontsize=11)

    ax1.set_title(
        f"SHAP Sum Distribution — N={n_samples:,}\n"
        f"base={base_prob:.2%}, median={np.median(shap_sum):+.2f}, mean={shap_sum.mean():+.2f}",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_shap_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig7_shap_distribution.png")

    percentiles = {}
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = float(np.percentile(shap_sum, p))
        cal_p = float(1 / (1 + np.exp(a_cal * (base_logodds + val) + b_cal)))
        percentiles[f"P{p}"] = {"shap_sum": val, "calibrated_prob": cal_p}
    return percentiles


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
def main():
    print(f"V10 SHAP Report — using shap.plots.* built-ins → {OUT_DIR}")
    print(f"  Features: {n_features}, Samples: {n_samples}")
    print(f"  Base log-odds: {base_logodds:.4f} → calibrated prob: {base_prob:.4f}\n")

    rho_stats = fig1_importance_comparison()
    fig2_beeswarm()
    fig3_heatmap()
    fig4_bar()
    fig5_waterfall()
    fig6_dependence()
    shap_percentiles = fig7_shap_distribution()

    # ── Save JSON ─────────────────────────────────────
    mean_abs = np.abs(shap_vals).mean(axis=0)
    total_mas = float(mean_abs.sum())
    cat_total = float(
        sum(
            mean_abs[i]
            for i, fn in enumerate(feature_names)
            if any(k in fn for k in ["university", "major"])
        )
    )
    num_total = float(total_mas - cat_total)

    report = {
        "metadata": {
            "version": "V10",
            "date": "2026-05-14",
            "method": "SHAP TreeExplainer (tree_path_dependent)",
            "model": "xgboost_20260316_092608.ubj (375 trees, 10 features)",
            "samples": int(n_samples),
            "hardware": "RTX 5090 Laptop GPU, 24GB",
            "base_logodds": float(base_logodds),
            "base_calibrated_prob": float(base_prob),
            "calibration": {"method": "sigmoid", "a": a_cal, "b": b_cal},
            "shap_version": shap.__version__,
        },
        "feature_importance": [
            {
                "rank": int(i + 1),
                "feature": feature_names[fi],
                "mean_abs_shap": float(mean_abs[fi]),
                "pct_of_total": float(mean_abs[fi] / total_mas * 100),
                "tier": (
                    "categorical"
                    if any(k in feature_names[fi] for k in ["university", "major"])
                    else "numeric"
                ),
            }
            for i, fi in enumerate(np.argsort(-mean_abs))
        ],
        "tier_summary": {
            "categorical_total_pct": float(cat_total / total_mas * 100),
            "numeric_total_pct": float(num_total / total_mas * 100),
            "cat_to_num_ratio": float(cat_total / num_total),
        },
        "shap_sum_percentiles": shap_percentiles,
        "shap_vs_gain_spearman": rho_stats,
        "top_interactions": summary.get("top_interactions", []),
        "case_studies": summary.get("case_studies", {}),
    }

    json_path = OUT_DIR / "shap_v10_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  {json_path} saved")
    print("\nDone — all 7 figures use shap.plots.* built-ins.")


if __name__ == "__main__":
    main()
