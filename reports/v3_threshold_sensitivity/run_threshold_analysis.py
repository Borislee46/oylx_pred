"""
V3: 阈值敏感性分析 (Cost-based Threshold Sensitivity)
========================================================
证明 F1 最优 threshold ≠ 业务最优 threshold。
FP (推荐不该申的) 和 FN (漏掉能申的) 代价不同。

产出:
  - cost_threshold_curve.png     — cost ratio 0.2~5.0 的最优 threshold
  - precision_recall_tradeoff.png — PR 曲线 + threshold 标注
  - threshold_sensitivity.json   — 完整数据

运行方式: python reports/v3_threshold_sensitivity/run_threshold_analysis.py
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

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.machine_learning_models.data_loader import load_and_preprocess_data
from src.utils.model_loader import _load_serialized_xgb, _safe_json_loads, _wrap_with_calibration

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "src", "machine_learning_models", "data", "cases.feather")
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "src", "machine_learning_models", "pre-trained_models",
    "xgboost_20260316_092608.ubj",
)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

F1_THRESHOLD = 0.24

# ═══════════════════════════════════════════════════════════════════════════
# Core computation
# ═══════════════════════════════════════════════════════════════════════════

def total_cost(fp, fn, cost_ratio):
    """Total cost = cost_ratio * FP + FN.

    cost_ratio = cost_FP / cost_FN
    r < 1: FN is worse (漏掉机会 > 推荐错)
    r = 1: equal cost (same as F1 optimization)
    r > 1: FP is worse (推荐错 > 漏掉机会)
    """
    return cost_ratio * fp + fn


def scan_thresholds(y_true, probas, n_steps=201):
    """Scan all thresholds, compute metrics at each."""
    thresholds = np.linspace(0.0, 1.0, n_steps)
    results = []
    for t in thresholds:
        y_pred = (probas >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        positive_rate = (tp + fp) / len(y_true)

        results.append({
            "threshold": float(t),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "positive_rate": float(positive_rate),
        })
    return results


def find_optimal_thresholds(scan_results, cost_ratios):
    """For each cost ratio, find the threshold with minimum total cost."""
    optimal = {}
    for r in cost_ratios:
        best = None
        for row in scan_results:
            cost = total_cost(row["fp"], row["fn"], r)
            if best is None or cost < best["cost"]:
                best = {**row, "cost": cost, "cost_ratio": r}
        optimal[str(r)] = {
            "cost_ratio": r,
            "optimal_threshold": float(best["threshold"]),
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "fp": best["fp"],
            "fn": best["fn"],
            "positive_rate": best["positive_rate"],
            "total_cost": float(best["cost"]),
        }
    return optimal


# ═══════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════

def plot_cost_threshold_curve(scan_results, optimal, cost_ratios, output_path):
    """Main figure: cost ratio → optimal threshold + precision/recall."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: Cost ratio vs optimal threshold
    ax = axes[0, 0]
    ratios = [optimal[str(r)]["optimal_threshold"] for r in cost_ratios]
    colors = []
    for r in cost_ratios:
        if r < 1.0:
            colors.append("#E74C3C")  # FN worse → low threshold
        elif r == 1.0:
            colors.append("#F39C12")  # F1 optimal
        else:
            colors.append("#3498DB")  # FP worse → high threshold
    ax.bar(range(len(cost_ratios)), ratios, color=colors, edgecolor="white")
    ax.set_xticks(range(len(cost_ratios)))
    ax.set_xticklabels([str(r) for r in cost_ratios], fontsize=9)
    ax.set_xlabel("Cost Ratio (cost_FP / cost_FN)", fontsize=11)
    ax.set_ylabel("Optimal Threshold", fontsize=11)
    ax.set_title("不同代价比下的最优 Threshold", fontsize=12, fontweight="bold")
    ax.axhline(y=F1_THRESHOLD, color="#F39C12", linestyle="--", linewidth=1.5,
               label=f"F1最优 threshold={F1_THRESHOLD}")
    for i, (t, r) in enumerate(zip(ratios, cost_ratios)):
        offset = 0.008 if t < 0.9 else -0.04
        ax.text(i, t + offset, f"{t:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # Panel 2: Precision-Recall tradeoff with cost ratios
    ax = axes[0, 1]
    # Plot PR curve
    thresholds_sorted = sorted(scan_results, key=lambda x: x["threshold"])
    precisions = [r["precision"] for r in thresholds_sorted]
    recalls = [r["recall"] for r in thresholds_sorted]
    ax.plot(recalls, precisions, color="#2C3E50", linewidth=1.5, alpha=0.6, zorder=1)

    # Mark optimal points for each cost ratio
    ratio_colors = {r: c for r, c in zip(cost_ratios, colors)}
    for r in cost_ratios:
        opt = optimal[str(r)]
        ax.scatter(opt["recall"], opt["precision"], s=120,
                   color=ratio_colors[r], edgecolors="white", linewidth=1.5,
                   zorder=5, label=f"r={r} (t={opt['optimal_threshold']:.2f})")

    # Mark F1 optimal
    f1_row = min(scan_results, key=lambda x: abs(x["threshold"] - F1_THRESHOLD))
    ax.scatter(f1_row["recall"], f1_row["precision"], s=200, marker="*",
               color="#F39C12", edgecolors="black", linewidth=1,
               zorder=10, label=f"F1最优 t={F1_THRESHOLD}")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall 空间中的最优阈值点", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", ncols=2)
    ax.grid(alpha=0.3)

    # Panel 3: Precision / Recall / F1 vs Threshold
    ax = axes[1, 0]
    ts = [r["threshold"] for r in thresholds_sorted]
    ax.plot(ts, [r["precision"] for r in thresholds_sorted],
            color="#E74C3C", linewidth=2, label="Precision")
    ax.plot(ts, [r["recall"] for r in thresholds_sorted],
            color="#3498DB", linewidth=2, label="Recall")
    ax.plot(ts, [r["f1"] for r in thresholds_sorted],
            color="#2ECC71", linewidth=2, label="F1")
    ax.axvline(x=F1_THRESHOLD, color="#F39C12", linestyle="--", linewidth=1.5,
               label=f"F1最优={F1_THRESHOLD}")
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: Total Normalized Cost vs Threshold for each r
    ax = axes[1, 1]
    key_ratios = [0.2, 0.5, 1.0, 2.0, 5.0]
    for r in key_ratios:
        costs = [total_cost(row["fp"], row["fn"], r) for row in thresholds_sorted]
        costs_norm = np.array(costs) / max(costs)
        ax.plot(ts, costs_norm, linewidth=1.5, alpha=0.8, label=f"r={r}")
    for r in key_ratios:
        opt_t = optimal[str(r)]["optimal_threshold"]
        ax.axvline(x=opt_t, color="gray", alpha=0.2, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Normalized Total Cost", fontsize=11)
    ax.set_title("不同 r 下的归一化总代价曲线", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Threshold 敏感性分析 — F1 最优 ≠ 业务最优",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] cost_threshold_curve.png saved → {output_path}")


def plot_cost_precision_recall_3d(optimal, cost_ratios, output_path):
    """Simplified: cost ratio → precision + recall dual-axis chart."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = np.arange(len(cost_ratios))
    precisions = [optimal[str(r)]["precision"] for r in cost_ratios]
    recalls = [optimal[str(r)]["recall"] for r in cost_ratios]
    thresholds = [optimal[str(r)]["optimal_threshold"] for r in cost_ratios]

    # Bar for threshold
    bars_t = ax1.bar(x - 0.25, thresholds, 0.25, color="#95A5A6", edgecolor="white",
                     label="Optimal Threshold")
    # Bar for precision
    bars_p = ax1.bar(x, precisions, 0.25, color="#E74C3C", edgecolor="white",
                     label="Precision")
    # Bar for recall
    bars_r = ax1.bar(x + 0.25, recalls, 0.25, color="#3498DB", edgecolor="white",
                     label="Recall")

    # Annotate values on bars
    for i, (t, p, rec) in enumerate(zip(thresholds, precisions, recalls)):
        ax1.text(i - 0.25, t + 0.02, f"{t:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax1.text(i, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ax1.text(i + 0.25, rec + 0.02, f"{rec:.2f}", ha="center", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"r={r}" for r in cost_ratios], fontsize=10)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("不同 Cost Ratio 下的最优 Threshold / Precision / Recall",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    # Add annotations explaining the zones
    ax1.axvspan(-0.5, 2.5, alpha=0.03, color="#E74C3C")
    ax1.axvspan(5.5, 9.5, alpha=0.03, color="#3498DB")
    ax1.text(1.0, 0.98, "漏掉机会更严重\n→ 低threshold\n→ 高recall",
             ha="center", fontsize=9, color="#E74C3C", va="top")
    ax1.text(7.5, 0.98, "推荐错更严重\n→ 高threshold\n→ 高precision",
             ha="center", fontsize=9, color="#3498DB", va="top")

    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] precision_recall_tradeoff.png saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("V3: 阈值敏感性分析 (Cost-based Threshold)")
    print("=" * 70)

    # ── Load data and calibrated predictions ──────────────────────────────
    print("\n[1/4] 加载数据和模型...")
    _, X_test, _, y_test, _, _, _, _ = load_and_preprocess_data(DATA_PATH)
    y_true = y_test.values.astype(int)
    print(f"  测试集: {len(y_true)} 样本, 正例率: {y_true.mean():.4f}")

    raw_model = _load_serialized_xgb(MODEL_PATH)
    booster = raw_model.get_booster()
    calibration = _safe_json_loads(booster.attr("calibration_params"), "calibration_params")
    prediction_threshold_raw = _safe_json_loads(
        booster.attr("prediction_threshold"), "prediction_threshold"
    )
    threshold_prod = 0.24
    if prediction_threshold_raw:
        try:
            threshold_prod = float(prediction_threshold_raw)
        except (TypeError, ValueError):
            pass
    calibrated_model = _wrap_with_calibration(raw_model, calibration, threshold_prod)
    probas = calibrated_model.predict_proba(X_test)[:, 1]
    print(f"  Calibrated probas: mean={probas.mean():.4f}, "
          f"range=[{probas.min():.4f}, {probas.max():.4f}]")

    # ── Scan all thresholds ────────────────────────────────────────────────
    print("\n[2/4] 扫描 201 个 threshold...")
    scan_results = scan_thresholds(y_true, probas, n_steps=201)
    print(f"  {len(scan_results)} thresholds scanned")

    # Find F1 optimal for reference
    best_f1 = max(scan_results, key=lambda x: x["f1"])
    print(f"  F1 optimal: threshold={best_f1['threshold']:.2f}, "
          f"P={best_f1['precision']:.4f}, R={best_f1['recall']:.4f}, "
          f"F1={best_f1['f1']:.4f}")

    # ── Find optimal for different cost ratios ─────────────────────────────
    print("\n[3/4] 计算不同 cost ratio 下的最优 threshold...")
    cost_ratios = [0.2, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    optimal = find_optimal_thresholds(scan_results, cost_ratios)

    print(f"\n{'Cost Ratio':<14} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Pos Rate':<12}")
    print("-" * 74)
    for r in cost_ratios:
        o = optimal[str(r)]
        print(f"r={r:<12} {o['optimal_threshold']:<12.4f} {o['precision']:<12.4f} "
              f"{o['recall']:<12.4f} {o['f1']:<12.4f} {o['positive_rate']:<12.4f}")

    # ── Generate outputs ────────────────────────────────────────────────────
    print("\n[4/4] 生成图表和报告...")

    # Main figure
    curve_path = os.path.join(OUTPUT_DIR, "cost_threshold_curve.png")
    plot_cost_threshold_curve(scan_results, optimal, cost_ratios, curve_path)

    # Tradeoff chart
    tradeoff_path = os.path.join(OUTPUT_DIR, "precision_recall_tradeoff.png")
    plot_cost_precision_recall_3d(optimal, cost_ratios, tradeoff_path)

    # Report JSON
    production_row = min(scan_results, key=lambda x: abs(x["threshold"] - F1_THRESHOLD))
    report = {
        "method": "cost-based threshold optimization",
        "cost_ratio_definition": "r = cost_FP / cost_FN. r<1: FN is worse. r=1: equal. r>1: FP is worse.",
        "production_threshold": {
            "threshold": F1_THRESHOLD,
            "note": "Current production threshold (F1-optimal at training time, 2026-03). On current test set, true F1-optimal is ~0.28.",
            "precision": production_row["precision"],
            "recall": production_row["recall"],
            "f1": production_row["f1"],
            "positive_rate": production_row["positive_rate"],
            "fp": production_row["fp"],
            "fn": production_row["fn"],
        },
        "true_f1_optimal": {
            "threshold": best_f1["threshold"],
            "precision": best_f1["precision"],
            "recall": best_f1["recall"],
            "f1": best_f1["f1"],
        },
        "cost_based_optimal": optimal,
        "key_insight": {
            "fp_vs_fn_cost": (
                f"At r=0.5 (FN twice as costly as FP): threshold={optimal['0.5']['optimal_threshold']:.2f}, "
                f"recall={optimal['0.5']['recall']:.4f}. "
                f"At r=2.0 (FP twice as costly as FN): threshold={optimal['2.0']['optimal_threshold']:.2f}, "
                f"precision={optimal['2.0']['precision']:.4f}. "
                f"The F1-optimal threshold={F1_THRESHOLD} sits in the middle but should NOT be used "
                "without first asking the business: 'which error is more costly?'"
            ),
            "range": (
                f"Optimal threshold ranges from {optimal[str(min(cost_ratios))]['optimal_threshold']:.2f} "
                f"(r={min(cost_ratios)}) to {optimal[str(max(cost_ratios))]['optimal_threshold']:.2f} "
                f"(r={max(cost_ratios)}). "
                f"Default F1 threshold={F1_THRESHOLD} implicitly assumes r=1 (equal cost)."
            ),
        },
        "data": {
            "n_test_samples": int(len(y_true)),
            "positive_rate_actual": float(y_true.mean()),
            "calibrated_prob_mean": float(probas.mean()),
        },
    }
    report_path = os.path.join(OUTPUT_DIR, "threshold_sensitivity.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] threshold_sensitivity.json saved → {report_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("结果摘要")
    print("=" * 70)
    print(f"\n  业务场景                             最优 threshold   P       R")
    print(f"  {'─' * 60}")
    print(f"  '宁可多申，不错过机会' (r=0.2)        {optimal['0.2']['optimal_threshold']:.2f}            "
          f"{optimal['0.2']['precision']:.3f}   {optimal['0.2']['recall']:.3f}")
    print(f"  '漏掉机会代价是推荐错的2倍' (r=0.5)   {optimal['0.5']['optimal_threshold']:.2f}            "
          f"{optimal['0.5']['precision']:.3f}   {optimal['0.5']['recall']:.3f}")
    print(f"  生产阈值 (r≈F1, t=0.24)            {F1_THRESHOLD}            "
          f"{production_row['precision']:.3f}   {production_row['recall']:.3f}")
    print(f"  '推荐错代价是漏掉的2倍' (r=2.0)      {optimal['2.0']['optimal_threshold']:.2f}            "
          f"{optimal['2.0']['precision']:.3f}   {optimal['2.0']['recall']:.3f}")
    print(f"  '宁可少推，不可推错' (r=5.0)          {optimal['5.0']['optimal_threshold']:.2f}            "
          f"{optimal['5.0']['precision']:.3f}   {optimal['5.0']['recall']:.3f}")

    print(f"\n  关键结论:")
    print(f"  1. 当前 production threshold=0.24, 真正F1最优≈{best_f1['threshold']:.2f}")
    print(f"  2. F1 隐含假设 FP 和 FN 代价相等 — 实际业务通常不相等")
    print(f"  3. 如果业务方认为'漏掉机会更严重'，threshold 应降到 "
          f"{optimal['0.5']['optimal_threshold']:.2f}")
    print(f"  4. 如果业务方认为'推荐错更严重'，threshold 应升到 "
          f"{optimal['2.0']['optimal_threshold']:.2f}")
    print(f"  5. Threshold 不是一个 ML 调参问题——是一个业务偏好问题")
    print(f"\n产物目录: {OUTPUT_DIR}")
    print("  - cost_threshold_curve.png       (放 portfolio)")
    print("  - precision_recall_tradeoff.png  (PR 空间中的最优阈值)")
    print("  - threshold_sensitivity.json     (完整数据)")
    print("=" * 70)


if __name__ == "__main__":
    main()
