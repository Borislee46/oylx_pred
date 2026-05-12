"""Generate school-major admission correlation matrix from cases.feather.

Computes phi coefficients between every pair of (university, major) programs
based on historical admission patterns. Used by the Monte Carlo simulation
in school_combination_optimizer_algorithm/monte_carlo.py to model correlated
admission outcomes via Cholesky decomposition.

Usage:
    python scripts/generate_correlation_matrix.py
    python scripts/generate_correlation_matrix.py --min-samples 5 --output cache/correlation_matrix.feather
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "src" / "machine_learning_models" / "data" / "cases.feather"
DEFAULT_OUTPUT = ROOT / "cache" / "correlation_matrix.feather"


def build_student_admission_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot cases into a (student × program) binary admission matrix.

    A "student" is identified by (background_university, background_major_original, gpa)
    since the same student may apply to multiple target programs.
    """
    df = df.dropna(subset=["target_university", "target_major", "admitted"])
    df["program"] = df["target_university"] + " - " + df["target_major"]
    df["student_key"] = (
        df["background_university"].fillna("?")
        + " | "
        + df["background_major_original"].fillna("?")
        + " | "
        + df["faculty"].fillna("?")
        + " | "
        + df["gpa"].round(1).astype(str)
        + " | "
        + "R" + df["research_count"].fillna(-1).astype(int).astype(str)
        + "P" + df["paper_count"].fillna(-1).astype(int).astype(str)
        + "I" + df["internship_count"].fillna(-1).astype(int).astype(str)
        + "A" + df["award_count"].fillna(-1).astype(int).astype(str)
        + " | "
        + "IELTS" + df["ielts"].round(1).fillna(-1).astype(str)
        + "TOEFL" + df["toefl"].round(0).fillna(-1).astype(str)
    )

    pivot = df.pivot_table(
        index="student_key",
        columns="program",
        values="admitted",
        aggfunc="max",
        fill_value=0,
    )
    return pivot.astype(np.int8)


def phi_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two binary vectors = phi coefficient."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a_std = a.std()
    b_std = b.std()
    if a_std == 0 or b_std == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_correlation_matrix(
    pivot: pd.DataFrame,
    min_samples: int = 5,
) -> pd.DataFrame:
    programs = list(pivot.columns)
    n = len(programs)

    # count students per program
    counts = pivot.sum(axis=0)

    # keep programs with enough students
    valid_mask = counts >= min_samples
    valid_programs = [p for p, ok in zip(programs, valid_mask) if ok]

    if len(valid_programs) < 2:
        print(f"ERROR: only {len(valid_programs)} programs with >= {min_samples} samples", file=sys.stderr)
        raise SystemExit(1)

    print(f"Programs with >= {min_samples} samples: {len(valid_programs)} / {n}")
    print(f"Computing {len(valid_programs) * (len(valid_programs) - 1) // 2} pairwise phi coefficients ...")

    pivot_valid = pivot[valid_programs]
    arr = pivot_valid.values.astype(np.float64)

    # compute phi for all pairs using numpy vectorized corrcoef on subsets
    # For large N, compute in batches
    m = len(valid_programs)
    corr = np.eye(m, dtype=np.float32)

    batch_size = 100
    for i in range(0, m, batch_size):
        end_i = min(i + batch_size, m)
        for j in range(i, m, batch_size):
            end_j = min(j + batch_size, m)
            chunk = np.corrcoef(arr[:, i:end_i].T, arr[:, j:end_j].T)
            ci = chunk[: end_i - i, :]  # everything for rows i..end_i
            # rows belong to block i, cols belong to block j
            sub = ci[:, end_i - i : end_i - i + end_j - j]
            corr[i:end_i, j:end_j] = sub
            if i != j:
                corr[j:end_j, i:end_i] = sub.T

        pct = min(100, (end_i * 100) // m)
        print(f"  {pct}% ...")

    corr = np.nan_to_num(corr, nan=0.0)
    return pd.DataFrame(corr, index=valid_programs, columns=valid_programs)


def main():
    parser = argparse.ArgumentParser(description="Generate school-major correlation matrix")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to cases.feather")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output path")
    parser.add_argument("--min-samples", type=int, default=5, help="Min students per program")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"ERROR: {cases_path} not found", file=sys.stderr)
        raise SystemExit(1)

    print(f"Loading {cases_path} ...")
    df = pd.read_feather(cases_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    pivot = build_student_admission_matrix(df)
    print(f"  Student-program matrix: {pivot.shape[0]} students × {pivot.shape[1]} programs")

    corr_matrix = compute_correlation_matrix(pivot, min_samples=args.min_samples)
    print(f"  Final matrix: {corr_matrix.shape[0]} × {corr_matrix.shape[1]}")

    # Also compute pair_weight_matrix: joint student count for each pair
    print("Computing pair weight matrix (joint sample counts) ...")
    valid_programs = list(corr_matrix.columns)
    pivot_valid = pivot[valid_programs]
    present = pivot_valid.values.astype(np.int8)
    # n_ij = number of students who have data for BOTH program i AND program j
    pair_counts = present.T @ present  # (m × m) matrix
    pair_weight = pd.DataFrame(
        pair_counts.astype(np.int32),
        index=valid_programs,
        columns=valid_programs,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.reset_index().to_feather(output_path)
    print(f"Saved correlation matrix to {output_path}")

    weight_path = output_path.parent / "pair_weight_matrix.feather"
    pair_weight.reset_index().to_feather(weight_path)
    print(f"Saved pair weight matrix to {weight_path}")

    # Quick stats
    vals = corr_matrix.values
    upper = vals[np.triu_indices_from(vals, k=1)]
    w_upper = pair_weight.values[np.triu_indices_from(pair_weight.values, k=1)]
    print(f"\nCorrelation stats (upper triangle, n={len(upper):,}):")
    print(f"  mean:   {upper.mean():.4f}")
    print(f"  std:    {upper.std():.4f}")
    print(f"  min:    {upper.min():.4f}")
    print(f"  p50:    {np.median(upper):.4f}")
    print(f"  p95:    {np.percentile(upper, 95):.4f}")
    print(f"  max:    {upper.max():.4f}")
    print(f"\nPair weight stats (joint students per pair):")
    print(f"  mean:   {w_upper.mean():.1f}")
    print(f"  p50:    {np.median(w_upper):.1f}")
    print(f"  min:    {w_upper.min()}")
    print(f"  max:    {w_upper.max()}")


if __name__ == "__main__":
    main()
