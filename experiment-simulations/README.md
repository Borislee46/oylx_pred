# Experiment Simulations

Statistical thinking demonstrations using real project data. Each simulation is a self-contained script that proves a DS concept — not just definitions, but verifiable evidence.

**Why this exists**: Most DS candidates can recite definitions. Few have run simulations on their own data to verify them. This directory is the difference between "I learned this" and "I verified this."

## Simulations

| # | Directory | Concept | Key Result |
|---|-----------|---------|------------|
| S1 | `rolling_backtest/` | Rolling Window Backtest | No drift detected; AUC CV=1.7% across 10 windows |
| S2 | `peeking/` | Peeking Problem | Type I Error: 5.7% → 24.0% → 44.5% (7.8x) |
| S3 | `multiple_testing/` | Multiple Testing | Bonferroni vs BH: both correct, Bonferroni misses more |
| S4 | `power/` | Power Analysis / MDE | MDE=2.4pp at N=12,344; need 71k for +1pp |
| S5 | `stratification/` | Stratified Sampling | Up to 6% variance reduction by school tier |
| S6 | `ood/` | OOD Detection | 14.3% YELLOW cases; random split hides RED tier |

## Quick Start

```bash
# All experiments run from project root
cd oylx_hk_pred_test-master

# S2: Peeking problem (no dependencies)
python experiment-simulations/peeking/peeking_problem_demo.py

# S4: Power analysis (no dependencies)
python experiment-simulations/power/mde_curve.py

# S3: Multiple testing (no dependencies)
python experiment-simulations/multiple_testing/bonferroni_vs_bh.py

# S5: Stratified sampling (no dependencies)
python experiment-simulations/stratification/stratified_vs_random.py

# S6: OOD detection (uses project data)
python experiment-simulations/ood/ood_three_tier.py

# S1: Rolling backtest (uses project data + model)
python experiment-simulations/rolling_backtest/time_window_drift.py
```

## Interview Relevance

| Question | Simulation | Answer |
|----------|-----------|--------|
| "How do you prevent p-hacking?" | S2 Peeking | "I simulated it — daily peeking inflates α from 5% to 25%" |
| "20 metrics, how to control FPs?" | S3 Multiple Testing | "Benjamini-Hochberg balances power vs FDR better than Bonferroni" |
| "What if you can't run an A/B test?" | S4 Power | "I computed MDE for my sample size — can only detect +2.4pp" |
| "Why stratify experiments?" | S5 Stratification | "Stratifying by school tier reduces variance ~6% in my data" |
| "How do you handle outliers?" | S6 OOD | "Three-tier system: GREEN normal, RED rule-only" |
| "When should you retrain?" | S1 Backtest | "Rolling window tracks Brier trend — retrain when trend turns positive" |

## Portfolio Value

These 6 simulations, combined with the `reports/` directory (V1-V4 calibration/ablation/threshold/bootstrap experiments), form a complete "DS research evidence" portfolio. A hiring manager can open any script, run it, and see real statistical thinking applied to real data — not textbook examples.
