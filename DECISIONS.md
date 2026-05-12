# DECISIONS.md

Design decision log for the Signals admission prediction system. Each entry documents the context, alternatives considered, the choice made, and observed consequences. This file exists because a system's intelligence is in its tradeoffs, not its code.

---

## DEC-001: Sigmoid (Platt scaling) over Isotonic calibration

**Date**: 2025-Q2  
**Context**: XGBoost `predict_proba` outputs are not true probabilities — boosting fits residuals without calibration constraints. Admission prediction requires interpretable probabilities ("you have a 68% chance").

**Alternatives considered**:
- **Isotonic regression**: Non-parametric, more flexible. Produces better Brier on calibration set.
- **No calibration**: Accept raw XGBoost probabilities as-is.

**Decision**: Sigmoid (Platt scaling, via `CalibratedClassifierCV(method='sigmoid')`).

**Why**:
- Isotonic overfits on small calibration sets (our ~10k calib set is borderline). The reliability diagram showed non-monotonic steps in low-probability bins (0-20% range), which is business-unacceptable — "probability goes down when score goes up" destroys trust.
- Sigmoid has only 2 parameters (a, b), forcing a smooth monotonic curve. Sacrifices ~0.003 Brier for guaranteed monotonicity and smoothness.
- Confirmed via V1 calibration analysis: calibration error component dropped 95% (0.020 → 0.001), ECE dropped 78% (0.118 → 0.026).

**Consequence**: Calibrated probabilities compressed from [0.0001, 0.9991] to [0.1304, 0.7253] — no ≈0% or ≈100% predictions, consistent with "nothing is certain in education."

---

## DEC-002: Quadratic GPA penalty over linear or threshold-based

**Date**: 2025-Q2  
**Context**: XGBoost predictions are reliable near the data mean but unconstrained for edge cases. A GPA of 2.1 applying to Tier-1 schools needs a domain-informed penalty.

**Alternatives considered**:
- **Linear penalty**: Fixed % per z-score unit below mean.
- **Hard threshold**: "GPA < 2.5 → ×0.5".
- **No penalty**: Trust the model's extrapolation.

**Decision**: Quadratic penalty: `min(0.8, 0.15 × z²)`, where `z = (mean - gpa) / std`, applied only when `gpa < mean`.

**Why**:
- Linear penalizes 2.5→2.7 and 3.5→3.7 identically. In reality, the marginal impact of +0.1 GPA differs enormously across the distribution — 2.5→2.6 matters far more than 3.5→3.6.
- Quadratic makes "near mean" penalty nearly invisible (z² ≈ 0 for z < 1), while "far from mean" penalty accelerates rapidly (z=2 → 0.15 × 4 = 0.60 penalty). This mirrors how admissions committees actually think: a slightly-below-average GPA doesn't hurt much; a severely-below-average GPA hurts a lot.
- Z-score based (not absolute threshold) auto-adapts to different application cohorts. A 3.0 might be "low" in a 3.5-mean pool but "fine" in a 3.0-mean pool.

**Consequence**: 46.8% of test cases receive a GPA penalty, but mean |Δprob| is only 0.006 — the quadratic shape correctly concentrates impact on the tail.

---

## DEC-003: Tiered (step-function) language penalty over continuous

**Date**: 2025-Q2  
**Context**: Language scores (IELTS/TOEFL) have real-world "hard thresholds" — universities publish minimum requirements.

**Alternatives considered**:
- **Continuous function** (like GPA's quadratic).
- **Hard cutoff**: "IELTS < 6.0 → reject".

**Decision**: Tiered step-function penalty with 5 levels, triggered when `score < pass_line` (mean - 0.5σ).

**Why**:
- Unlike GPA, language proficiency has actual gatekeeping thresholds. IELTS 5.9 and 6.0 are numerically close but semantically worlds apart — one meets minimum, the other doesn't.
- A continuous function would smooth over this cliff, misrepresenting the real admission process.
- `pass_line = mean - 0.5σ` is a statistical proxy for "competitive threshold" — above this, most admitted students cluster; below, admissions drop sharply.

**Consequence**: 35.2% of test cases affected. The tiered structure produces a mean |Δprob| of 0.031 — significantly larger per-case impact than GPA penalty, reflecting the "cliff" effect of language thresholds.

---

## DEC-004: Cross-major penalty ×0.5, cross-faculty penalty ×0.3

**Date**: 2025-Q2  
**Context**: The model predicts P(admit | target_university, target_major). It doesn't know that "same university, different major" can mean completely different admission difficulty. Even worse, "different faculty" (Science → Law) is a near-impossible pivot.

**Alternatives considered**:
- **Data-driven only**: Let similarity scores and case data determine everything.
- **Same penalty for both**: One multiplier for any "cross-domain" application.

**Decision**: Two-tier domain penalty — cross-major ×0.5, cross-faculty ×0.3.

**Why**:
- Cross-major (same knowledge system, different branch): penalty is moderate. A CS major applying to Data Science is a lateral move.
- Cross-faculty (entirely different knowledge system): penalty is severe. A Chemistry major applying to Law is essentially starting over — the applicant lacks prerequisites, the admitting department doubts commitment, and historical data (<1% of cases) is too sparse for the model to learn this.
- Domain knowledge > data when data is sparse. This is a first-principles design, not a data-driven one.

**Consequence**: Faculty penalty affects 19.5% of cases with mean |Δprob| = 0.035. The language penalty has the largest single-layer probability impact (|Δprob| = 0.048), while the cross-major penalty causes the largest ranking disruption (τ = 0.0185). Together, these layers form an integrated system — V2 ablation confirmed that removing any single layer causes near-complete ranking reordering due to the decay cascade.

---

## DEC-005: `scale_pos_weight` over SMOTE for class imbalance

**Date**: 2025-Q1  
**Context**: Admission data is imbalanced — ~34% positive rate, roughly 1:2 ratio. Models trained on imbalanced data tend to predict the majority class.

**Alternatives considered**:
- **SMOTE / ADASYN**: Synthetic minority oversampling.
- **Random undersampling**: Discard majority samples.
- **Class weights**: Inverse-frequency weighting.

**Decision**: `scale_pos_weight = n_negative / n_positive` (≈1.96), passed directly to XGBoost.

**Why**:
- SMOTE generates synthetic samples via interpolation in feature space. With high-cardinality categorical features (1051 majors, 1322 universities), SMOTE produces meaningless synthetic cases — an interpolated point between "香港大学+计算机" and "香港中文大学+历史" has no real-world interpretation.
- Undersampling discards information — unacceptable when data is already scarce (12k effective samples).
- `scale_pos_weight` mathematically adjusts gradient contributions without fabricating or discarding data. It's also compatible with monotonic constraints (SMOTE can break monotonicity in synthetic samples).

**Consequence**: Training stable, no synthetic data artifacts. The model is conservative on minority class predictions, which is appropriate for this domain.

---

## DEC-006: Monotonic constraints on GPA, language, and experience counts

**Date**: 2025-Q1  
**Context**: In admission prediction, certain features have an indisputable directional relationship with the outcome: higher GPA → higher probability; more experience → higher probability.

**Alternatives considered**:
- **No constraints**: Let the model learn relationships purely from data.
- **Post-hoc correction**: Fix violations after training.

**Decision**: XGBoost `monotone_constraints = (1, 1, 1, 1, 1, 1)` for all continuous features.

**Why**:
- Business trust: If a model tells a user "your GPA increased by 0.2 but your predicted probability went down," the user (and advisor) immediately lose trust. The system dies on a single bad case.
- Extrapolation safety: Training data lacks extreme values (very high/low GPA). Without constraints, the model could learn spurious reversal patterns in data-sparse regions. Monotonicity guarantees reasonable behavior even for OOD inputs.
- Regularization effect: Reduces the hypothesis space, preventing the model from fitting noise that happens to correlate negatively with outcome.

**Consequence**: The model never produces a case where "more experience → lower probability." This was confirmed implicitly — no advisor has ever reported a monotonicity violation.

---

## DEC-007: `DECAY_FACTOR = 0.85` for penalty arbitration

**Date**: 2025-Q3  
**Context**: When multiple penalty layers fire simultaneously, direct multiplication (`prob × 0.85 × 0.7 × 0.3 = prob × 0.1785`) is catastrophically aggressive. An arbitration mechanism is needed.

**Alternatives considered**:
- **Decay = 0.5**: Fast decay, the 2nd penalty barely matters.
- **Decay = 0.9**: Slow decay, near-full stacking.
- **Cap total only**: Just enforce the 70% cap, let penalties stack linearly.

**Decision**: `PENALTY_DECAY_FACTOR = 0.85`, applied progressively to penalties sorted by severity. Total penalty capped at 70%.

**Why**:
- 0.5: 2nd penalty retains only half its weight — makes multi-penalty cases indistinguishable from single-penalty cases. Defeats the purpose of having multiple layers.
- 0.9: 3 penalties ≈ 0.9 + 0.81 + 0.73 = 2.44 → almost always hits the 70% cap immediately. The cap becomes the only active mechanism, making the decay meaningless.
- 0.85: 2 penalties ≈ 0.85 + 0.72 = 1.57 effective ratio; 3 penalties ≈ 0.85 + 0.72 + 0.61 = 2.18. The decay is noticeable but not crippling — the 1st penalty is always felt at full strength, the 3rd at ~60%.
- Decay is applied per-penalty, not to the total — the primary problem (e.g., low GPA) should express fully; secondary problems (e.g., cross-major) should add diminishing marginal impact.

**Consequence**: V2 ablation revealed this as a defining design feature — the decay cascade means removing one layer amplifies all remaining layers. The system behaves as an integrated whole, not 5 independent if-else blocks.

---

## DEC-008: Threshold = 0.24 (not 0.5)

**Date**: 2025-Q1  
**Context**: Default binary classification threshold is 0.5. In imbalanced settings (34% positive rate), this produces near-zero positive predictions.

**Alternatives considered**:
- **0.5 (default)**: Standard binary threshold.
- **Youden's J statistic**: Maximize sensitivity + specificity.
- **F1-optimal**: Threshold that maximizes F1 on the validation set.

**Decision**: `DEFAULT_PREDICTION_THRESHOLD = 0.24`, originally F1-optimal at training time (2026-03). On the current test set, the true F1-optimal is ~0.28 — the 0.04 shift reflects minor calibration drift between train-time and eval-time data splits.

**Why**:
- 0.5 in a 34%-positive setting means the model predicts "admit" for almost no one — useless for a recommendation system.
- 0.24 favors recall (finding opportunities) over precision (avoiding false recommendations). This is a product decision — the system is designed to surface options for advisor review, not to auto-reject.
- The F1-optimal threshold at training time may differ from the current test set — this is normal and expected. V3 cost-based analysis shows the threshold should be re-evaluated periodically based on current business priorities.

**Consequence**: 54.9% of predictions are "positive" at this threshold (vs 33.7% actual rate). This was initially misdiagnosed as "model overestimates by 21%." V1 calibration analysis proved otherwise — mean calibrated probability (0.3429) nearly equals actual rate (0.3374). The "gap" is a threshold effect, not a calibration failure.

**V3 analysis added**: The optimal threshold varies from 0.16 to 0.71 depending on the FP/FN cost ratio. The production threshold 0.24 implicitly favors recall — this should be explicitly confirmed with business stakeholders, not assumed from F1.

---

## DEC-009: log1p transform + 99th percentile capping for count features

**Date**: 2025-Q1  
**Context**: Experience counts (research, awards, internships, papers) follow a right-skewed distribution — most students have 0-2, outliers have 5-10+.

**Alternatives considered**:
- **Raw counts**: No transformation.
- **Standardization (z-score)**: Center and scale.
- **log1p + 95th percentile cap**.

**Decision**: `log1p` transform + 99th percentile capping.

**Why**:
- Raw counts: XGBoost's quantile-based splits handle skew better than linear models, but extreme outliers (e.g., "10 internships" — possibly a data entry error) still pull splits.
- Standardization: Loses interpretability; doesn't address the skew shape.
- log1p (not `log(x)`): `log(0) = -inf`, and many students genuinely have 0 experience counts. `log1p(0) = 0` — correct baseline.
- 99th (not 95th) percentile cap: 95th is too aggressive, capping ~5% of legitimate multi-experience students. 99th removes the top 1% (likely data errors or extreme outliers) while preserving genuine high-experience signals.

**Consequence**: Feature distributions stabilized. The model no longer over-weights a single "10 internships" outlier.

---

## DEC-010: Median imputation over mean or zero-fill

**Date**: 2025-Q1  
**Context**: Missing values in numeric features (GPA, language scores) need handling. The choice of imputation value directly affects the model's baseline assumption about unknown students.

**Alternatives considered**:
- **Zero-fill**: Missing → 0.
- **Mean imputation**: Missing → average.
- **Drop rows**: Remove incomplete cases.

**Decision**: Median imputation for numeric features.

**Why**:
- Zero-fill: For GPA, 0 is not a "neutral" value — it's an extreme negative signal. A student with missing GPA gets treated identically to a student who failed every course. This introduces severe negative bias.
- Mean imputation: For right-skewed distributions (experience counts), the mean is pulled up by outliers. A missing internship count gets imputed above what most students actually have — optimistic and wrong.
- Median: Robust to skew. Imputes the "typical" value — a conservative, defensible baseline for unknown information.
- Dropping rows wastes scarce labeled data (12k samples).

**Consequence**: Conservative baseline for unknown students. The model treats "missing data" as "typical student," which is the least-wrong default when you can't ask the student to clarify.

---

## DEC-011: Four-tier JSON repair for LLM agent output

**Date**: 2025-Q3  
**Context**: DeepSeek API occasionally returns malformed JSON — trailing commas, missing quotes, extra text outside JSON blocks. ~1.2% of requests fail direct `json.loads()`.

**Alternatives considered**:
- **Retry on failure**: Re-request from the API.
- **Single repair method**: Just call an API-based repair.
- **Give up and return error**.

**Decision**: Four-tier cascade: direct parse → lightweight regex (<1ms) → `json_repair` library (<5ms) → API repair fallback (~9s).

**Why**:
- Retry wastes API budget on transient syntax issues that are trivial to fix locally.
- API-only repair adds ~9s latency to 1.2% of requests — unacceptable for UX.
- The 4-tier design catches 98%+ of failures without an API call. Only truly broken JSON (unrecoverable structure) reaches tier 4.
- Lightweight regex specifically targets DeepSeek's known failure modes (trailing commas in arrays, unquoted keys) — this is a data-informed, not generic, fix.

**Consequence**: <0.02% of requests actually hit the API repair tier. Agent reliability is 99.8%+ without sacrificing response time.

---

## DEC-012: `flooring = 0.005`, not 0

**Date**: 2025-Q3  
**Context**: After penalty arbitration, probabilities can approach but should never equal 0.

**Decision**: `ARBITRATION_MIN_PROBABILITY = 0.005` (0.5%).

**Why**:
- 0 means "absolutely impossible." In education admissions, there is almost no truly impossible case — special talent admissions, policy changes, quota adjustments can all override normal expectations.
- 0.005 = "extraordinarily unlikely but not impossible." This preserves the possibility while being honest about the improbability.
- UX: a "0%" on screen feels like the system is passing judgment ("you have no chance"). "0.5%" feels like data ("based on historical patterns, this is extremely rare"). The difference is emotional, not mathematical.

**Consequence**: No user ever sees a 0% prediction. The floor value is small enough to not mislead but non-zero enough to not feel dismissive.

---

## DEC-013: TOEFL/IELTS merged into single `language_score`

**Date**: 2025-Q1  
**Context**: Students submit either TOEFL or IELTS, rarely both. Two separate features means most rows have one missing value.

**Alternatives considered**:
- **Two features with NaN handling**: `toefl_score`, `ielts_score`.
- **Pick one, drop the other**: Only support TOEFL or IELTS.

**Decision**: Normalize both to [0,1] (TOEFL/120, IELTS/9), merge via `max(toefl_norm, ielts_norm)`.

**Why**:
- Two features: Sparse — each student only has one. The model learns from whichever is non-null, but the missing pattern itself becomes a feature (IELTS-takers vs TOEFL-takers), which is spurious.
- Pick one: Unacceptable — students take different tests based on region and school requirements.
- Merged: Both tests measure the same underlying construct (English proficiency). Normalization to percentage-of-maximum makes them directly comparable. `max()` handles the rare case where a student has both scores.

**Consequence**: `language_score` ranges [0, 1] with interpretive clarity — 0.83 = "83% of max score." V4 bootstrap showed this feature has the highest rank variance (std=0.63) among all features, suggesting the merge may lose some signal. This is a candidate for future improvement (potentially: keep both and use a more sophisticated fusion).

---

## DEC-014: Streamlit over FastAPI + React

**Date**: 2024 (project inception)  
**Context**: Need to build an internal tool for ~730 advisors. ML pipeline is the core value; frontend is secondary.

**Alternatives considered**:
- **FastAPI + React**: Full-stack, separate backend/frontend.
- **Jupyter notebooks**: Serve predictions ad-hoc.
- **Gradio**: ML-focused UI framework.

**Decision**: Streamlit (single-process, Python-only).

**Why**:
- Development speed: One developer, full-stack. Streamlit eliminates the frontend build step entirely.
- Target users: Internal advisors at XDF. No need for the polish or scalability of a public-facing React app.
- Python-native: All ML code (XGBoost, calibration, TF-IDF) is in Python. No serialization/deserialization layer needed between model and UI.
- Tradeoff accepted: Single-process = no concurrency. If DAU grows from 10 to 1000, Streamlit becomes the bottleneck. The architecture allows replacing the Streamlit frontend with FastAPI without touching the ML pipeline.

**Consequence**: System deployed and stable for 1+ year. The `json_api.py` module already exists as a proto-API layer, making future FastAPI migration a thin wrapper task rather than a rewrite.

---

*Last updated: 2026-05-09. This file grows as design decisions accumulate. Each entry should be short enough to read in 30 seconds, detailed enough to defend in an interview.*
