# Model Card: Signals Admission Prediction System

> Format follows [Google Model Cards](https://modelcards.withgoogle.com/about). This document is a statement of honesty — what the model can do, what it cannot do, and what we don't yet know.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model type** | XGBoost classifier (gradient boosted trees) |
| **Calibration** | Sigmoid (Platt scaling) via `CalibratedClassifierCV` |
| **Post-processing** | 5-layer probability adjustment pipeline (GPA, language, cross-major, cross-faculty, professional degree) |
| **Version** | `xgboost_20260316_092608` |
| **Training date** | 2026-03-16 |
| **Framework** | XGBoost 3.0.2, scikit-learn 1.8.0 |
| **Input features** | 10 features: 4 categorical (target/background university/major), 6 numeric (GPA, language_score, 4 experience counts) |
| **Output** | Calibrated probability of admission: P(admit \| university, major, student profile) ∈ [0.005, ~0.73] |
| **Prediction threshold** | 0.24 (F1-optimal; not business-optimal — see V3 analysis) |
| **Maintainer** | lijiapeng8@xdf.cn |

### Model Architecture

```
Raw features → Feature Engineering → XGBoost (375 trees, depth=10) → Sigmoid Calibration
    → Post-processing Pipeline:
        Layer 1: GPA quadratic penalty
        Layer 2: Language tiered penalty
        Layer 3: Cross-major penalty (×0.5)
        Layer 4: Cross-faculty penalty (×0.3)
        Layer 5: Professional degree penalty (×0.7)
        Arbiter: Decay-weighted stacking, capped at 70% total penalty
    → Final calibrated probability
```

---

## Intended Use

- **Primary**: Assist study-abroad advisors at XDF in making school/major recommendations by providing data-driven probability estimates.
- **User**: Trained advisors who interpret probabilities in context, not end students.
- **Decision type**: Decision support, NOT automated decision-making. Advisors are expected to override predictions based on their professional judgment.
- **Geography**: Hong Kong, Singapore, Macau, Malaysia master's programs.
- **Coverage**: 23 universities, 1,051 majors.

### The system IS designed to:
- Surface viable school/major combinations ranked by predicted admission probability
- Quantify how much each factor (GPA, language, background match) affects the prediction
- Provide a trace showing every adjustment step for transparency

### The system is NOT designed to:
- Make final admission decisions (that's the university's job)
- Replace advisor judgment
- Be shown directly to prospective students without advisor interpretation
- Predict outcomes for undergraduate, PhD, or non-Asian programs

---

## Out-of-Scope Use

**Do NOT use this model for:**

1. **Student-facing predictions without advisor mediation**: A "12%" probability shown to a student without context causes demoralization. The system includes tier categorization (reach/match/safety) specifically to prevent this.

2. **Automated rejection**: The model floor is 0.005, not 0. Using predictions to auto-filter applicants violates the system's design principle — "no case is absolutely impossible."

3. **Non-Asian institutions**: Training data covers HK/SG/MO/MY programs only. Extrapolation to UK/US/AU programs will produce unreliable predictions.

4. **Programs outside the 23-school coverage**: New schools not in `school_base.feather` have no valid combinations and cannot be predicted.

5. **PhD, undergraduate, or non-master's programs**: The training data is exclusively master's applicants.

6. **High-stakes standalone decisions**: The model's Brier Score is 0.186. A 15% vs 25% predicted probability should not be the sole basis for advising a student to abandon a target school.

---

## Training Data

| Aspect | Details |
|--------|---------|
| **Source** | XDF internal application records |
| **Sample size** | 61,716 cases (49,372 train, 12,344 test) |
| **Time range** | Not timestamped (see Limitations) |
| **Positive rate** | 33.7% (admitted / total applications) |
| **Geographic coverage** | Applicants to HK, SG, MO, MY master's programs |
| **Feature completeness** | Text experience fields (research_detail, etc.) missing for ~19% of cases (weighted down to 0.85) |

### Known Data Biases

1. **Selection bias (selective labels)**: We only observe admission outcomes for cases where the student actually applied. Students who were discouraged from applying (or self-censored) have no labels. The training distribution of "background → target" is shaped by advisor behavior, not by the true population of possible applicants.

2. **Self-fulfilling prophecy risk**: If the system predicts low probability → advisors stop recommending → students don't apply → the model never observes counterfactual outcomes. This feedback loop is currently unmonitored.

3. **Recency**: The latest 10,000 cases receive a 1.1× sample weight to partially compensate for potential concept drift. This is a heuristic, not a rigorous drift correction.

4. **University representation**: Tier-1 universities (HKU, NUS, etc.) are overrepresented in the data because they receive more applications. Tier-3/4 schools have sparser data, making predictions for them less reliable.

5. **Major aggregation**: 1,051 unique major names are reduced to 94 background_major categories and matched through a similarity system. Rare majors with <5 cases effectively have no data-driven prediction — the system falls back to similarity-based estimates.

---

## Evaluation

### Core Metrics (on 12,344-sample held-out test set)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ROC-AUC | 0.750 | Discriminatory power moderate; limited by inherent uncertainty in admissions |
| Average Precision | 0.612 | Better than AUC for imbalanced setting; indicates reasonable ranking |
| Brier Score | 0.186 | Mean squared error of probability predictions |
| Log Loss | 0.551 | Penalizes confident-wrong predictions |
| **ECE** (post-calibration) | **0.026** | Expected calibration error — excellent calibration |
| Calibration reliability | 0.001 | Nearly perfect (0 = perfect) |
| Refinement (resolution) | 0.039 | Moderate discrimination across risk groups |

### Calibration Quality (V1 analysis)

- **Raw XGBoost**: Mean predicted = 39.5%, actual = 33.7%, bias = +5.8pp, ECE = 0.118
- **After sigmoid calibration**: Mean predicted = 34.3%, actual = 33.7%, bias = +0.6pp, ECE = 0.026
- Calibration error reduced by **95%** (0.020 → 0.001)
- Calibrated probability range: [0.13, 0.73] — no extreme predictions

### Ablation Results (V2 analysis)

Removing all 5 post-processing layers drops Kendall's τ to near zero (essentially no correlation with the full-pipeline ranking). The post-processing pipeline is not a minor adjustment — without it, the recommendation ranking is completely different from what advisors see. This means the 5-layer pipeline effectively IS the ranking mechanism; XGBoost provides the raw signal that the pipeline reshapes.

| Layer removed | Kendall τ vs baseline | Mean \|Δprob\| | Affected |
|--------------|----------------------|----------------|----------|
| Language | 0.023 | 0.048 | 35.2% |
| Cross-faculty | 0.023 | 0.035 | 19.5% |
| GPA | 0.019 | 0.010 | 46.8% |
| Cross-major | 0.019 | 0.004 | 35.9% |
| Professional | 0.276 | <0.001 | 0.2% |

### Feature Stability (V4 bootstrap, 1000 rounds)

Top-3 features (target_major, background_university, background_major) have perfectly locked ranks (CI width = 0). Paper_count and internship_count are statistically indistinguishable (98% CI overlap) — only one is needed for feature selection.

---

## Ethical Considerations

### Selective Labels & Feedback Loops

This is the most serious unresolved ethical issue with this system.

**The problem**: The model trains on (student, university, major) → admitted outcomes. But we only observe outcomes for applications that were actually submitted. If the system (or advisors influenced by it) systematically discourages certain combinations, those combinations never generate training data, and the model's confidence in "this is a bad match" grows without ever being tested.

**Current mitigation**: None automated. Advisors provide ad-hoc feedback, but this is not systematic.

**Planned**: An exploration arm — 5% of recommendations deliberately include "model-skeptical" combinations to collect counterfactual data. This requires ethical sign-off (recommending combinations the model believes are unlikely to succeed).

### Fairness Across Student Backgrounds

- The model uses `background_university` as a feature. This encodes China's university hierarchy (C9 > 985 > 211 > 双非). Students from lower-tier universities receive systematically lower predictions — this may reflect real admission patterns but also risks perpetuating them.
- GPA is standardized within the training distribution, but grading standards vary across universities. A 3.5 from a grade-deflated institution may represent stronger performance than a 3.8 from a grade-inflated one.

### Transparency

Every prediction includes an adjustment trace showing exactly which penalties were applied and by how much. This is deliberately designed for contestability — an advisor can see that a 20% probability resulted from "GPA penalty -8%, cross-major penalty -35%" and challenge the cross-major penalty if they have additional context.

---

## Known Limitations

### 1. No temporal awareness
The training data lacks reliable timestamps. The model cannot distinguish between "admission standards in 2023" and "admission standards in 2025." If HKU suddenly expands enrollment by 200%, the model will continue predicting by old standards until enough new data accumulates (typically 1 application cycle = 6-12 months).

**S1 rolling backtest**: No clear drift detected in current data, but this is expected for a random split — true drift would only appear in future data.

### 2. Cold start for new schools and majors
New programs with 0 historical cases have no data-driven prediction. The system falls back to similarity-based estimates (using E5 embeddings of major descriptions). This is a heuristic — accuracy for cold-start cases is unknown and likely worse.

### 3. Policy and regulation changes are invisible
The model has no awareness of external shocks: visa policy changes, quota adjustments, new bilateral agreements, changes in language requirements. These must be manually monitored and can make predictions systematically off for months.

### 4. OOD behavior is defined by domain rules, not data
For students with extreme profiles (GPA > 3.95, IELTS 9.0, 3+ publications), the model enters OOD territory. The post-processing layers (flooring, capping, asymmetric OOD handling) provide guardrails, but these are domain-rule-based, not data-validated. The OOD detection system (GREEN/YELLOW/RED tiers) correctly identifies 14.3% of test cases as YELLOW (1-2 features OOD), but the RED tier (3+ features simultaneously OOD) has zero cases in the current test set — the random train/test split keeps cases in-distribution. Production traffic may reveal RED-tier cases that the system has never been tested on.

### 5. Accuracy ceiling from inherent uncertainty
Admissions involve factors the model cannot see: statement of purpose quality, recommendation letter strength, interview performance, cohort competition, quota availability. A Brier Score of 0.186 means predictions carry meaningful uncertainty — the model cannot (and should not claim to) predict individual outcomes with high confidence.

### 6. Language score merging loses signal
TOEFL and IELTS are merged into a single `language_score` via max normalization. V4 bootstrap showed this feature has the highest rank variance (std=0.63) across bootstrap rounds, suggesting the merge may discard useful information. This is a candidate for architectural improvement.

---

## Maintenance & Monitoring

| Aspect | Current State | Ideal State |
|--------|--------------|-------------|
| Retrain cadence | Manual, ad-hoc | Scheduled (quarterly or drift-triggered) |
| Drift detection | Not automated | PSI check on feature distributions monthly |
| Calibration monitoring | Not automated | Brier + ECE on recent cohorts after each admission cycle |
| Feedback collection | Ad-hoc advisor reports | Systematic "prediction vs outcome" tracking |
| Model versioning | File timestamps | Explicit version registry with changelog |

---

*This model card reflects the system as of 2026-05-09. It should be updated after each retraining and whenever a limitation is discovered or addressed.*
