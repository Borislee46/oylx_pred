# WritePrint — Algorithm & Research Notes

## Current State (2026-04-28)

**Model:** Ridge regression, 10 features, 9 training samples, LOO-CV MAE 16.3%.

**Dominant signal:** `first_person` r=-0.901. The model is essentially an I/me/my counter with 9 auxiliary features. AI-generated text avoids first-person pronouns; human personal statements use them naturally.

## Known Weaknesses

1. **Over-reliance on first_person.** A student who writes with few I/me/my (common in some cultures or STEM fields) gets a high AI score. An AI text stuffed with "I think" scores 0%.
2. **9 samples is not enough.** Model can only learn ~2 strong linear signals. Cannot capture interaction effects (e.g., long_words + passive + uniform_sentences together = AI).
3. **No interaction features.** `long_word_ratio × passive_ratio`, `first_person × sentence_CV`, etc. are not in the model.
4. **Linear only.** AI detection is inherently non-linear. XGBoost would fit better but needs 30+ samples.
5. **No per-sample uncertainty.** ±18% is a global estimate. BayesianRidge would give per-prediction confidence intervals.

## Optimization Roadmap (when more labeled samples are available)

### Phase 1: Add interaction features (need 15+ samples)
- `long_word × passive` — AI uses both long words AND passive voice together
- `first_person × sentence_CV` — human = varied sentences AND many I/me
- `ttr × avg_word_len` — low diversity + long words = AI pattern
- `comma_density × long_word_ratio` — AI uses complex punctuation with long words

### Phase 2: Switch to non-linear model (need 25+ samples)
- XGBoost with monotonic constraints (already used in main prediction pipeline)
- BayesianRidge for per-prediction uncertainty
- CalibratedClassifierCV(sigmoid) for proper probability calibration

### Phase 3: Ensemble (need 50+ samples)
- Stacking: Ridge + XGBoost + BayesianRidge → meta-learner
- Cross-validation strategy: Leave-One-Out for small N, 5-fold for larger N

## LLM Rewriting Insights

Feature-guided prompting works dramatically better than generic "make it human":
- Give the LLM specific numerical targets (e.g., "add 15+ I/me/my", "average sentence < 15 words")
- Show the feature analysis so the LLM knows exactly what to change
- Model: deepseek-v4-flash (test) / deepseek-v3.2 (prod)

## Labeling Guide

For new samples:
- Filename format: `{description}-{score}%.pdf`
- Score = Turnitin AI detection score (0-100)
- 0-20 = human, 20-40 = mostly human, 40-60 = mixed, 60-80 = AI patterns, 80-100 = strongly AI
- Place PDFs in `data/sample/`
- Model retrains automatically on next `get_model()` call
