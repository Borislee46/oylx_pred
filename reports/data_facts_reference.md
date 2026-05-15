---
name: Data Facts Reference
description: Authoritative data dimensions, sparsity stats, model features — verified 2026-05-14
type: reference
---

## Data Facts (verified 2026-05-14 against cases.feather)

| Fact | Value |
|------|-------|
| Total rows | 61,716 |
| Admit rate | 33.7% |
| Target schools | 23 |
| (School, Major) pairs | 1,210 |
| Unique 4-key combos (bg_uni + bg_major + target_uni + target_major) | 47,835 |
| Median samples per combo | 1 |
| Combos with ≤1 sample | 88.5% |
| Combos with ≤4 samples | 99.3% |
| Max samples in any combo | 30 |

## Model (xgboost_20260316_092608.ubj)

10 features: `target_university, target_major, background_university, background_major, gpa, research_count, paper_count, internship_count, award_count, language_score`

## What-If Simulator

`what_if_simulator.py` `_build_input_data()` correctly passes all 10 features including `paper_count` and `award_count`. 5 hardcoded scenarios per school, uses top 5 majors per target school, aggregates by averaging.

## School Profiles

`school_profiles.py`: MIN_SAMPLES_FOR_PROFILE=10, MIN_SAMPLES_FOR_PERCENTILE=5. Per-school aggregation has sufficient data (all 23 schools ≥71 rows).

## Key distinction

- **School-level aggregation** (profiles, what-if): sufficient data
- **Individual prediction** (specific background → specific target): extreme sparsity, model extrapolates
- The CLAUDE.md "52,327 unique combos" number uses `background_major_original` instead of mapped `background_major`; verified count is 53,166 with original, 47,835 with mapped.

## Common mistakes to avoid

- Do NOT fabricate sample sizes (e.g., "68 samples" was wrong)
- Do NOT claim `paper_count`/`award_count` are missing from what-if — they're in `_build_input_data()`
- School profiles and what-if aggregation are at target-school level, NOT at 4-key combo level
