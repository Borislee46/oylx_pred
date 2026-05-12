# Agent Evaluation Framework

## Why

We have 5 LLM agents in production. When we change a prompt, we need to know: did it actually improve things, or did it silently break edge cases? LLM outputs are non-deterministic, so a single manual test is insufficient.

## Architecture

```
eval_dataset.jsonl    ← annotated eval cases (you build this)
    │
    ├── lead_in_metrics.py   ← per-field extraction F1
    ├── explain_metrics.py   ← keyword coverage, length, safety
    └── regression_test.py   ← prompt A vs prompt B comparison
```

## How to Build Eval Data

1. Collect 20-30 real advisor inputs that cover the diversity of your user base.
2. Run the agent on each input, save the output.
3. Manually annotate the `expected` field — what the agent SHOULD have produced.
4. Put each case in `eval_dataset.jsonl`.

Start small (10 cases per agent). Grow as you discover failure modes.

## How to Run

```bash
# LeadInAgent: measure field extraction quality
python -m src.agent.eval.lead_in_metrics

# ExplainAgent: check for forbidden phrases, length compliance
python -m src.agent.eval.explain_metrics

# Regression test: did a prompt change break anything?
python -m src.agent.eval.regression_test --set-baseline   # first time
python -m src.agent.eval.regression_test                  # after prompt change
```

## Metrics

### LeadInAgent

| Metric | What it measures |
|--------|-----------------|
| Per-field F1 | Did the agent correctly extract university, major, GPA, language_score? |
| List overlap | For target_schools, target_majors — how many expected items were found? |
| False positive rate | Did the agent hallucinate values where none exist? |

### ExplainAgent

| Metric | What it measures |
|--------|-----------------|
| JSON parse success | Can the output be parsed? (tier 1-4 repair applied) |
| Must-mention coverage | Are key insights present? (e.g., "高录取概率", "背景匹配") |
| Must-not-mention violations | Are forbidden phrases absent? (e.g., "无希望") |
| Length compliance | Is output within min/max length bounds? |

### Regression Test

| Result | Meaning |
|--------|---------|
| No regressions | New prompt is at least as good as baseline on all cases |
| N regressions | N cases got worse after prompt change — investigate before deploying |
| M improvements | M cases improved — expected if the prompt change was intentional |

## Interview Narrative

> "I built an eval framework for my 5 LLM agents. Each agent has annotated test cases, per-agent metrics, and a regression test that blocks prompt changes from silently degrading. The regression test catches Goodhart failures — when a prompt change improves the eval metric but hurts actual quality on edge cases."

## Current State

- [x] Framework code complete
- [ ] Eval dataset populated with real agent outputs
- [ ] Baseline established (`baseline_results.json`)
- [ ] Integrated into CI (future)
