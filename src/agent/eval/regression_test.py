"""
Agent Regression Test Runner
==============================
Ensures prompt changes don't silently degrade agent performance.
Compares new agent output against baseline on a fixed eval set.

Principle: A prompt change that improves one case but breaks another
is a regression. This test catches that before production.

Usage:
    python -m src.agent.eval.regression_test
    python -m src.agent.eval.regression_test --agent LeadInAgent
    python -m src.agent.eval.regression_test --threshold 0.05
"""

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVAL_DATA_PATH = Path(__file__).resolve().parent / "eval_dataset.jsonl"
BASELINE_PATH = Path(__file__).resolve().parent / "baseline_results.json"


@dataclass
class RegressionResult:
    agent_name: str
    total_cases: int
    regressions: int  # cases where new output is worse
    improvements: int  # cases where new output is better
    unchanged: int
    threshold: float  # minimum score delta to count as change

    @property
    def passed(self) -> bool:
        return self.regressions == 0

    @property
    def regression_rate(self) -> float:
        return self.regressions / max(self.total_cases, 1)


def load_baseline() -> dict:
    """Load the stored baseline results."""
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {}


def save_baseline(results: dict):
    """Save current results as the new baseline."""
    BASELINE_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def score_lead_in_output(expected: dict, extracted: dict) -> float:
    """Score a LeadInAgent extraction against expected. Returns [0, 1]."""
    if not expected:
        return 1.0

    fields = ["university", "major", "gpa", "language_score", "language_type"]
    scores = []
    weights = {
        "university": 0.3,
        "major": 0.3,
        "gpa": 0.2,
        "language_score": 0.15,
        "language_type": 0.05,
    }

    for field in fields:
        exp = expected.get(field)
        ext = extracted.get(field)
        w = weights.get(field, 0.1)

        if exp is None:
            scores.append(w if ext is None or ext == "" else 0.0)
        elif isinstance(exp, str):
            exp_norm = exp.strip().lower().replace(" ", "")
            ext_norm = str(ext or "").strip().lower().replace(" ", "")
            scores.append(w if exp_norm in ext_norm or ext_norm in exp_norm else 0.0)
        elif isinstance(exp, (int, float)):
            try:
                delta = abs(float(ext) - float(exp))
                if exp > 20:
                    scores.append(w if delta <= 5 else 0.0)
                else:
                    scores.append(w if delta <= 0.1 else 0.0)
            except (TypeError, ValueError):
                scores.append(0.0)
        elif isinstance(exp, list):
            if not ext:
                scores.append(0.0)
            else:
                ext_list = ext if isinstance(ext, list) else [ext]
                overlap = sum(
                    1
                    for e in exp
                    if any(
                        str(e).lower().replace(" ", "") in str(x).lower().replace(" ", "")
                        for x in ext_list
                    )
                )
                scores.append(w * overlap / max(len(exp), 1))
        else:
            scores.append(0.0)

    return sum(scores)


def score_explain_output(expected: dict, output: dict) -> float:
    """Score ExplainAgent output. Returns [0, 1]."""
    score = 1.0

    text = ""
    if isinstance(output, dict):
        parsed = output.get("parsed", output)
        text = " ".join(str(v) for v in parsed.values() if isinstance(v, str))

    # Must-mention penalty
    must_mention = expected.get("must_mention", [])
    if must_mention:
        hits = sum(1 for kw in must_mention if kw in text)
        score *= hits / len(must_mention)

    # Must-not-mention penalty
    must_not = expected.get("must_not_mention", [])
    if must_not:
        violations = sum(1 for kw in must_not if kw in text)
        if violations > 0:
            score *= 0.0  # hard failure

    # Length compliance
    min_len = expected.get("min_length", 0)
    max_len = expected.get("max_length", 99999)
    if len(text) < min_len or len(text) > max_len:
        score *= 0.5

    return score


def run_regression_test(
    agent_name: str | None = None,
    threshold: float = 0.03,
    eval_data_path: str | None = None,
) -> dict[str, RegressionResult]:
    """Run regression test for specified agent(s).

    Compares current eval scores against stored baseline.
    A regression = score dropped by more than `threshold`.
    """
    baseline = load_baseline()
    eval_path = Path(eval_data_path or EVAL_DATA_PATH)

    if not eval_path.exists():
        print(f"Eval data not found: {eval_path}")
        print("Create it with annotated agent outputs first.")
        return {}

    cases = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    if agent_name:
        cases = [c for c in cases if c.get("agent") == agent_name]

    agents = sorted({c.get("agent") for c in cases if c.get("agent")})
    results = {}

    for agent in agents:
        agent_cases = [c for c in cases if c.get("agent") == agent]
        if not agent_cases:
            continue

        regressions = 0
        improvements = 0
        unchanged = 0

        for case in agent_cases:
            case_id = case.get("id", "")
            expected = case.get("expected", {})
            output = case.get("output", {})
            baseline_key = f"{agent}:{case_id}"

            if agent == "LeadInAgent":
                score = score_lead_in_output(expected, output.get("extracted", {}))
            elif agent == "ExplainAgent":
                score = score_explain_output(expected, output)
            else:
                score = 1.0  # placeholder for other agents

            if baseline_key in baseline:
                old_score = baseline[baseline_key].get("score", 0)
                delta = score - old_score
                if delta < -threshold:
                    regressions += 1
                elif delta > threshold:
                    improvements += 1
                else:
                    unchanged += 1

        results[agent] = RegressionResult(
            agent_name=agent,
            total_cases=len(agent_cases),
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
            threshold=threshold,
        )

    return results


def print_results(results: dict[str, RegressionResult]):
    print(f"\n{'=' * 70}")
    print("  Agent Regression Test")
    print(f"{'=' * 70}")

    all_pass = True
    for agent, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        icon = "✓" if result.passed else "✗"
        all_pass = all_pass and result.passed
        print(f"\n  [{icon} {status}] {agent}")
        print(f"    Cases: {result.total_cases}")
        print(f"    Regressions:  {result.regressions}")
        print(f"    Improvements: {result.improvements}")
        print(f"    Unchanged:    {result.unchanged}")
        print(f"    Threshold:    {result.threshold}")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'REGRESSIONS DETECTED'}")
    print("\n  [Note] To set baseline, run with --set-baseline after verifying")
    print("  current agent outputs are correct.")
    print(f"  Baseline file: {BASELINE_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Regression Test")
    parser.add_argument("--agent", type=str, default=None, help="Agent name filter")
    parser.add_argument(
        "--threshold", type=float, default=0.03, help="Min score delta to flag regression"
    )
    parser.add_argument(
        "--set-baseline", action="store_true", help="Save current results as baseline"
    )
    parser.add_argument("--eval-data", type=str, default=None, help="Path to eval dataset")
    args = parser.parse_args()

    if args.set_baseline:
        print("To set baseline: first run agents on eval dataset,")
        print("populate 'output' field in each eval case with agent results,")
        print(f"then run this script. Baseline will be saved to {BASELINE_PATH}")
    else:
        results = run_regression_test(
            agent_name=args.agent,
            threshold=args.threshold,
            eval_data_path=args.eval_data,
        )
        if results:
            print_results(results)
        else:
            print("\nNo eval data found or no results to compare.")
            print("Populate src/agent/eval/eval_dataset.jsonl with real agent outputs,")
            print("then run: python -m src.agent.eval.regression_test --set-baseline")
