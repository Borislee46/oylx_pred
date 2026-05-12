"""
LeadInAgent Evaluation: Field Extraction Metrics
==================================================
Measures how accurately the agent extracts structured fields from
free-text advisor input. Core metric: per-field F1 / exact match.

Usage:
    python -m src.agent.eval.lead_in_metrics --eval-data src/agent/eval/eval_dataset.jsonl
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldScore:
    exact_match: int = 0
    partial_match: int = 0
    total: int = 0
    false_positives: int = 0  # extracted a value when expected was empty

    @property
    def precision(self) -> float:
        denom = self.exact_match + self.partial_match + self.false_positives
        return (self.exact_match + self.partial_match) / max(denom, 1)

    @property
    def recall(self) -> float:
        return (self.exact_match + self.partial_match) / max(self.total, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class EvalReport:
    agent_name: str
    total_cases: int
    field_scores: dict[str, FieldScore] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)

    @property
    def overall_f1(self) -> float:
        if not self.field_scores:
            return 0.0
        return sum(s.f1 for s in self.field_scores.values()) / len(self.field_scores)


def normalize(s: str | None) -> str:
    if not s:
        return ""
    return s.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def fuzzy_contains(expected: str, extracted: str) -> bool:
    """Check if expected is contained within extracted (or vice versa)."""
    e = normalize(expected)
    x = normalize(extracted)
    if not e or not x:
        return False
    return e in x or x in e


def gpa_match(expected: float | None, extracted: Any) -> bool:
    """GPA matching: allow ±0.1 tolerance for 4.0-scale, or ratio for 100-scale."""
    if expected is None:
        return extracted is None
    try:
        ext_val = float(extracted)
    except (TypeError, ValueError):
        return False
    exp_val = float(expected)
    if exp_val > 10:  # 100-scale
        return abs(ext_val - exp_val) <= 5.0
    return abs(ext_val - exp_val) <= 0.1


def language_score_match(expected: float | None, extracted: Any) -> bool:
    """Language score: ±0.5 tolerance for IELTS, ±5 for TOEFL."""
    if expected is None:
        return extracted is None
    try:
        ext_val = float(extracted)
    except (TypeError, ValueError):
        return False
    exp_val = float(expected)
    if exp_val > 20:  # TOEFL scale
        return abs(ext_val - exp_val) <= 5.0
    return abs(ext_val - exp_val) <= 0.5


def list_overlap(expected: list[str] | None, extracted: list[str] | None) -> tuple[int, int]:
    """Count matches and total for list fields (schools, majors)."""
    if not expected:
        return 0, 0
    if not extracted:
        return 0, len(expected)
    matched = 0
    for exp_item in expected:
        for ext_item in extracted:
            if fuzzy_contains(exp_item, ext_item):
                matched += 1
                break
    return matched, len(expected)


def evaluate_lead_in(cases: list[dict]) -> EvalReport:
    """Evaluate LeadInAgent extraction quality.

    cases: list of {id, input, expected, extracted}
    """
    report = EvalReport(agent_name="LeadInAgent", total_cases=len(cases))

    scalar_fields = [
        ("university", "exact"),
        ("major", "fuzzy"),
        ("gpa", "gpa"),
        ("language_score", "lang"),
        ("language_type", "exact"),
    ]
    list_fields = ["target_schools", "target_majors"]
    optional_text_fields = ["research", "internship", "award", "paper"]

    for field_name, _match_type in scalar_fields:
        report.field_scores[field_name] = FieldScore()
        report.field_scores[field_name].total = sum(
            1 for c in cases if c.get("expected", {}).get(field_name) is not None
        )

    for field_name in list_fields:
        report.field_scores[field_name] = FieldScore()
        # List field totals are accumulated per-item, not per-case

    for field_name in optional_text_fields:
        report.field_scores[field_name] = FieldScore()
        report.field_scores[field_name].total = sum(
            1 for c in cases if c.get("expected", {}).get(field_name)
        )

    for case in cases:
        expected = case.get("expected", {})
        extracted = case.get("extracted", {})

        for field_name, match_type in scalar_fields:
            exp_val = expected.get(field_name)
            ext_val = extracted.get(field_name)
            fs = report.field_scores[field_name]

            if exp_val is None:
                if ext_val is not None and ext_val != "" and ext_val != []:
                    fs.false_positives += 1
                continue

            if match_type == "exact":
                if normalize(str(ext_val)) == normalize(str(exp_val)):
                    fs.exact_match += 1
            elif match_type == "fuzzy":
                if fuzzy_contains(str(exp_val), str(ext_val)):
                    fs.exact_match += 1
            elif match_type == "gpa":
                if gpa_match(exp_val, ext_val):
                    fs.exact_match += 1
            elif match_type == "lang":
                if language_score_match(exp_val, ext_val):
                    fs.exact_match += 1

        for field_name in list_fields:
            exp_list = expected.get(field_name, [])
            ext_list = extracted.get(field_name, [])
            if not isinstance(ext_list, list):
                ext_list = [ext_list] if ext_list else []
            if not isinstance(exp_list, list):
                exp_list = [exp_list] if exp_list else []

            fs = report.field_scores[field_name]
            if exp_list:
                matched, total_expected = list_overlap(exp_list, ext_list)
                # exact = correctly found items, total = all expected items
                fs.exact_match += matched
                fs.total += total_expected
                # False positives: extracted items that don't match any expected
                fp_count = max(0, len(ext_list) - matched)
                fs.false_positives += fp_count
            elif ext_list:
                # No expected items, but agent extracted some → all are false positives
                fs.false_positives += len(ext_list)

        for field_name in optional_text_fields:
            exp_val = expected.get(field_name, "")
            ext_val = extracted.get(field_name, "")
            fs = report.field_scores[field_name]
            if exp_val and ext_val:
                if fuzzy_contains(str(exp_val), str(ext_val)):
                    fs.exact_match += 1
            elif exp_val and not ext_val:
                pass  # false negative
            elif not exp_val and ext_val:
                fs.false_positives += 1
            elif not exp_val and not ext_val:
                fs.exact_match += 1  # correctly empty

    return report


def print_report(report: EvalReport):
    print(f"\n{'=' * 60}")
    print(f"  {report.agent_name} — Extraction Quality Report")
    print(f"  {report.total_cases} eval cases")
    print(f"{'=' * 60}")
    print(f"\n{'Field':<22} {'P':>8} {'R':>8} {'F1':>8} {'Exact':>8} {'Total':>8}")
    print(f"{'-' * 62}")
    for field_name, fs in sorted(report.field_scores.items()):
        print(
            f"{field_name:<22} {fs.precision:>8.3f} {fs.recall:>8.3f} "
            f"{fs.f1:>8.3f} {fs.exact_match:>8} {fs.total:>8}"
        )
    print(f"{'-' * 62}")
    print(f"{'OVERALL (macro avg)':<22} {'':>8} {'':>8} {report.overall_f1:>8.3f}")


def load_eval_cases(path: str, agent_filter: str | None = None) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if agent_filter and case.get("agent") != agent_filter:
                continue
            cases.append(case)
    return cases


if __name__ == "__main__":
    # Demo: run with mock extracted data
    demo_cases = [
        {
            "expected": {
                "university": "清华大学",
                "major": "计算机科学",
                "gpa": 3.8,
                "language_score": 7.0,
                "language_type": "雅思",
                "target_schools": ["香港大学", "香港中文大学"],
                "target_majors": ["数据科学"],
            },
            "extracted": {
                "university": "清华大学",
                "major": "计算机科学与技术",
                "gpa": 3.8,
                "language_score": 7.0,
                "language_type": "雅思",
                "target_schools": ["香港大学", "香港中文大学", "香港科技大学"],
                "target_majors": ["数据科学"],
            },
        },
        {
            "expected": {
                "university": "双非一本",
                "major": "软件工程",
                "gpa": 3.4,
                "language_score": 100.0,
                "language_type": "托福",
                "target_schools": ["新加坡国立大学", "新加坡南洋理工大学"],
                "target_majors": ["人工智能"],
            },
            "extracted": {
                "university": "北京联合大学",
                "major": "软件工程",
                "gpa": 3.5,
                "language_score": 100.0,
                "language_type": "托福",
                "target_schools": ["新加坡国立大学"],
                "target_majors": ["计算机科学"],
            },
        },
    ]
    report = evaluate_lead_in(demo_cases)
    print_report(report)
    print("\n[Note] This is a demo with mock extracted data.")
    print("  To evaluate real agent output: update cases with actual extracted fields.")
    print("  Eval dataset: src/agent/eval/eval_dataset.jsonl")
