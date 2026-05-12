"""
ExplainAgent Evaluation: Quality & Safety Checks
==================================================
Measures: keyword coverage, length compliance, JSON parse success rate,
and safety constraint violations (must mention / must not mention).

Usage:
    python -m src.agent.eval.explain_metrics --eval-data src/agent/eval/eval_dataset.jsonl
"""

from dataclasses import dataclass, field


@dataclass
class ExplainEvalReport:
    agent_name: str = "ExplainAgent"
    total_cases: int = 0
    passed: int = 0
    failures: list[dict] = field(default_factory=list)

    # Aggregate metrics
    json_parse_success: int = 0
    json_parse_failure: int = 0
    length_violations: int = 0
    must_mention_violations: int = 0
    must_not_mention_violations: int = 0
    overall_pass_rate: float = 0.0

    def finalize(self):
        self.overall_pass_rate = self.passed / max(self.total_cases, 1)


def evaluate_explain(cases: list[dict]) -> ExplainEvalReport:
    """Evaluate ExplainAgent output against expected constraints.

    Each case has:
      - input: {background, predictions}
      - expected: {must_mention, must_not_mention, min_length, max_length}
      - output: {explanation_json_string_or_dict}
    """
    report = ExplainEvalReport(total_cases=len(cases))

    for case in cases:
        expected = case.get("expected", {})
        output = case.get("output", {})
        case_id = case.get("id", "unknown")
        failures = []

        raw_output = output.get("raw", "") if isinstance(output, dict) else str(output)
        parsed = output.get("parsed", {}) if isinstance(output, dict) else {}

        # Check 1: JSON parse success
        if parsed or (isinstance(output, dict) and output.get("parsed")):
            report.json_parse_success += 1
        elif raw_output and not parsed:
            report.json_parse_failure += 1
            failures.append("json_parse_failed")

        # Check 2: Length compliance
        text = ""
        if parsed:
            text = " ".join(str(v) for v in parsed.values() if isinstance(v, str))
        elif raw_output:
            text = raw_output
        text_len = len(text)
        min_len = expected.get("min_length", 0)
        max_len = expected.get("max_length", 9999)
        if text_len < min_len:
            report.length_violations += 1
            failures.append(f"too_short({text_len}<{min_len})")
        if text_len > max_len:
            report.length_violations += 1
            failures.append(f"too_long({text_len}>{max_len})")

        # Check 3: Must-mention keywords
        for keyword in expected.get("must_mention", []):
            if keyword not in text:
                report.must_mention_violations += 1
                failures.append(f"missing_keyword:{keyword}")

        # Check 4: Must-not-mention keywords
        for keyword in expected.get("must_not_mention", []):
            if keyword in text:
                report.must_not_mention_violations += 1
                failures.append(f"forbidden_keyword:{keyword}")

        if not failures:
            report.passed += 1
        else:
            report.failures.append({"id": case_id, "failures": failures})

    report.finalize()
    return report


def print_report(report: ExplainEvalReport):
    print(f"\n{'=' * 60}")
    print(f"  {report.agent_name} — Quality & Safety Report")
    print(f"  {report.total_cases} eval cases")
    print(f"{'=' * 60}")
    print(
        f"\n  Overall pass rate:     {report.overall_pass_rate:.1%} ({report.passed}/{report.total_cases})"
    )
    print(f"  JSON parse success:    {report.json_parse_success}/{report.total_cases}")
    print(f"  JSON parse failure:    {report.json_parse_failure}/{report.total_cases}")
    print(f"  Length violations:     {report.length_violations}")
    print(f"  Must-mention missing:  {report.must_mention_violations}")
    print(f"  Forbidden mentioned:   {report.must_not_mention_violations}")

    if report.failures:
        print("\n  Failed cases:")
        for f in report.failures[:5]:
            print(f"    [{f['id']}] {', '.join(f['failures'])}")

    print("\n  [Note] This is a framework. Populate eval_dataset.jsonl with")
    print("  real agent outputs to get actual metrics.")


if __name__ == "__main__":
    # Demo with mock data
    demo_cases = [
        {
            "id": "explain_001",
            "expected": {
                "must_mention": ["高录取概率"],
                "must_not_mention": ["无希望"],
                "min_length": 80,
                "max_length": 500,
            },
            "output": {
                "parsed": {
                    "overview": "你的背景在目标专业中具有很强的匹配度，清华CS的学术训练结合3.8的GPA，为你申请数据科学方向提供了坚实基础。",
                    "strengths": "GPA 3.8处于竞争区间上沿，字节跳动的NLP实习经历加分明显，香港大学对清华毕业生认可度高。",
                    "concerns": "数据科学方向竞争激烈，建议补充相关项目经历。",
                    "summary": "整体录取概率在58%-72%区间，属于冲刺-匹配档，建议重点准备港大和港中文的申请材料。",
                }
            },
        },
        {
            "id": "explain_002",
            "expected": {
                "must_mention": ["挑战", "提升"],
                "must_not_mention": [],
                "min_length": 80,
                "max_length": 500,
            },
            "output": {
                "parsed": {
                    "overview": "你的背景在数据科学方向面临较大挑战，需要从多个维度提升竞争力。",
                    "strengths": "",
                    "concerns": "GPA和语言成绩偏低",
                    "summary": "建议提升语言成绩至6.5+，同时补充相关实习经历以提高申请成功率。",
                }
            },
        },
    ]
    report = evaluate_explain(demo_cases)
    print_report(report)
