"""Agent evaluation framework.

Module layout:
    eval_dataset.jsonl   — annotated eval cases (grow over time)
    lead_in_metrics.py   — LeadInAgent field extraction F1
    explain_metrics.py   — ExplainAgent quality & safety checks
    regression_test.py   — prompt regression detection
"""

from src.agent.eval.explain_metrics import ExplainEvalReport, evaluate_explain
from src.agent.eval.lead_in_metrics import EvalReport, FieldScore, evaluate_lead_in
from src.agent.eval.regression_test import (
    RegressionResult,
    run_regression_test,
    score_explain_output,
    score_lead_in_output,
)

__all__ = [
    "evaluate_lead_in",
    "evaluate_explain",
    "run_regression_test",
    "EvalReport",
    "ExplainEvalReport",
    "RegressionResult",
    "FieldScore",
    "score_lead_in_output",
    "score_explain_output",
]
