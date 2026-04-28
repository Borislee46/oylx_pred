"""WritePrint — AI text detection for personal statements."""

from src.pages.write_print.engine import analyze
from src.pages.write_print.model import get_model, fit_model, print_fit_report
from src.pages.write_print.features import compute_features, read_input
from src.pages.write_print.cli import diagnose
from src.pages.write_print.rewriter import generate_full_rewrite, generate_sentence_rewrites

__all__ = [
    "analyze", "diagnose", "get_model", "fit_model", "print_fit_report",
    "compute_features", "read_input", "generate_full_rewrite", "generate_sentence_rewrites",
]
