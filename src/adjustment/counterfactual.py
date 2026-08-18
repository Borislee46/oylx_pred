import dataclasses
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.adjustment.admission_cache import get_baseline_admit_lookup
from src.adjustment.arbitrator import AdjustmentArbitrator
from src.adjustment.engine import (
    AdjustmentFactor,
    AdjustmentFactorType,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce

logger = setup_logger("page3", "prediction")

if TYPE_CHECKING:
    from src.adjustment.adjustment_pipeline import (
        AdjustmentContext,
        ProbabilityAdjustmentPipeline,
    )

GPA_PERTURBATION = 0.2
LANG_PERTURBATION = 0.05
INTERN_PERTURBATION = 1
COUNTERFACTUAL_TOP_N = 3


def _target_keys(results: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(r.get("university")), str(r.get("major")))
        for r in results
        if r.get("university") and r.get("major")
    }


def lookup_baseline_admit_rates(
    lookup: dict[tuple[str, str], tuple[float, int]],
    results: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int]]:
    targets = _target_keys(results)
    if not targets or not lookup:
        return {}, {}

    rates: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for key in targets:
        entry = lookup.get(key)
        if entry is None:
            continue
        rates[key] = entry[0]
        counts[key] = entry[1]
    return rates, counts


def compute_baseline_admit_rates(
    cases_df: pd.DataFrame | None,
    results: list[dict[str, Any]],
    lookup: dict[tuple[str, str], tuple[float, int]] | None = None,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int]]:
    if lookup is None:
        if cases_df is None or cases_df.empty:
            return {}, {}
        lookup = get_baseline_admit_lookup(cases_df)
    return lookup_baseline_admit_rates(lookup, results)


def _add_static_penalties(
    arbitrator: AdjustmentArbitrator,
    gpa_penalty: float,
    lang_penalty: float,
) -> None:
    if gpa_penalty > 0:
        arbitrator.add_factor(
            AdjustmentFactor(
                name="GPA Penalty",
                value=gpa_penalty,
                factor_type=AdjustmentFactorType.PENALTY,
                description="GPA成绩影响",
            ),
            is_static=True,
        )
    if lang_penalty > 0:
        arbitrator.add_factor(
            AdjustmentFactor(
                name="Language Penalty",
                value=lang_penalty,
                factor_type=AdjustmentFactorType.PENALTY,
                description="语言成绩影响",
            ),
            is_static=True,
        )


def _build_scenarios(ctx: "AdjustmentContext") -> dict[str, tuple[float, float, int]]:
    return {
        "origin": (ctx.gpa, ctx.language_score, ctx.internship_count),
        "gpa_up": (
            min(4.0, ctx.gpa + GPA_PERTURBATION),
            ctx.language_score,
            ctx.internship_count,
        ),
        "gpa_down": (
            max(0.0, ctx.gpa - GPA_PERTURBATION),
            ctx.language_score,
            ctx.internship_count,
        ),
        "lang_up": (
            ctx.gpa,
            min(1.0, ctx.language_score + LANG_PERTURBATION),
            ctx.internship_count,
        ),
        "intern_up": (
            ctx.gpa,
            ctx.language_score,
            ctx.internship_count + INTERN_PERTURBATION,
        ),
    }


def compute_counterfactuals(
    pipeline: "ProbabilityAdjustmentPipeline",
    result: dict[str, Any],
    ctx: "AdjustmentContext",
    original_prob: float,
) -> dict[str, float]:
    if pipeline.probability_adjuster is None or ctx.gpa is None or ctx.language_score is None:
        logger.debug(
            "反事实跳过 | adjuster=%s gpa=%s lang=%s",
            pipeline.probability_adjuster is not None,
            ctx.gpa,
            ctx.language_score,
        )
        return {}

    scenarios = _build_scenarios(ctx)
    out: dict[str, float] = {}

    univ = result.get("university", "?")
    major = result.get("major", "?")
    for key, (gpa, lang, intern) in scenarios.items():
        cf_arbitrator = AdjustmentArbitrator(include_trace=False)
        penalties = pipeline.probability_adjuster.get_penalties(gpa, lang)
        _add_static_penalties(
            cf_arbitrator,
            penalties.get("gpa", 0),
            penalties.get("language", 0),
        )

        cf_ctx = dataclasses.replace(
            ctx,
            gpa=gpa,
            language_score=lang,
            internship_count=intern,
        )

        cf_result = {k: v for k, v in result.items() if not k.startswith("_adjustment")}
        cf_result["probability"] = original_prob

        adjusted = pipeline.adjust_single(cf_result, cf_ctx, cf_arbitrator)
        out[key] = float(adjusted.get("probability", original_prob))

    logger.debug(
        "反事实模拟 | %s@%s origin=%.4f gpa_up=%.4f gpa_down=%.4f lang_up=%.4f intern_up=%.4f",
        major,
        univ,
        out.get("origin", original_prob),
        out.get("gpa_up", 0),
        out.get("gpa_down", 0),
        out.get("lang_up", 0),
        out.get("intern_up", 0),
    )
    return out


def attach_trace_extras(
    pipeline: "ProbabilityAdjustmentPipeline",
    results: list[dict[str, Any]],
    ctx: "AdjustmentContext",
    original_probs: list[float],
) -> None:
    baseline_map, baseline_counts = compute_baseline_admit_rates(
        ctx.cases_df, results, lookup=ctx.baseline_admit_lookup
    )

    for r in results:
        univ, major = r.get("university"), r.get("major")
        if univ and major:
            rate = baseline_map.get((str(univ), str(major)))
            if rate is not None:
                r["_baseline_admit_rate"] = rate
                r["_baseline_sample_count"] = baseline_counts.get((str(univ), str(major)), 0)

    if pipeline.probability_adjuster is None or ctx.gpa is None or ctx.language_score is None:
        return

    indexed = sorted(
        range(len(results)),
        key=lambda i: clip_probability_coerce(results[i].get("probability")),
        reverse=True,
    )
    for i in indexed[:COUNTERFACTUAL_TOP_N]:
        cf = compute_counterfactuals(pipeline, results[i], ctx, original_probs[i])
        if cf:
            results[i]["_counterfactuals"] = cf
