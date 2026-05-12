"""
Counterfactual + Baseline 数据生成

为 trace 可视化提供两类辅助数据：

- baseline_admit_rate: 历史 (target_university, target_major) 的平均录取率，作为
  瀑布图的对比锚点。回答"这个 case 比平均高还是低"。

- counterfactuals: 在 GPA / 语言 / 实习 维度做小幅扰动，重跑核心调整链
  （不含 text boost），输出每个扰动场景下的最终概率。回答"如果背景再
  好/坏一点，概率会怎么变"。

设计取舍：
1. counterfactual 跳过 text_boost——文本提升与 GPA/Lang/Intern 解耦，
   重跑只是浪费且会污染 attribution。
2. 每个扰动重新计算 GPA/Lang penalty（因为它们依赖 z-score），
   重新构造静态 arbitrator factors，复用 pipeline.adjust_single 的动态层。
3. 仅对 batch 中 top N 计算（按 XGBoost 原始概率排序），避免对长尾结果
   做无意义的扰动。
"""

import dataclasses
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.pages.prediction.result_modifier.arbitrator import AdjustmentArbitrator
from src.pages.prediction.result_modifier.types import (
    AdjustmentFactor,
    AdjustmentFactorType,
)

if TYPE_CHECKING:
    from src.pages.prediction.result_modifier.adjustment_pipeline import (
        AdjustmentContext,
        ProbabilityAdjustmentPipeline,
    )

GPA_PERTURBATION = 0.2
LANG_PERTURBATION = 0.05
INTERN_PERTURBATION = 1
COUNTERFACTUAL_TOP_N = 3


def compute_baseline_admit_rates(
    cases_df: pd.DataFrame | None,
    results: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int]]:
    if cases_df is None or cases_df.empty:
        return {}, {}

    required = ["target_university", "target_major", "admitted"]
    if not all(col in cases_df.columns for col in required):
        return {}, {}

    targets = {
        (str(r.get("university")), str(r.get("major")))
        for r in results
        if r.get("university") and r.get("major")
    }
    if not targets:
        return {}, {}

    grouped = cases_df.groupby(["target_university", "target_major"])["admitted"].agg(
        ["mean", "count"]
    )
    rates = {}
    counts = {}
    for (univ, major), row in grouped.iterrows():
        key = (str(univ), str(major))
        if key in targets:
            rates[key] = float(row["mean"])
            counts[key] = int(row["count"])
    return rates, counts


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
        return {}

    scenarios = _build_scenarios(ctx)
    out: dict[str, float] = {}

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

    return out


def attach_trace_extras(
    pipeline: "ProbabilityAdjustmentPipeline",
    results: list[dict[str, Any]],
    ctx: "AdjustmentContext",
    original_probs: list[float],
) -> None:
    baseline_map, baseline_counts = compute_baseline_admit_rates(ctx.cases_df, results)

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
        key=lambda i: float(results[i].get("probability", 0.0) or 0.0),
        reverse=True,
    )
    for i in indexed[:COUNTERFACTUAL_TOP_N]:
        cf = compute_counterfactuals(pipeline, results[i], ctx, original_probs[i])
        if cf:
            results[i]["_counterfactuals"] = cf
