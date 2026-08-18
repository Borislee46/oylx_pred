import random
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from src.adjustment import (
    AdjustmentContext,
    ProbabilityAdjustmentPipeline,
)
from src.adjustment.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.adjustment.config import (
    DEFAULT_TEXT_BOOST_CONFIG,
    DEFAULT_UNIVERSITY_DIFFICULTY_ORDER,
    load_adjustment_flags,
)
from src.adjustment.fallback import (
    compute_fallback_probabilities,
)
from src.adjustment.probability_adjuster import (
    ProbabilityAdjuster,
)
from src.adjustment.text_boost_provider import (
    get_text_boost_provider,
)
from src.adjustment.tier_calibration import (
    apply_tier_rank_repair,
    build_tier_map,
)
from src.adjustment.utils import has_any_experience
from src.pages.prediction.app_data import load_bg_target_similarity_cache
from src.pages.prediction.core.ui_messages import (
    PIPELINE_MESSAGES,
    PIPELINE_PHASE_MAP,
    format_pipeline_done_progress,
    format_pipeline_empty_progress,
    format_pipeline_prep_progress,
    format_pipeline_refine_progress,
)
from src.pages.prediction.core.utils import get_background_faculty
from src.pages.prediction.flow.preparer import (
    validate_and_clean_input,
)
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.flow.run_prediction import (
    run_single_prediction,
)
from src.pages.prediction.page_data_loader import (
    cached_get_prediction_model,
    machine_learning_model,
)
from src.pages.prediction.results_handler import (
    combine_and_deduplicate_results,
    combine_and_deduplicate_results_with_sources,
)
from src.utils.analytics import track as _track
from src.utils.logger import setup_logger
from src.utils.session_manager import PredictionResultModel

# ProbabilityAdjustmentPipeline.__init__ 接受的 flag 键。
# enable_language_requirement_penalty 等链外开关由 run_single_prediction /
# result_processor 显式消费，不能经 **adjustment_flags 透传（否则 TypeError）。
_ADJUSTMENT_PIPELINE_FLAG_KEYS = frozenset(
    {
        "enable_gpa_penalty",
        "enable_language_penalty",
        "enable_cross_major_penalty",
        "enable_cross_faculty_penalty",
        "enable_professional_penalty",
        "enable_text_boost",
        "ablation_tag",
    }
)


def _adjustment_pipeline_kwargs(flags: dict[str, bool]) -> dict[str, bool]:
    return {k: v for k, v in flags.items() if k in _ADJUSTMENT_PIPELINE_FLAG_KEYS}


prediction_handler_logger = setup_logger("page3", "prediction")

ProgressCallback = Callable[[str], None]


def _get_adjustment_flags(overrides: dict[str, bool] | None = None) -> dict[str, bool]:
    flags = load_adjustment_flags()
    if overrides:
        flags.update({k: v for k, v in overrides.items() if k in flags})
    return flags


def _execute_prediction_pipeline(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    all_universities_target: list[str],
    all_majors_target: list[str],
    reporter: ProgressReporter,
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
    page_state: machine_learning_model | None = None,
    cached_combinations: list[tuple[str, str]] | None = None,
    adjustment_flag_overrides: dict[str, bool] | None = None,
) -> PredictionResultModel:
    t_start = time.monotonic()
    adjustment_flags = _get_adjustment_flags(adjustment_flag_overrides)
    prediction_handler_logger.info(
        "Pipe 启动 | model=%s targets=%d features=%d fingerprint=%d cache=%s",
        model_name,
        len(all_universities_target),
        len(loaded_feature_names),
        cases_df_fingerprint,
        bool(cached_combinations),
    )

    prediction_model = cached_get_prediction_model(model_name)

    if prediction_model is None:
        return PredictionResultModel(meta={"error": "model_load_failed"})

    if page_state is None:
        page_state = machine_learning_model.resource_loader()
    cases_df = page_state.cases_df

    if cases_df_fingerprint != page_state.cases_df_fingerprint:
        prediction_handler_logger.warning(
            f"案例数据指纹不匹配: 期望 {cases_df_fingerprint}, 实际 {page_state.cases_df_fingerprint}"
        )
        return PredictionResultModel(
            meta={
                "error": "cases_df_fingerprint_mismatch",
                "expected_cases_df_fingerprint": cases_df_fingerprint,
                "actual_cases_df_fingerprint": page_state.cases_df_fingerprint,
            }
        )

    cleaned_input = validate_and_clean_input(input_data)
    current_input_data = {**input_data, **cleaned_input}
    bg_target_similarity_cache = load_bg_target_similarity_cache()

    reporter.emit(
        format_pipeline_prep_progress(
            bg_university=cleaned_input.get("background_university"),
            bg_major=cleaned_input.get("background_major"),
            language_type=cleaned_input.get("language_type"),
            language_score=cleaned_input.get("language_score"),
            target_universities=cleaned_input.get("target_universities"),
            similarity_cache_loaded=bool(bg_target_similarity_cache),
        ),
        force=True,
        phase=PIPELINE_PHASE_MAP["init_engine"],
    )

    num_target_universities = len(cleaned_input.get("target_universities", []))
    cross_faculty_confirmed = input_data.get("_cross_faculty_confirmed", False)

    gpa = cleaned_input.get("gpa")
    language_score = cleaned_input.get("language_score")
    background_university = cleaned_input.get("background_university")
    background_major = cleaned_input.get("background_major", "")

    probability_adjuster = (
        ProbabilityAdjuster(
            cases_df if cases_df is not None else pd.DataFrame(),
            data_hash=cases_df_fingerprint,
        )
        if gpa is not None and language_score is not None
        else None
    )

    sim_results, cross_results, user_specified_results, meta = run_single_prediction(
        current_input_data=current_input_data,
        prediction_model=prediction_model,
        cases_df=cases_df,
        bg_target_similarity_cache=bg_target_similarity_cache,
        expected_features=loaded_feature_names,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        num_target_universities=num_target_universities,
        cross_faculty_confirmed=cross_faculty_confirmed,
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        language_type=cleaned_input.get("language_type"),
        background_university=background_university,
        progress_reporter=reporter,
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
        page_state=page_state,
        cached_combinations=cached_combinations,
        enable_language_requirement_penalty=adjustment_flags.get(
            "enable_language_requirement_penalty", True
        ),
    )

    if meta and meta.get("fallback_eligible") and meta.get("_fallback_combinations"):
        prediction_handler_logger.info(
            "触发人口统计兜底 | missing=%s combinations=%d",
            meta.get("missing_features"),
            len(meta["_fallback_combinations"]),
        )
        fallback_combinations = meta["_fallback_combinations"]
        fallback_results = compute_fallback_probabilities(
            fallback_combinations,
            cases_df,
            background_university or "",
            background_major,
            similarity_scores=bg_target_similarity_cache,
        )
        if fallback_results:
            _fb_text_provider = (
                get_text_boost_provider(DEFAULT_TEXT_BOOST_CONFIG)
                if input_data.get("_has_valid_experience")
                else None
            )
            _fb_quality_verifier = None
            if _fb_text_provider is not None:
                try:
                    from src.agent.text_preprocessing_agent import TextPreprocessingAgent

                    _fb_qa = TextPreprocessingAgent()
                    _fb_quality_verifier = _fb_qa.validate_quality_batch
                except Exception:
                    pass
            pipeline = ProbabilityAdjustmentPipeline(
                probability_adjuster=None,
                text_boost_provider=_fb_text_provider,
                quality_verifier=_fb_quality_verifier,
                **_adjustment_pipeline_kwargs(adjustment_flags),
            )
            bg_faculty_fb = (
                background_faculty
                or current_input_data.get("faculty")
                or get_background_faculty(background_major, cases_df)
            )
            fb_adj_ctx = AdjustmentContext(
                gpa=gpa,
                language_score=language_score,
                background_university=background_university,
                background_major=background_major,
                background_faculty=bg_faculty_fb,
                internship_count=cleaned_input.get("internship_count", 0),
                user_specified_majors=cleaned_input.get("target_majors", []),
                experience_details=cleaned_input.get("experience_details", {}),
                cases_df=cases_df,
                admitted_combinations=admitted_combinations,
            )
            fallback_results = pipeline.adjust_batch(
                fallback_results,
                fb_adj_ctx,
                progress_reporter=reporter,
                batch_tag="兜底估算",
            )
            unified = combine_and_deduplicate_results(fallback_results, [], None)
            meta["fallback_level"] = min(
                (r.get("_fallback_level", 4) for r in fallback_results),
                default=4,
            )
            reporter.emit(
                format_pipeline_done_progress(),
                force=True,
                phase=PIPELINE_PHASE_MAP["done"],
            )
            prediction_handler_logger.info(
                "Pipe 完成(Fallback) | fallback=%d fallback_level=%d total_elapsed=%.3fs",
                len(fallback_results),
                meta.get("fallback_level", -1),
                time.monotonic() - t_start,
            )
            return PredictionResultModel(
                similarity_results=fallback_results,
                cross_major_results=[],
                user_specified_results=None,
                unified_results=unified,
                meta=meta,
            )

    if meta and meta.get("error"):
        prediction_handler_logger.info(
            "Pipe 异常终止 | error=%s total_elapsed=%.3fs",
            meta.get("error"),
            time.monotonic() - t_start,
        )
        return PredictionResultModel(meta=meta)

    admitted_combos = (
        admitted_combinations
        if admitted_combinations is not None
        else get_admitted_combinations_from_dataframe(cases_df, background_major)
    )

    bg_faculty = (
        background_faculty
        if background_faculty
        else (
            current_input_data.get("faculty") or get_background_faculty(background_major, cases_df)
        )
    )

    route_labels: list[str] = []
    if sim_results:
        route_labels.append("相似专业")
    if cross_results:
        route_labels.append("跨专业")
    if user_specified_results:
        route_labels.append("用户指定")

    reporter.emit(
        format_pipeline_refine_progress(
            route_labels=route_labels,
            bg_faculty=bg_faculty if isinstance(bg_faculty, str) else None,
            soft_background_on=bool(input_data.get("_has_valid_experience")),
        ),
        force=True,
        phase=PIPELINE_PHASE_MAP["initial_filter"],
    )

    adj_ctx = AdjustmentContext(
        gpa=gpa,
        language_score=language_score,
        background_university=background_university,
        background_major=background_major,
        background_faculty=bg_faculty,
        internship_count=cleaned_input.get("internship_count", 0),
        user_specified_majors=cleaned_input.get("target_majors", []),
        experience_details=cleaned_input.get("experience_details", {}),
        cases_df=cases_df,
        admitted_combinations=admitted_combos,
    )

    t_adjust = time.monotonic()
    _text_provider = (
        get_text_boost_provider(DEFAULT_TEXT_BOOST_CONFIG)
        if input_data.get("_has_valid_experience")
        else None
    )
    _quality_verifier = None
    if _text_provider is not None:
        try:
            from src.agent.text_preprocessing_agent import TextPreprocessingAgent

            _qa = TextPreprocessingAgent()
            _quality_verifier = _qa.validate_quality_batch
        except Exception:
            pass

    pipeline = ProbabilityAdjustmentPipeline(
        probability_adjuster=probability_adjuster,
        text_boost_provider=_text_provider,
        quality_verifier=_quality_verifier,
        **_adjustment_pipeline_kwargs(adjustment_flags),
    )

    _precomputed_quality = None
    if _text_provider is not None and has_any_experience(adj_ctx.experience_details or {}):
        quality_tags = _text_provider.get_quality_tags(adj_ctx.experience_details)
        llm_verified = None
        if quality_tags and _quality_verifier is not None:
            try:
                field_cn = {
                    "research_details": "科研",
                    "award_details": "奖项",
                    "internship_details": "实习",
                    "paper_details": "论文",
                }
                verify_input = {}
                for fk, fcn in field_cn.items():
                    text = str(adj_ctx.experience_details.get(fk, "") or "").strip()
                    if text:
                        verify_input[fk] = {
                            "label": fcn,
                            "text": text,
                            "signal_hits": list(quality_tags.get(fk, [])),
                        }
                if verify_input:
                    llm_verified = _quality_verifier(verify_input)
            except Exception:
                pass
        _precomputed_quality = {
            "quality_tags": quality_tags,
            "llm_verified": llm_verified or {},
        }

    sim_results = pipeline.adjust_batch(
        sim_results,
        adj_ctx,
        progress_reporter=reporter,
        batch_tag="相似专业",
        precomputed_quality=_precomputed_quality,
    )
    cross_results = pipeline.adjust_batch(
        cross_results,
        adj_ctx,
        progress_reporter=reporter,
        batch_tag="跨专业",
        precomputed_quality=_precomputed_quality,
    )
    if user_specified_results:
        user_specified_results = pipeline.adjust_batch(
            user_specified_results,
            adj_ctx,
            progress_reporter=reporter,
            batch_tag="用户指定",
            precomputed_quality=_precomputed_quality,
        )

    prediction_handler_logger.info(
        "调整链完成 | sim=%d cross=%d user=%d elapsed=%.3fs",
        len(sim_results),
        len(cross_results),
        len(user_specified_results) if user_specified_results else 0,
        time.monotonic() - t_adjust,
    )

    unique_results, _representatives = combine_and_deduplicate_results_with_sources(
        sim_results, cross_results, user_specified_results
    )

    t_tier = time.monotonic()
    tier_map = build_tier_map(DEFAULT_UNIVERSITY_DIFFICULTY_ORDER)
    tiered_results = [r for r in unique_results if tier_map.get(str(r.get("university", "")))]
    available_tiers = {tier_map[str(r["university"])] for r in tiered_results}
    if len(available_tiers) >= 2:
        unique_results = apply_tier_rank_repair(unique_results, tier_map)
        prediction_handler_logger.info(
            "Tier 校准完成 | tiers=%s results=%d elapsed=%.3fs",
            sorted(available_tiers),
            len(unique_results),
            time.monotonic() - t_tier,
        )

        _l6_steps: dict[tuple, dict] = {}
        for r in unique_results:
            steps = r.get("_adjustment_steps")
            if steps:
                key = (str(r.get("university", "")), str(r.get("major", "")))
                _l6_steps[key] = steps[-1]

        if _l6_steps:
            for source_list in [sim_results, cross_results, user_specified_results]:
                if not source_list:
                    continue
                for r in source_list:
                    key = (str(r.get("university", "")), str(r.get("major", "")))
                    # 只回写真正进入 unified 的代表条目；同一 key 出现在多个
                    # source list 时，非代表条目的链/概率不能被 L6 覆盖。
                    if _representatives.get(key) is not r:
                        continue
                    l6_step = _l6_steps.get(key)
                    if l6_step is None:
                        continue
                    existing_steps = r.get("_adjustment_steps")
                    if existing_steps is not None:
                        existing_steps.append(dict(l6_step))
                    else:
                        existing_steps = []
                        r["_adjustment_steps"] = existing_steps
                        existing_steps.append(dict(l6_step))
                    r["probability"] = l6_step.get("after", r.get("probability", 0))
                    trace = r.get("_adjustment_trace")
                    if trace is not None:
                        trace["tier_rank_repair"] = l6_step.get("delta", 0)

    if not unique_results:
        meta = meta or {}
        meta["error"] = "empty_results"
        empty_msg = PIPELINE_MESSAGES["empty_results"]
        meta.setdefault(
            "user_message",
            random.choice(empty_msg) if isinstance(empty_msg, list) else empty_msg,
        )
        _track(
            "prediction_empty",
            candidate_count_before_filter=len(sim_results)
            + len(cross_results)
            + len(user_specified_results or []),
            sim_count=len(sim_results),
            cross_count=len(cross_results),
            usr_count=len(user_specified_results or []),
        )
        prediction_handler_logger.info(
            "Pipe 完成(空结果) | sim=%d cross=%d user=%d total_elapsed=%.3fs",
            len(sim_results),
            len(cross_results),
            len(user_specified_results) if user_specified_results else 0,
            time.monotonic() - t_start,
        )
        reporter.emit(
            format_pipeline_empty_progress(
                bg_major=background_major,
                target_universities=cleaned_input.get("target_universities"),
            ),
            force=True,
            phase=PIPELINE_PHASE_MAP["empty_results"],
        )
        return PredictionResultModel(meta=meta)

    reporter.emit(
        format_pipeline_done_progress(),
        force=True,
        phase=PIPELINE_PHASE_MAP["done"],
    )

    total_elapsed = time.monotonic() - t_start
    prediction_handler_logger.info(
        "Pipe 完成 | sim=%d cross=%d user=%d unified=%d tier_calibrated=%s total_elapsed=%.3fs",
        len(sim_results),
        len(cross_results),
        len(user_specified_results) if user_specified_results else 0,
        len(unique_results),
        len(available_tiers) >= 2,
        total_elapsed,
    )

    return PredictionResultModel(
        similarity_results=sim_results,
        cross_major_results=cross_results,
        user_specified_results=user_specified_results,
        unified_results=unique_results,
        meta=meta,
    )


def _prepare_list_args(input_data: dict[str, Any]) -> tuple[list[str], list[str]]:
    def _to_list(key: str) -> list[str]:
        raw = input_data.get(key)
        return [str(x) for x in raw] if isinstance(raw, list) else []

    return _to_list("_all_universities_target"), _to_list("_all_majors_target")


def run_prediction_pipeline(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
    adjustment_flag_overrides: dict[str, bool] | None = None,
) -> PredictionResultModel:
    all_universities_target, all_majors_target = _prepare_list_args(input_data)
    reporter = ProgressReporter(None)

    return _execute_prediction_pipeline(
        input_data=input_data,
        model_name=model_name,
        cases_df_fingerprint=cases_df_fingerprint,
        loaded_feature_names=loaded_feature_names,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        reporter=reporter,
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
        adjustment_flag_overrides=adjustment_flag_overrides,
    )


def run_prediction_pipeline_with_progress(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    *,
    progress_cb: ProgressCallback | None = None,
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
    page_state: machine_learning_model | None = None,
    cached_combinations: list[tuple[str, str]] | None = None,
    adjustment_flag_overrides: dict[str, bool] | None = None,
) -> PredictionResultModel:
    all_universities_target, all_majors_target = _prepare_list_args(input_data)
    reporter = ProgressReporter(progress_cb)

    return _execute_prediction_pipeline(
        input_data=input_data,
        model_name=model_name,
        cases_df_fingerprint=cases_df_fingerprint,
        loaded_feature_names=loaded_feature_names,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        reporter=reporter,
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
        page_state=page_state,
        cached_combinations=cached_combinations,
        adjustment_flag_overrides=adjustment_flag_overrides,
    )
