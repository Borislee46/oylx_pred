from __future__ import annotations

from typing import Any

from src.pages.prediction.result_modifier.config import (
    LOGIT_UPLIFT_DEFAULT_CAP_MIN_FACTOR,
    LOGIT_UPLIFT_DEFAULT_CAP_QUALITY_GAMMA,
    LOGIT_UPLIFT_DEFAULT_SIM_GATE_MAX_MIN,
    LOGIT_UPLIFT_DEFAULT_SIM_GATE_SUM_MIN,
    LOGIT_UPLIFT_DEFAULT_SMOOTHING,
)
from src.pages.prediction.result_modifier.providers.logit_uplift import (
    DeltaCalculator,
    ProbabilityApplier,
    SimilarityComputer,
    TextProcessor,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.model_loader import (
    get_model_loader,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.signal_scorer import (
    SignalScorer,
)
from src.pages.prediction.result_modifier.text_boost_provider import TextBoostProvider
from src.pages.prediction.result_modifier.utils import has_any_experience
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class LogitUpliftProvider(TextBoostProvider):
    def __init__(
        self,
        vectorizer_path: str,
        centroids_path: str,
        weights_path: str,
        max_total_boost: float = 0.05,
        sim_gate_sum_min: float | None = None,
        sim_gate_max_min: float | None = None,
        smoothing: float | None = None,
        cap_min_factor: float | None = None,
        cap_quality_gamma: float | None = None,
        high_signal: dict[str, Any] | None = None,
    ) -> None:
        self._max_total_boost = float(max_total_boost)
        self._sim_gate_sum_min = (
            LOGIT_UPLIFT_DEFAULT_SIM_GATE_SUM_MIN
            if sim_gate_sum_min is None
            else float(sim_gate_sum_min)
        )
        self._sim_gate_max_min = (
            LOGIT_UPLIFT_DEFAULT_SIM_GATE_MAX_MIN
            if sim_gate_max_min is None
            else float(sim_gate_max_min)
        )
        self._smoothing = LOGIT_UPLIFT_DEFAULT_SMOOTHING if smoothing is None else float(smoothing)
        self._cap_min_factor = (
            LOGIT_UPLIFT_DEFAULT_CAP_MIN_FACTOR if cap_min_factor is None else float(cap_min_factor)
        )
        self._cap_quality_gamma = (
            LOGIT_UPLIFT_DEFAULT_CAP_QUALITY_GAMMA
            if cap_quality_gamma is None
            else float(cap_quality_gamma)
        )

        text_keys = (
            "research_details",
            "award_details",
            "internship_details",
            "paper_details",
        )
        count_keys = ("research_count", "award_count", "internship_count", "paper_count")

        hs = high_signal or {}
        hs_enabled = bool(hs.get("enabled", False))
        signal_scorer: SignalScorer | None = None
        novelty_weight = 0.0
        novelty_min_chars = 12
        if hs_enabled:
            enabled_fields_raw = hs.get("enabled_fields")
            enabled_fields = (
                tuple(enabled_fields_raw)
                if isinstance(enabled_fields_raw, list)
                and all(isinstance(x, str) and x.strip() for x in enabled_fields_raw)
                else None
            )
            lexicon_path = hs.get("lexicon_path")
            signal_scorer = SignalScorer(
                lexicon_path=lexicon_path if isinstance(lexicon_path, str) else None,
                enabled_fields=enabled_fields,
                per_field_cap=float(hs.get("bonus_cap_per_field", 0.6)),
                lexicon_weight=float(hs.get("lexicon_weight", 1.0)),
            )
            novelty_weight = float(hs.get("novelty_weight", 0.12))
            novelty_min_chars = int(hs.get("novelty_min_chars", 12))

        self._text_processor = TextProcessor(text_keys=text_keys, count_keys=count_keys)
        self._model_loader = get_model_loader(
            vectorizer_path=vectorizer_path,
            centroids_path=centroids_path,
            weights_path=weights_path,
        )
        self._similarity_computer = SimilarityComputer(
            model_loader=self._model_loader,
            text_processor=self._text_processor,
            signal_scorer=signal_scorer,
            novelty_weight=novelty_weight,
            novelty_min_chars=novelty_min_chars,
        )
        self._delta_calculator = DeltaCalculator(
            model_loader=self._model_loader,
            similarity_computer=self._similarity_computer,
            text_processor=self._text_processor,
            sim_gate_sum_min=self._sim_gate_sum_min,
            sim_gate_max_min=self._sim_gate_max_min,
        )
        self._probability_applier = ProbabilityApplier(
            text_processor=self._text_processor,
            max_total_boost=self._max_total_boost,
            smoothing=self._smoothing,
            cap_min_factor=self._cap_min_factor,
            cap_quality_gamma=self._cap_quality_gamma,
        )

    def apply(self, probabilities: list[float], experience_details: dict[str, Any]) -> list[float]:
        if not probabilities:
            return probabilities
        if not has_any_experience(experience_details):
            return probabilities

        sig = self._text_processor.make_signature(experience_details)
        try:
            delta_logit, sims, remarks = self._delta_calculator.cached_delta_logit(sig)
        except (
            FileNotFoundError,
            OSError,
            EOFError,
            ValueError,
            TypeError,
            RuntimeError,
            ImportError,
            AttributeError,
        ) as e:
            logger.error(
                f"[背提文本加成算法] LogitUpliftProvider 计算 delta_logit 失败: {str(e)}",
                exc_info=True,
            )
            return probabilities

        if delta_logit <= 0:
            return probabilities

        updated = self._probability_applier.apply_probability_boost(
            probabilities=probabilities,
            delta_logit=delta_logit,
            sims=sims,
        )

        return updated
