from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class _DictCompat(BaseModel):
    @staticmethod
    def _resolve_alias(cls: type, key: str) -> str | None:
        for field_name, field_info in cls.model_fields.items():
            va = field_info.validation_alias
            if va is not None:
                if isinstance(va, str) and va == key:
                    return field_name
            sa = field_info.serialization_alias
            if sa is not None:
                if isinstance(sa, str) and sa == key:
                    return field_name
            al = field_info.alias
            if al is not None:
                if isinstance(al, str) and al == key:
                    return field_name
        return None

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        resolved = self._resolve_alias(type(self), key)
        if resolved is not None:
            return getattr(self, resolved)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        if hasattr(self, key):
            return True
        return self._resolve_alias(type(self), key) is not None


class HarnessStages(_DictCompat):
    lead_in: str = Field(default="tools", validation_alias="lead_in")
    prediction: str = Field(default="ui", validation_alias="prediction")
    explain: str = Field(default="ui", validation_alias="explain")

    model_config = {"extra": "allow"}


class AppConfig(_DictCompat):
    model_config = {"extra": "allow"}

    e2_x3id_config: str = Field(default="", validation_alias="E2_X3ID_CONFIG")
    e2_x3key_config: str = Field(default="", validation_alias="E2_X3KEY_CONFIG")
    e2_verify_code_api_url: str = Field(default="", validation_alias="E2_VERIFY_CODE_API_URL")
    streamlit_app_base_url: str = Field(default="", validation_alias="STREAMLIT_APP_BASE_URL")
    e2_login_url_template: str = Field(default="", validation_alias="E2_LOGIN_URL_TEMPLATE")
    callback_page_path: str = Field(default="", validation_alias="CALLBACK_PAGE_PATH")
    e2_callback_base_url: str = Field(default="", validation_alias="E2_CALLBACK_BASE_URL")

    open_ai_api_key: str = Field(default="", validation_alias="OPEN_AI_API_KEY")
    open_ai_base_url: str = Field(default="", validation_alias="OPEN_AI_BASE_URL")
    open_ai_model: str = Field(default="deepseek-v4-flash", validation_alias="OPEN_AI_MODEL")

    open_ai_base_url_fallback: str = Field(default="", validation_alias="OPEN_AI_BASE_URL_FALLBACK")
    open_ai_api_key_fallback: str = Field(default="", validation_alias="OPEN_AI_API_KEY_FALLBACK")
    open_ai_alt_model: str = Field(default="", validation_alias="OPEN_AI_ALT_MODEL")

    harness_enabled: bool = Field(default=False, validation_alias="HARNESS_ENABLED")
    harness_stages: HarnessStages = Field(
        default_factory=HarnessStages, validation_alias="HARNESS_STAGES"
    )
    harness_max_tool_rounds: int = Field(default=12, validation_alias="HARNESS_MAX_TOOL_ROUNDS")
    harness_turn_budget_ms: int = Field(default=60_000, validation_alias="HARNESS_TURN_BUDGET_MS")
    harness_observability: str = Field(default="off", validation_alias="HARNESS_OBSERVABILITY")
    harness_tool_cache: bool = Field(default=True, validation_alias="HARNESS_TOOL_CACHE")
    harness_history_max_messages: int = Field(
        default=40, validation_alias="HARNESS_HISTORY_MAX_MESSAGES"
    )
    harness_guard_enabled: bool = Field(default=False, validation_alias="HARNESS_GUARD_ENABLED")
    harness_max_concurrency: int = Field(default=1, validation_alias="HARNESS_MAX_CONCURRENCY")

    enable_lead_in_agent: bool = Field(default=False, validation_alias="ENABLE_LEAD_IN_AGENT")
    enable_lead_in_hub: bool = Field(default=False, validation_alias="ENABLE_LEAD_IN_HUB")
    lead_in_agent_mode: str = Field(default="tools", validation_alias="LEAD_IN_AGENT_MODE")
    lead_in_tool_code_mode: bool = Field(default=False, validation_alias="LEAD_IN_TOOL_CODE_MODE")

    ghost_api_key: str = Field(default="", validation_alias="GHOST_API_KEY")
    ghost_api_base_url: str = Field(default="", validation_alias="GHOST_API_BASE_URL")
    ghost_api_model: str = Field(default="", validation_alias="GHOST_API_MODEL")

    intent_gate_enabled: bool | None = Field(default=None, validation_alias="INTENT_GATE_ENABLED")
    open_ai_model_fallback: str | None = Field(
        default=None, validation_alias="OPEN_AI_MODEL_FALLBACK"
    )

    @model_validator(mode="after")
    def _migrate_legacy_keys(self) -> AppConfig:
        extra = self.model_extra or {}
        if not self.harness_enabled and "ENABLE_LEAD_IN_HUB" in extra:
            if self.enable_lead_in_hub:
                self.harness_enabled = True
        return self


class HarnessConfig(_DictCompat):
    enabled: bool = False
    stages: HarnessStages = Field(default_factory=HarnessStages)
    max_tool_rounds: int = 12
    turn_budget_ms: int = 60_000
    observability: str = "off"
    tool_cache: bool = True
    history_max_messages: int = 40
    guard_enabled: bool = False
    max_concurrency: int = 1
    lead_in_tool_code_mode: bool = False

    @classmethod
    def from_app_config(cls, cfg: AppConfig) -> HarnessConfig:
        return cls(
            enabled=cfg.harness_enabled,
            stages=cfg.harness_stages,
            max_tool_rounds=cfg.harness_max_tool_rounds,
            turn_budget_ms=cfg.harness_turn_budget_ms,
            observability=cfg.harness_observability,
            tool_cache=cfg.harness_tool_cache,
            history_max_messages=cfg.harness_history_max_messages,
            guard_enabled=cfg.harness_guard_enabled,
            max_concurrency=cfg.harness_max_concurrency,
            lead_in_tool_code_mode=cfg.lead_in_tool_code_mode,
        )


class IntentGateConfig(_DictCompat):
    enabled: bool = True
    mode: str = "llm_tool"
    templates: dict[str, str] = Field(default_factory=dict)
    planned_degree_hints: dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class SchoolConstants(_DictCompat):
    school_level_priority: dict[str, int] = Field(default_factory=dict)
    school_level_scores: dict[str, float] = Field(default_factory=dict)
    overseas_school_levels: list[str] = Field(default_factory=list)
    language_boost_multipliers: dict[str, float] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @classmethod
    def load(cls, path: str | None = None) -> SchoolConstants:
        import json
        from pathlib import Path

        if path is None:
            json_path = Path(__file__).resolve().parents[2] / "config" / "school_constants.json"
        else:
            json_path = Path(path)

        try:
            if json_path.exists():
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                return cls.model_validate(raw)
        except Exception:
            pass

        return cls()

    def priority_for(self, level: str | None) -> int:
        key = level if level is not None else "null"
        return self.school_level_priority.get(key, 11)

    def score_for(self, level: str | None) -> float:
        key = level if level is not None else "null"
        return self.school_level_scores.get(key, 0.50)


class ContractTier(_DictCompat):
    name: str = ""
    label: str = ""
    price_cny: int = 0
    max_schools: int = 0
    max_majors: int = 0
    team: str = ""
    regions: list[str] = Field(default_factory=list)
    note: str = ""
    _enabled: bool = True

    model_config = {"extra": "allow"}


class ContractConfig(_DictCompat):
    default_tier: str = "joint_hkmo_sg"
    email_to_tier: dict[str, str] = Field(default_factory=dict)
    applications: dict[str, dict] = Field(default_factory=dict)
    upgrades: dict[str, dict] = Field(default_factory=dict)
    prompt_context_template: dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @classmethod
    def load(cls, path: str | None = None) -> ContractConfig:
        import json
        from pathlib import Path

        if path is None:
            json_path = Path(__file__).resolve().parents[2] / "config" / "contract_config.json"
        else:
            json_path = Path(path)

        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            return cls.model_validate(raw)
        except Exception:
            return cls()


class TierDifficultyWeights(_DictCompat):
    safety_base: float = 0.50
    safety_coeff: float = 0.15
    target_base: float = 0.25
    target_coeff: float = 0.10

    model_config = {"extra": "allow"}


class TierConfig(_DictCompat):
    safety_threshold: float = 0.50
    match_threshold: float = 0.20
    difficulty_weights: TierDifficultyWeights = Field(default_factory=TierDifficultyWeights)
    labels: dict[str, str] = Field(default_factory=dict)
    colors: dict[str, str] = Field(default_factory=dict)
    label_normalise: dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @classmethod
    def load(cls, path: str | None = None) -> TierConfig:
        import json
        from pathlib import Path

        if path is None:
            json_path = Path(__file__).resolve().parents[2] / "config" / "tier_config.json"
        else:
            json_path = Path(path)

        try:
            if json_path.exists():
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                return cls.model_validate(raw)
        except Exception:
            pass
        return cls()


class _AdjustmentFlags(_DictCompat):
    enable_gpa_penalty: bool = True
    enable_language_penalty: bool = True
    enable_cross_major_penalty: bool = False
    enable_cross_faculty_penalty: bool = True
    enable_professional_penalty: bool = True
    model_config = {"extra": "allow"}


class PipelineConfig(_DictCompat):
    adjustment_flags: _AdjustmentFlags = Field(default_factory=_AdjustmentFlags)
    gpa_penalty: dict = Field(default_factory=dict)
    language_penalty: dict = Field(default_factory=dict)
    cross_major_penalty: dict = Field(default_factory=dict)
    faculty_penalty: dict = Field(default_factory=dict)
    professional_penalty: dict = Field(default_factory=dict)
    arbitration: dict = Field(default_factory=dict)
    probability: dict = Field(default_factory=dict)
    text_boost: dict = Field(default_factory=dict)
    fuzzy_bias: dict = Field(default_factory=dict)
    comprehensive_score: dict = Field(default_factory=dict)
    combination_pool: dict = Field(default_factory=dict)
    manual_adjust: dict = Field(default_factory=dict)
    user_specified: dict = Field(default_factory=dict)
    agent: dict = Field(default_factory=dict)
    bayesian_shrinkage: dict = Field(default_factory=dict)
    scoring: dict = Field(default_factory=dict)
    school_stats: dict = Field(default_factory=dict)
    tier_calibration: dict = Field(default_factory=dict)
    fallback: dict = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @classmethod
    def load(cls, path: str | None = None) -> PipelineConfig:
        import json
        from pathlib import Path

        if path is None:
            json_path = Path(__file__).resolve().parents[2] / "config" / "pipeline_config.json"
        else:
            json_path = Path(path)
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            return cls.model_validate(raw)
        except Exception:
            return cls()


class _ProfileTier(_DictCompat):
    min_avg_prob: float = 0.0
    max_penalty_count: int = 0
    model_config = {"extra": "allow"}


class ProfileClassificationConfig(_DictCompat):
    cross_major_proportion_cutoff: float = 0.4
    strong_elite: _ProfileTier = Field(default_factory=_ProfileTier)
    medium_mixed: _ProfileTier = Field(default_factory=_ProfileTier)

    model_config = {"extra": "allow"}

    @classmethod
    def load(cls, path: str | None = None) -> ProfileClassificationConfig:
        import json
        from pathlib import Path

        if path is None:
            json_path = (
                Path(__file__).resolve().parents[2] / "config" / "profile_classification.json"
            )
        else:
            json_path = Path(path)
        try:
            if json_path.exists():
                return cls.model_validate(json.loads(json_path.read_text(encoding="utf-8")))
        except Exception:
            pass
        return cls()


class DevConfig(_DictCompat):
    debug_mode: bool = Field(default=False, validation_alias="DEBUG_MODE")
    debug_user: dict = Field(default_factory=dict, validation_alias="DEBUG_USER")

    model_config = {"extra": "allow"}

    @classmethod
    def load(cls, path: str | None = None) -> DevConfig:
        import json
        from pathlib import Path

        if path is None:
            json_path = Path(__file__).resolve().parents[2] / "config" / "dev_config.json"
        else:
            json_path = Path(path)
        try:
            if json_path.exists():
                return cls.model_validate(json.loads(json_path.read_text(encoding="utf-8")))
        except Exception:
            pass
        return cls()


class SalesBlocksPolicy(_DictCompat):
    positioning: str = "application_assist"
    max_default_upgrades: int = 2
    always_preselect_application: bool = True
    default_upgrade_priority: list[str] = Field(default_factory=list)
    exclude_from_default: list[str] = Field(default_factory=list)
    slot_labels: dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @classmethod
    def load(cls, path: str | None = None) -> SalesBlocksPolicy:
        import json
        from pathlib import Path

        if path is None:
            json_path = Path(__file__).resolve().parents[2] / "config" / "sales_blocks_policy.json"
        else:
            json_path = Path(path)
        try:
            if json_path.exists():
                return cls.model_validate(json.loads(json_path.read_text(encoding="utf-8")))
        except Exception:
            pass
        return cls()
