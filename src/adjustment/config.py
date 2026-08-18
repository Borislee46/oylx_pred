import json
import logging
from functools import lru_cache
from pathlib import Path

from src.utils.schools.config_loader import UNIVERSITY_DIFFICULTY_ORDER as _CFG_DIFFICULTY_ORDER

_logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_project_root() -> Path:
    start = Path(__file__).resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    return start.parents[2]


PROJECT_ROOT: Path = _get_project_root()

DEFAULT_TEXT_BOOST_CONFIG: dict = {
    # 2026-07-20: 文本概率提升已主动下线。
    # 调整链 ECE 已放大 3.9× base model（in-sample 0.1016/0.0263，Base 为 sigmoid-era [L]；
    # R-033/R-061/R-083：4.2× → 3.9× 同步），外部验证偏差 -67pp，
    # 在无 held-out 验证的前提下叠加文本提升不可靠。
    # get_quality_tags() 仍正常工作（含金量标签展示在 trace 中）。
    "enabled": False,
    "max_total_boost": 0.15,
    "sim_gate_sum_min": 0.10,
    "sim_gate_max_min": 0.08,
    "smoothing": 0.7,
    "cap_min_factor": 0.10,
    "cap_quality_gamma": 1.2,
    "high_signal": {
        "enabled": True,
        "lexicon_path": str(PROJECT_ROOT / "config" / "text_high_signal_terms.json"),
        "lexicon_weight": 1.0,
        "novelty_weight": 0.12,
        "novelty_min_chars": 12,
        "bonus_cap_per_field": 0.6,
        "max_reasons": 3,
    },
    "model_paths": {
        "tfidf_vectorizer": str(
            PROJECT_ROOT / "src" / "ml" / "pre-trained_models" / "tfidf_vectorizer.joblib"
        ),
        "tfidf_centroids": str(
            PROJECT_ROOT / "src" / "ml" / "pre-trained_models" / "tfidf_centroids.npz"
        ),
        "text_uplift_weights": str(
            PROJECT_ROOT / "src" / "ml" / "pre-trained_models" / "text_uplift_weights.json"
        ),
    },
}

GPA_MINIMUM: float = 2.0
GPA_PENALTY_SEVERE_THRESHOLD: float = 0.95
GPA_PENALTY_MAX_COEFFICIENT: float = 0.8
GPA_PENALTY_QUADRATIC_COEFFICIENT: float = 0.15

LANGUAGE_MINIMUM: float = 0.6
LANGUAGE_PENALTY_SEVERE_THRESHOLD: float = 0.95
LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER: float = 0.5
LANGUAGE_PENALTY_LEVEL_1_THRESHOLD: float = 0.85
LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER: float = 1.5
LANGUAGE_PENALTY_LEVEL_2_THRESHOLD: float = 0.7
LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER: float = 1.0
LANGUAGE_PENALTY_LEVEL_3_THRESHOLD: float = 0.4
COMPREHENSIVE_SCORE_WEIGHTS: tuple[float, float, float] = (0.4, 0.3, 0.3)
PROBABILITY_MIN_VALUE: float = 0.001
PROBABILITY_ADJUSTMENT_THRESHOLD: float = 0.01
PROBABILITY_EXTREME_STD_MULTIPLIER: float = 2.0
# ── 以上三个概率参数（PROBABILITY_*）当前无生产消费，仅保留于
#    pipeline_config.json 的 "probability" 段（由 generate_pipeline_config.py 生成）。
#    修改它们不会影响调整链输出；如需启用请先接入 arbitration/adjustment 逻辑。
CROSS_MAJOR_PENALTY_FACTOR: float = 0.5
FACULTY_PENALTY_LIGHT: float = 0.70
FACULTY_PENALTY_MEDIUM: float = 0.50
FACULTY_PENALTY_HEAVY: float = 0.30
COMPREHENSIVE_SCORE_BOOST_THRESHOLD: float = 0.6
SELECTION_SCORE_BOOST_FACTOR: float = 0.3
PREDICTION_RULES_PATH: Path = PROJECT_ROOT / "config" / "prediction_rules.json"
UNIVERSITY_DIFFICULTY_CONFIG_PATH: Path = PREDICTION_RULES_PATH


def load_prediction_rules() -> dict:
    if PREDICTION_RULES_PATH.exists():
        try:
            with open(PREDICTION_RULES_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            _logger.warning(
                "Failed to load prediction rules from %s, using defaults: %s",
                PREDICTION_RULES_PATH,
                e,
                exc_info=True,
            )
    return {}


_rules = load_prediction_rules()

PROFESSIONAL_MAJORS: list[str] = _rules.get(
    "PROFESSIONAL_MAJORS", ["Business Administration", "MBA"]
)
PROFESSIONAL_MAJORS_LOWER: list[str] = [m.lower() for m in PROFESSIONAL_MAJORS]
PROFESSIONAL_REDUCTION_FACTOR: float = 0.30
PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR: float = 0.50
MIN_SIMILARITY_THRESHOLD: float = 0.87  # V9: 0.88-0.89 admit rate inversion
HIGHER_SIMILARITY_THRESHOLD: float = 0.92
FUZZY_BIAS_THRESHOLD_HIGH: float = 92.0
FUZZY_BIAS_THRESHOLD_MID: float = 82.0
FUZZY_BIAS_THRESHOLD_LOW: float = 72.0
FUZZY_BIAS_MULTIPLIER_HIGH: float = 1.25
FUZZY_BIAS_MULTIPLIER_MID: float = 1.15
FUZZY_BIAS_MULTIPLIER_LOW: float = 1.05
UNIVERSITY_COUNT_THRESHOLD: int = 2
CROSS_MAJOR_SIMILARITY_MIN: float = 0.8
CROSS_MAJOR_SIGMOID_STEEPNESS: float = 25.0
CROSS_MAJOR_SIGMOID_MIDPOINT: float = 0.87
CROSS_MAJOR_EVIDENCE_PRIOR_STRENGTH: float = 5.0
CROSS_MAJOR_EVIDENCE_RATIO_THRESHOLD: float = 0.85
CROSS_MAJOR_EVIDENCE_MIN_CASES: int = 5
COMBINATION_POOL_SEMANTIC_MIN: float = 0.6
COMBINATION_POOL_FUZZY_MIN: int = 90
TOP_N_RECOMMENDATIONS: int = 30
USER_SPECIFIED_SMALL_RANGE_THRESHOLD: int = 20
USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD: int = 100
USER_SPECIFIED_MEDIUM_RANGE_TOP_N: int = 50
USER_SPECIFIED_LARGE_RANGE_TOP_N: int = 100
PROBABILITY_ADJUSTER_CACHE_SIZE: int = 50
SIMILARITY_ADJUSTMENT_RULES_PATH: Path = (
    PROJECT_ROOT / "config" / "similarity_adjustment_rules.json"
)
DEFAULT_UNIVERSITY_DIFFICULTY_ORDER: tuple[str, ...] = _CFG_DIFFICULTY_ORDER
AGENT_MIN_SAFE_RELAX_THRESHOLD: float = 0.87
AGENT_BOUNDARY_SIMILARITY_RANGE: float = 0.03
AGENT_MAX_BOUNDARY_CASES: int = 20
AGENT_TAIL_PERCENTAGE: float = 0.2
AGENT_MIN_BALANCE_DIFF_MIN: int = 3
AGENT_MIN_BALANCE_DIFF_RATIO: float = 0.15
AGENT_NO_CHANGE_THRESHOLD: int = 3
ENABLE_AGENT_BALANCE: bool = False
LANGUAGE_REQUIREMENT_PENALTY_STEEPNESS: float = 7.0
LANGUAGE_REQUIREMENT_PENALTY_MIDPOINT: float = 0.5
LANGUAGE_PENALTY_SIGMOID_STEEPNESS: float = 12.0
MAX_TOTAL_PENALTY_RATIO: float = 0.7
PENALTY_CEILING_BY_LAYERS: dict[int, float] = {
    1: 0.70,
    2: 0.55,
    3: 0.45,
}
# ── PENALTY_CEILING_BY_LAYERS：仅被 src/ml/experimental 实验管线消费，
#    生产仲裁器（adjustment/arbitrator.py）使用平坦的 MAX_TOTAL_PENALTY_RATIO。
#    调整此值不会影响线上调整链。
MAX_TOTAL_BOOST_RATIO: float = 0.3
PENALTY_DECAY_FACTOR: float = 0.85
BOOST_DECAY_FACTOR: float = 0.8

BAYESIAN_SHRINKAGE_PRIOR_STRENGTH: int = 30
BAYESIAN_SHRINKAGE_GLOBAL_PRIOR: float = 0.337
ARBITRATION_MIN_PROBABILITY: float = 0.005
SCHOOL_STATS_MIN_N: int = 5
TIER_BOUNDARIES: tuple[int, int, int] = (5, 12, 19)
# 2026-07-20: 从 DYKSTRA_MAX_ITER / TIER_ISOTONIC_MIN_VIOLATION 重命名。
# 实现是相邻 extrema 拉平的启发式修复，非真正的 isotonic regression / Dykstra 投影。
TIER_RANK_REPAIR_MAX_ITER: int = 20  # 与 pipeline_config.json tier_calibration.max_iter 保持一致
TIER_RANK_REPAIR_MIN_VIOLATION: float = 0.08
FALLBACK_N_THRESHOLD: int = 5
BETA_BINOMIAL_PRIOR_STRENGTH: float = 5.0
WILSON_Z: float = 1.96


def _load_pipeline_config() -> dict:
    json_path = PROJECT_ROOT / "config" / "pipeline_config.json"
    try:
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        _logger.warning(
            "Failed to load pipeline_config.json, using hardcoded defaults", exc_info=True
        )
    return {}


def _apply_json_overrides(cfg: dict) -> None:
    if not cfg:
        return

    g = globals()
    overrides = {
        "GPA_MINIMUM": ("gpa_penalty", "minimum"),
        "GPA_PENALTY_SEVERE_THRESHOLD": ("gpa_penalty", "severe_threshold"),
        "GPA_PENALTY_MAX_COEFFICIENT": ("gpa_penalty", "max_coefficient"),
        "GPA_PENALTY_QUADRATIC_COEFFICIENT": ("gpa_penalty", "quadratic_coefficient"),
        "LANGUAGE_MINIMUM": ("language_penalty", "minimum"),
        "LANGUAGE_PENALTY_SEVERE_THRESHOLD": ("language_penalty", "severe_threshold"),
        "LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER": ("language_penalty", "pass_line_multiplier"),
        "LANGUAGE_PENALTY_LEVEL_1_THRESHOLD": ("language_penalty", "levels", "L1", "threshold"),
        "LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER": ("language_penalty", "levels", "L1", "multiplier"),
        "LANGUAGE_PENALTY_LEVEL_2_THRESHOLD": ("language_penalty", "levels", "L2", "threshold"),
        "LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER": ("language_penalty", "levels", "L2", "multiplier"),
        "LANGUAGE_PENALTY_LEVEL_3_THRESHOLD": ("language_penalty", "levels", "L3", "threshold"),
        "LANGUAGE_PENALTY_SIGMOID_STEEPNESS": ("language_penalty", "sigmoid_steepness"),
        "LANGUAGE_REQUIREMENT_PENALTY_STEEPNESS": (
            "language_penalty",
            "requirement_penalty_steepness",
        ),
        "LANGUAGE_REQUIREMENT_PENALTY_MIDPOINT": (
            "language_penalty",
            "requirement_penalty_midpoint",
        ),
        "CROSS_MAJOR_PENALTY_FACTOR": ("cross_major_penalty", "factor"),
        "CROSS_MAJOR_SIMILARITY_MIN": ("cross_major_penalty", "similarity_min"),
        "MIN_SIMILARITY_THRESHOLD": ("cross_major_penalty", "min_similarity_threshold"),
        "HIGHER_SIMILARITY_THRESHOLD": ("cross_major_penalty", "higher_similarity_threshold"),
        "CROSS_MAJOR_SIGMOID_STEEPNESS": ("cross_major_penalty", "sigmoid_steepness"),
        "CROSS_MAJOR_SIGMOID_MIDPOINT": ("cross_major_penalty", "sigmoid_midpoint"),
        "CROSS_MAJOR_EVIDENCE_PRIOR_STRENGTH": ("cross_major_penalty", "evidence_prior_strength"),
        "CROSS_MAJOR_EVIDENCE_RATIO_THRESHOLD": ("cross_major_penalty", "evidence_ratio_threshold"),
        "CROSS_MAJOR_EVIDENCE_MIN_CASES": ("cross_major_penalty", "evidence_min_cases"),
        "FACULTY_PENALTY_LIGHT": ("faculty_penalty", "light"),
        "FACULTY_PENALTY_MEDIUM": ("faculty_penalty", "medium"),
        "FACULTY_PENALTY_HEAVY": ("faculty_penalty", "heavy"),
        "PROFESSIONAL_REDUCTION_FACTOR": ("professional_penalty", "reduction_factor"),
        "PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR": (
            "professional_penalty",
            "user_specified_reduction_factor",
        ),
        "MAX_TOTAL_PENALTY_RATIO": ("arbitration", "max_total_penalty_ratio"),
        "MAX_TOTAL_BOOST_RATIO": ("arbitration", "max_total_boost_ratio"),
        "PENALTY_DECAY_FACTOR": ("arbitration", "penalty_decay_factor"),
        "BOOST_DECAY_FACTOR": ("arbitration", "boost_decay_factor"),
        "ARBITRATION_MIN_PROBABILITY": ("arbitration", "min_probability"),
        "PENALTY_CEILING_BY_LAYERS": ("arbitration", "penalty_ceiling_by_layers"),
        "PROBABILITY_MIN_VALUE": ("probability", "min_value"),
        "PROBABILITY_ADJUSTMENT_THRESHOLD": ("probability", "adjustment_threshold"),
        "PROBABILITY_EXTREME_STD_MULTIPLIER": ("probability", "extreme_std_multiplier"),
        "FUZZY_BIAS_THRESHOLD_HIGH": ("fuzzy_bias", "threshold_high"),
        "FUZZY_BIAS_THRESHOLD_MID": ("fuzzy_bias", "threshold_mid"),
        "FUZZY_BIAS_THRESHOLD_LOW": ("fuzzy_bias", "threshold_low"),
        "FUZZY_BIAS_MULTIPLIER_HIGH": ("fuzzy_bias", "multiplier_high"),
        "FUZZY_BIAS_MULTIPLIER_MID": ("fuzzy_bias", "multiplier_mid"),
        "FUZZY_BIAS_MULTIPLIER_LOW": ("fuzzy_bias", "multiplier_low"),
        "COMPREHENSIVE_SCORE_BOOST_THRESHOLD": ("comprehensive_score", "boost_threshold"),
        "SELECTION_SCORE_BOOST_FACTOR": ("comprehensive_score", "selection_score_boost_factor"),
        "COMBINATION_POOL_SEMANTIC_MIN": ("combination_pool", "semantic_min"),
        "COMBINATION_POOL_FUZZY_MIN": ("combination_pool", "fuzzy_min"),
        "TOP_N_RECOMMENDATIONS": ("combination_pool", "top_n_recommendations"),
        "UNIVERSITY_COUNT_THRESHOLD": ("combination_pool", "university_count_threshold"),
        "AGENT_MIN_SAFE_RELAX_THRESHOLD": ("agent", "min_safe_relax_threshold"),
        "AGENT_BOUNDARY_SIMILARITY_RANGE": ("agent", "boundary_similarity_range"),
        "AGENT_MAX_BOUNDARY_CASES": ("agent", "max_boundary_cases"),
        "AGENT_TAIL_PERCENTAGE": ("agent", "tail_percentage"),
        "AGENT_MIN_BALANCE_DIFF_MIN": ("agent", "min_balance_diff_min"),
        "AGENT_MIN_BALANCE_DIFF_RATIO": ("agent", "min_balance_diff_ratio"),
        "AGENT_NO_CHANGE_THRESHOLD": ("agent", "no_change_threshold"),
        "BAYESIAN_SHRINKAGE_PRIOR_STRENGTH": ("bayesian_shrinkage", "prior_strength"),
        "BAYESIAN_SHRINKAGE_GLOBAL_PRIOR": ("bayesian_shrinkage", "global_prior"),
        "COMPREHENSIVE_SCORE_WEIGHTS": ("scoring", "comprehensive_score_weights"),
        "SCHOOL_STATS_MIN_N": ("school_stats", "min_n"),
        "TIER_BOUNDARIES": ("tier_calibration", "boundaries"),
        "TIER_RANK_REPAIR_MAX_ITER": ("tier_calibration", "max_iter"),
        "FALLBACK_N_THRESHOLD": ("fallback", "n_threshold"),
        "WILSON_Z": ("fallback", "wilson_z"),
        "DEFAULT_TEXT_BOOST_CONFIG": ("text_boost", None),
    }

    def _get(cfg_dict, *keys):
        for k in keys:
            if isinstance(cfg_dict, dict):
                cfg_dict = cfg_dict.get(k)
            else:
                return None
        return cfg_dict

    applied_count = 0
    for var_name, json_path in overrides.items():
        if var_name == "DEFAULT_TEXT_BOOST_CONFIG":
            tb = _get(cfg, "text_boost")
            if tb and isinstance(tb, dict):
                boost_cfg = g.get("DEFAULT_TEXT_BOOST_CONFIG", {})
                if isinstance(boost_cfg, dict):
                    _TB_KEY_MAP: dict[str, str] = {}
                    _TB_BARE = [
                        "max_total_boost",
                        "sim_gate_sum_min",
                        "sim_gate_max_min",
                        "smoothing",
                        "cap_min_factor",
                        "cap_quality_gamma",
                    ]
                    for _k in _TB_BARE:
                        _TB_KEY_MAP[_k] = _k
                        _TB_KEY_MAP[f"logit_uplift_{_k}"] = _k
                    for json_k, py_k in _TB_KEY_MAP.items():
                        if json_k in tb:
                            boost_cfg[py_k] = tb[json_k]
                            applied_count += 1
            continue

        val = _get(cfg, *json_path)
        if val is not None:
            if var_name == "PENALTY_CEILING_BY_LAYERS" and isinstance(val, dict):
                val = {int(k): v for k, v in val.items()}
            if var_name in (
                "COMPREHENSIVE_SCORE_WEIGHTS",
                "TIER_BOUNDARIES",
            ) and isinstance(val, list):
                val = tuple(val)
            g[var_name] = val
            applied_count += 1
    _logger.info("Applied pipeline_config.json overrides: %d parameter(s)", applied_count)


ADJUSTMENT_FLAGS_DEFAULTS: dict[str, bool] = {
    "enable_gpa_penalty": True,
    "enable_language_penalty": True,
    "enable_cross_major_penalty": True,
    "enable_cross_faculty_penalty": True,
    "enable_professional_penalty": True,
    # 单校语言要求惩罚（LanguageRequirementPenalty，链外乘数）。
    # 此前无开关，消融实验无法覆盖该路径；默认开启以保持行为不变。
    "enable_language_requirement_penalty": True,
    # 文本提升模块总开关（含含金量标签路径）。此前缺失导致 pipeline_config.json
    # 与 predict_api 消融实验中的 enable_text_boost 键被 load_adjustment_flags 过滤，
    # 开关永远无法生效。当前概率提升在 LogitUpliftProvider.apply 内部已下线，
    # 此开关控制的是整个文本模块（标签 + 未来恢复的 apply）。
    "enable_text_boost": True,
}


def load_adjustment_flags() -> dict[str, bool]:
    """Load adjustment flags from pipeline_config.json, falling back to defaults."""
    flags = dict(ADJUSTMENT_FLAGS_DEFAULTS)
    cfg = _load_pipeline_config()
    stored = cfg.get("adjustment_flags", {})
    if isinstance(stored, dict):
        flags.update({k: v for k, v in stored.items() if k in flags})
    return flags


_apply_json_overrides(_load_pipeline_config())
