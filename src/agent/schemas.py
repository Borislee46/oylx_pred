import logging
import math
from typing import Any

from pydantic import BaseModel, Field

from src.utils.config_models import TierConfig

_logger = logging.getLogger(__name__)

_tier_cfg = TierConfig.load()

TIER_THRESHOLD_SAFETY: float = _tier_cfg.safety_threshold
TIER_THRESHOLD_MATCH: float = _tier_cfg.match_threshold

_DW = _tier_cfg.difficulty_weights
_TIER_DIFF_SAFETY_BASE: float = _DW.safety_base
_TIER_DIFF_SAFETY_COEFF: float = _DW.safety_coeff
_TIER_DIFF_TARGET_BASE: float = _DW.target_base
_TIER_DIFF_TARGET_COEFF: float = _DW.target_coeff


def _jenks_breaks_1d(values: list[float], k: int = 3) -> list[int]:
    n = len(values)
    if n <= k:
        return list(range(1, n))

    pref = [0.0] * (n + 1)
    pref2 = [0.0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + values[i]
        pref2[i + 1] = pref2[i] + values[i] ** 2

    def _ssd(lo: int, hi: int) -> float:
        cnt = hi - lo
        if cnt <= 0:
            return 0.0
        s = pref[hi] - pref[lo]
        s2 = pref2[hi] - pref2[lo]
        return s2 - s * s / cnt

    best_ssd = float("inf")
    best = None
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            ssd = _ssd(0, i) + _ssd(i, j) + _ssd(j, n)
            if ssd < best_ssd:
                best_ssd = ssd
                best = (i, j)
    return list(best) if best else [n // 3, 2 * n // 3]


def compute_tiers(probs: list[float]) -> list[str]:
    n = len(probs)
    _logger.debug("compute_tiers | n=%d", n)

    def _by_threshold(p: float) -> str:
        if p >= TIER_THRESHOLD_SAFETY:
            return "保底"
        if p >= TIER_THRESHOLD_MATCH:
            return "适中"
        return "冲刺"

    if n == 0:
        return []
    if n < 3:
        return [_by_threshold(p) for p in probs]

    if any(not isinstance(p, (int, float)) or not math.isfinite(p) for p in probs):
        _logger.warning(
            "compute_tiers | 存在非有限概率值，回退阈值标签: %s",
            [str(p) for p in probs],
        )
        return [
            _by_threshold(p) if isinstance(p, (int, float)) and math.isfinite(p) else "适中"
            for p in probs
        ]

    uniq = sorted(set(probs))
    if len(uniq) <= 2:
        mapping = {p: _by_threshold(p) for p in uniq}
        return [mapping[p] for p in probs]

    breaks = _jenks_breaks_1d(uniq, k=min(3, len(uniq)))
    i, j = breaks[0], breaks[1]

    label: dict[float, str] = {}
    for rank, p in enumerate(uniq):
        if rank < i:
            label[p] = "冲刺"
        elif rank < j:
            label[p] = "冲刺" if p < TIER_THRESHOLD_MATCH else "适中"
        else:
            label[p] = "保底" if p >= TIER_THRESHOLD_SAFETY else "适中"

    _logger.debug(
        "compute_tiers | counts: 保底=%d 适中=%d 冲刺=%d",
        sum(1 for p in probs if label[p] == "保底"),
        sum(1 for p in probs if label[p] == "适中"),
        sum(1 for p in probs if label[p] == "冲刺"),
    )
    return [label[p] for p in probs]


class ExtractedBackground(BaseModel):
    model_config = {"extra": "allow"}

    university: str = ""
    major: str = ""
    major_2: str = ""
    degree_type: str = ""
    gpa: float | None = None
    gpa_scale: str = ""
    language_score: float | None = None
    language_type: str = ""
    standardized_test_type: str = ""
    standardized_test_score: float | None = None
    country: str = ""
    target_schools: list[str] = Field(default_factory=list)
    target_majors: list[str] = Field(default_factory=list)
    research: str = ""
    research_count: int | None = None
    internship: str = ""
    internship_count: int | None = None
    award: str = ""
    award_count: int | None = None
    paper: str = ""
    paper_count: int | None = None

    def keys(self):
        return self.model_fields.keys()

    def items(self):
        return ((k, getattr(self, k)) for k in self.model_fields)

    def values(self):
        return (getattr(self, k) for k in self.model_fields)

    def __iter__(self):
        return iter(self.model_fields)

    def __getitem__(self, key: str) -> Any:
        if key not in self.model_fields:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self.model_fields:
            return default
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.model_fields

    def __bool__(self) -> bool:
        for field_name in self.model_fields:
            val = getattr(self, field_name)
            if val not in (None, "", [], {}):
                return True
        return False
