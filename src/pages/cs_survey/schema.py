from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataSource:
    id: str
    label: str
    path: str
    dim1_short: str = ""


@dataclass
class ScoringSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScoringSpec:
        t = d["type"]
        params = {k: v for k, v in d.items() if k != "type"}
        return cls(type=t, params=params)


@dataclass
class PillarSpec:
    name: str
    scoring: dict[str, ScoringSpec]


@dataclass
class DetailRowSpec:
    pillar: str
    label: str
    scoring: ScoringSpec


@dataclass
class FeedbackDetailSpec:
    code: str
    text: str
    pillar: str
    subcat: str
    opinion_max: float = 2.0
    praise_min: float = 4.0


@dataclass
class ProductGroupSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        if self.type == "q_multi_items":
            items = self.params.get("items", {})
            return [items[k] for k in sorted(items.keys(), key=lambda x: int(x))]
        if self.type == "coded_pair_praise_suggestion":
            return [p["label"] for p in self.params.get("pairs", [])]
        return []


@dataclass
class ScoreColsSpec:
    mode: str
    cols: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    reverse: list[str] = field(default_factory=list)


@dataclass
class ThemeRule:
    theme: str
    any_of: list[list[str]] = field(default_factory=list)


@dataclass
class ThemeRuleset:
    default: str
    rules: list[ThemeRule]

    @property
    def known_themes(self) -> list[str]:
        seen: list[str] = []
        for r in self.rules:
            if r.theme not in seen:
                seen.append(r.theme)
        if self.default not in seen:
            seen.append(self.default)
        return seen


@dataclass
class ViewSpec:
    type: str
    key: str
    label: str


@dataclass
class CrossFilterSpec:
    column: str
    all_label: str = "全部"
    label: str = ""


@dataclass
class SurveyConfig:
    id: str
    title: str
    subtitle: str
    updated_at: str
    sources: list[DataSource]
    cross_filters: dict[str, CrossFilterSpec]
    pillars: list[PillarSpec]
    products: dict[str, ProductGroupSpec]
    score_cols: dict[str, ScoreColsSpec]
    score_distribution_cols: dict[str, list[str]]
    details: dict[str, list[DetailRowSpec]]
    feedback_detail: dict[str, list[FeedbackDetailSpec]]
    themes: dict[str, ThemeRuleset]
    views: list[ViewSpec]
    source_path: str = ""

    @property
    def pillar_names(self) -> list[str]:
        return [p.name for p in self.pillars]

    def source(self, source_id: str) -> DataSource:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)

    def view(self, key: str) -> ViewSpec | None:
        for v in self.views:
            if v.key == key:
                return v
        return None


def parse_survey_config(raw: dict[str, Any], source_path: str = "") -> SurveyConfig:
    sources = [DataSource(**s) for s in raw["sources"]]

    cross_filters = {
        name: CrossFilterSpec(**spec) for name, spec in (raw.get("cross_filters") or {}).items()
    }

    pillars: list[PillarSpec] = []
    for p in raw.get("pillars", []):
        scoring = {sid: ScoringSpec.from_dict(s) for sid, s in p["scoring"].items()}
        pillars.append(PillarSpec(name=p["name"], scoring=scoring))

    products: dict[str, ProductGroupSpec] = {}
    for sid, spec in (raw.get("products") or {}).items():
        t = spec["type"]
        params = {k: v for k, v in spec.items() if k != "type"}
        products[sid] = ProductGroupSpec(type=t, params=params)

    score_cols: dict[str, ScoreColsSpec] = {}
    for sid, spec in (raw.get("score_cols") or {}).items():
        score_cols[sid] = ScoreColsSpec(
            mode=spec["mode"],
            cols=list(spec.get("cols", [])),
            raw=list(spec.get("raw", [])),
            reverse=list(spec.get("reverse", [])),
        )

    dist_cols = {
        sid: list(cols) for sid, cols in (raw.get("score_distribution_cols") or {}).items()
    }

    details: dict[str, list[DetailRowSpec]] = {}
    for sid, rows in (raw.get("details") or {}).items():
        details[sid] = [
            DetailRowSpec(
                pillar=r["pillar"],
                label=r["label"],
                scoring=ScoringSpec.from_dict(r["scoring"]),
            )
            for r in rows
        ]

    feedback_detail: dict[str, list[FeedbackDetailSpec]] = {}
    for sid, rows in (raw.get("feedback_detail") or {}).items():
        feedback_detail[sid] = [FeedbackDetailSpec(**r) for r in rows]

    themes: dict[str, ThemeRuleset] = {}
    for kind, spec in (raw.get("themes") or {}).items():
        rules = [
            ThemeRule(theme=r["theme"], any_of=[list(g) for g in r.get("any_of", [])])
            for r in spec.get("rules", [])
        ]
        themes[kind] = ThemeRuleset(default=spec.get("default", "其他"), rules=rules)

    views = [ViewSpec(**v) for v in raw.get("views", [])]

    return SurveyConfig(
        id=raw["id"],
        title=raw.get("title", raw["id"]),
        subtitle=raw.get("subtitle", ""),
        updated_at=str(raw.get("updated_at", "")),
        sources=sources,
        cross_filters=cross_filters,
        pillars=pillars,
        products=products,
        score_cols=score_cols,
        score_distribution_cols=dist_cols,
        details=details,
        feedback_detail=feedback_detail,
        themes=themes,
        views=views,
        source_path=source_path,
    )
