from __future__ import annotations

from collections import Counter

import pandas as pd

from ..schema import FeedbackDetailSpec, ProductGroupSpec, ScoringSpec, SurveyConfig, ThemeRuleset


def classify(text: str, ruleset: ThemeRuleset) -> str:
    t = (text or "").strip()
    if not t:
        return ruleset.default
    for rule in ruleset.rules:
        for group in rule.any_of:
            if all(kw in t for kw in group):
                return rule.theme
    return ruleset.default


def _collect_texts_likert(
    df: pd.DataFrame, pairs: list[tuple[str, str, float, float]], *, mode: str
) -> list[str]:
    out: list[str] = []
    for code_col, text_col, opinion_max, praise_min in pairs:
        if code_col not in df.columns:
            continue
        sc = pd.to_numeric(df[code_col], errors="coerce")
        if text_col not in df.columns:
            continue
        tx = df[text_col].astype(str).str.strip()
        for idx in df.index:
            v = sc.loc[idx]
            if pd.isna(v):
                continue
            fv = float(v)
            if mode == "opinion" and fv > opinion_max:
                continue
            if mode == "praise" and fv < praise_min:
                continue
            t = str(tx.loc[idx]).strip()
            if t and t.lower() != "nan":
                out.append(t)
    return out


def theme_pairs_for_source(
    cfg: SurveyConfig, source_id: str
) -> list[tuple[str, str, float, float]]:
    specs: list[ScoringSpec] = []
    for pillar in cfg.pillars:
        spec = pillar.scoring.get(source_id)
        if spec is not None:
            specs.append(spec)
    prod: ProductGroupSpec | None = cfg.products.get(source_id)
    if prod is not None and prod.type == "q_multi_items":
        specs.append(ScoringSpec(type="q_multi_items", params=prod.params))
    return feedback_pairs_from_specs(specs)


def feedback_pairs_from_specs(
    specs: list[ScoringSpec],
) -> list[tuple[str, str, float, float]]:
    out: list[tuple[str, str, float, float]] = []
    for spec in specs:
        t = spec.type
        p = spec.params
        if t == "likert":
            opinion_max = float(p.get("opinion_max", 2.0))
            praise_min = float(p.get("praise_min", 4.0))
            out.extend((x[0], x[1], opinion_max, praise_min) for x in p.get("pairs", []))
        elif t == "q_multi_items":
            base = p["base"]
            items = p.get("items", {})
            opinion_max = float(p.get("opinion_max", 3.0))
            praise_min = float(p.get("praise_min", 4.0))
            for i in sorted(items.keys(), key=lambda x: int(x)):
                out.append((f"{base}__{i}", f"{base}__{i}__open2", opinion_max, praise_min))
    return out


def build_theme_rows(
    df: pd.DataFrame,
    pairs: list[tuple[str, str, float, float]],
    ruleset: ThemeRuleset,
    *,
    mode: str,
) -> list[dict]:
    texts = _collect_texts_likert(df, pairs, mode=mode)
    cnt = Counter(classify(t, ruleset) for t in texts)
    rows = [{"反馈类型": theme, "反馈量": int(cnt.get(theme, 0))} for theme in ruleset.known_themes]
    rows = [r for r in rows if r["反馈量"] > 0]
    rows.sort(key=lambda r: r["反馈量"], reverse=True)
    return rows


def build_feedback_detail_rows(
    df: pd.DataFrame,
    dim1_short: str,
    spec: list[FeedbackDetailSpec],
) -> list[dict]:
    rows: list[dict] = []
    for s in spec:
        if s.code not in df.columns:
            continue
        sc = pd.to_numeric(df[s.code], errors="coerce")
        tx = (
            df[s.text].astype(str).str.strip()
            if s.text in df.columns
            else pd.Series("", index=df.index, dtype=str)
        )
        for idx in df.index:
            v = sc.loc[idx]
            if pd.isna(v):
                continue
            text = str(tx.loc[idx]).strip()
            if not text or text.lower() == "nan":
                continue
            fv = float(v)
            opinion_max = float(getattr(s, "opinion_max", 2.0))
            praise_min = float(getattr(s, "praise_min", 4.0))
            if fv <= opinion_max:
                ftype = "意见"
            elif fv >= praise_min:
                ftype = "表扬"
            else:
                continue
            rows.append(
                {
                    "条线": dim1_short,
                    "三大类": s.pillar,
                    "子类": s.subcat,
                    "分数": round(fv, 2),
                    "反馈类型": ftype,
                    "反馈": text,
                }
            )
    return rows
