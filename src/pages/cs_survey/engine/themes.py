from __future__ import annotations

from collections import Counter

import pandas as pd

from ..schema import FeedbackDetailSpec, ProductGroupSpec, ScoringSpec, SurveyConfig, ThemeRuleset
from ..text_utils import is_meaningful_text


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
            if is_meaningful_text(t):
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


def build_theme_rows_from_feedback(
    df_feedback: pd.DataFrame,
    ruleset: ThemeRuleset,
    *,
    mode: str,
) -> list[dict]:
    if (
        df_feedback.empty
        or "反馈类型" not in df_feedback.columns
        or "反馈" not in df_feedback.columns
    ):
        return []
    target = "意见" if mode == "opinion" else "表扬"
    texts = [
        str(x).strip()
        for x in df_feedback.loc[df_feedback["反馈类型"].astype(str).str.strip() == target, "反馈"]
        .astype(str)
        .tolist()
        if is_meaningful_text(x)
    ]
    cnt = Counter(classify(t, ruleset) for t in texts)
    rows = [{"反馈类型": theme, "反馈量": int(cnt.get(theme, 0))} for theme in ruleset.known_themes]
    rows = [r for r in rows if r["反馈量"] > 0]
    rows.sort(key=lambda r: r["反馈量"], reverse=True)
    return rows


def annotate_feedback_themes(
    df_feedback: pd.DataFrame,
    opinion_ruleset: ThemeRuleset,
    praise_ruleset: ThemeRuleset,
    *,
    theme_col: str = "主题",
) -> pd.DataFrame:
    if (
        df_feedback.empty
        or "反馈类型" not in df_feedback.columns
        or "反馈" not in df_feedback.columns
    ):
        return df_feedback.copy()
    out = df_feedback.copy()
    themes: list[str] = []
    for _, row in out.iterrows():
        feedback_type = str(row.get("反馈类型", "")).strip()
        text = str(row.get("反馈", "")).strip()
        if feedback_type == "意见":
            themes.append(classify(text, opinion_ruleset))
        elif feedback_type == "表扬":
            themes.append(classify(text, praise_ruleset))
        else:
            themes.append("")
    out[theme_col] = themes
    return out


def build_feedback_detail_rows(
    df: pd.DataFrame,
    dim1_short: str,
    spec: list[FeedbackDetailSpec],
) -> list[dict]:
    rows: list[dict] = []
    for s in spec:
        if getattr(s, "type", "likert") == "text_pair_praise_suggestion":
            praise_tx = (
                df[s.code].astype(str).str.strip()
                if s.code in df.columns
                else pd.Series("", index=df.index, dtype=str)
            )
            opinion_tx = (
                df[s.text].astype(str).str.strip()
                if s.text in df.columns
                else pd.Series("", index=df.index, dtype=str)
            )
            for idx in df.index:
                praise_text = str(praise_tx.loc[idx]).strip()
                if is_meaningful_text(praise_text):
                    rows.append(
                        {
                            "条线": dim1_short,
                            "三大类": s.pillar,
                            "子类": s.subcat,
                            "分数": None,
                            "反馈类型": "表扬",
                            "反馈": praise_text,
                        }
                    )
                opinion_text = str(opinion_tx.loc[idx]).strip()
                if is_meaningful_text(opinion_text):
                    rows.append(
                        {
                            "条线": dim1_short,
                            "三大类": s.pillar,
                            "子类": s.subcat,
                            "分数": None,
                            "反馈类型": "意见",
                            "反馈": opinion_text,
                        }
                    )
            continue
        if getattr(s, "type", "likert") == "ordinal_feedback":
            if s.code not in df.columns:
                continue
            sc = pd.to_numeric(df[s.code], errors="coerce")
            tx = (
                df[s.text].astype(str).str.strip()
                if s.text in df.columns
                else pd.Series("", index=df.index, dtype=str)
            )
            for idx in df.index:
                fv_raw = sc.loc[idx]
                if pd.isna(fv_raw):
                    continue
                text = str(tx.loc[idx]).strip()
                if not is_meaningful_text(text):
                    continue
                fv = float(fv_raw)
                if fv <= 2.0:
                    ftype = "表扬"
                elif fv >= 4.0:
                    ftype = "意见"
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
            continue
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
            if not is_meaningful_text(text):
                continue
            fv = float(v)
            if getattr(s, "type", "likert") == "coded_pair_praise_suggestion":
                if fv == 1.0:
                    ftype = "表扬"
                elif fv == 2.0:
                    ftype = "意见"
                else:
                    continue
            else:
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
