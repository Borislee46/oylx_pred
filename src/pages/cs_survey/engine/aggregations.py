from __future__ import annotations

import pandas as pd

from ..schema import (
    DetailRowSpec,
    MultiSelectSpec,
    PillarSpec,
    ProductGroupSpec,
    ScoringSpec,
    SurveyConfig,
)
from ..text_utils import is_meaningful_text
from .scoring import ScoreResult, _reverse_low_good, score


def _row(name_col: str, name: str, r: ScoreResult) -> dict:
    return {
        name_col: name,
        "评分均值": round(r.mean, 3) if r.mean == r.mean else None,
        "意见率": round(r.op_rate, 2),
        "表扬率": round(r.pr_rate, 2),
        "意见量": int(r.op_n),
        "表扬量": int(r.pr_n),
    }


def build_pillar_matrix(
    df: pd.DataFrame, pillars: list[PillarSpec], source_id: str
) -> pd.DataFrame:
    rows = []
    for p in pillars:
        spec = p.scoring.get(source_id)
        if spec is None:
            continue
        rows.append(_row("维度", p.name, score(df, spec)))
    return pd.DataFrame(rows)


def build_detail_matrix(df: pd.DataFrame, details: list[DetailRowSpec]) -> pd.DataFrame:
    rows = []
    for d in details:
        r = score(df, d.scoring)
        row = {
            "大类": d.pillar,
            "明细": d.label,
            "评分均值": round(r.mean, 3) if r.mean == r.mean else None,
            "意见率": round(r.op_rate, 2),
            "表扬率": round(r.pr_rate, 2),
            "意见量": int(r.op_n),
            "表扬量": int(r.pr_n),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_product_matrix(df: pd.DataFrame, product_cfg: ProductGroupSpec) -> pd.DataFrame:
    rows = []
    t = product_cfg.type
    p = product_cfg.params
    if t == "q_multi_items":
        base = p["base"]
        items = p.get("items", {})
        for idx in sorted(items.keys(), key=lambda x: int(x)):
            spec = ScoringSpec(
                type="likert",
                params={
                    "pairs": [[f"{base}__{idx}", f"{base}__{idx}__open2"]],
                    "opinion_max": float(p.get("opinion_max", 3.0)),
                    "praise_min": float(p.get("praise_min", 4.0)),
                },
            )
            rows.append(_row("维度", items[idx], score(df, spec)))
    elif t == "coded_pair_praise_suggestion":
        for pair in p.get("pairs", []):
            spec = ScoringSpec(
                type="coded_pair_praise_suggestion",
                params={"code": pair["code"], "text": pair["text"]},
            )
            rows.append(_row("维度", pair["label"], score(df, spec)))
    return pd.DataFrame(rows)


def build_product_detail(df: pd.DataFrame, product_cfg: ProductGroupSpec) -> pd.DataFrame:
    matrix = build_product_matrix(df, product_cfg)
    if matrix.empty:
        return matrix
    name = "产品大类" if product_cfg.type == "q_multi_items" else "产品类型"
    out = matrix.rename(columns={"维度": "明细"}).copy()
    out.insert(0, "大类", name)
    return out


def score_distribution(
    df: pd.DataFrame, cols: list[str], reverse_cols: set[str] | None = None
) -> pd.DataFrame:
    vals: list[int] = []
    reverse = reverse_cols or set()
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if c in reverse:
            s = s.map(_reverse_low_good)
        s = s[(s >= 1) & (s <= 5)]
        vals.extend(s.round().astype(int).tolist())
    vc = pd.Series(vals).value_counts() if vals else pd.Series(dtype=int)
    return pd.DataFrame([{"分值": k, "数量": int(vc.get(k, 0))} for k in range(1, 6)])


def source_score_distribution(df: pd.DataFrame, cfg: SurveyConfig, source_id: str) -> pd.DataFrame:
    vals = [int(round(x)) for x in pool_scores(df, cfg, source_id) if 1.0 <= float(x) <= 5.0]
    vc = pd.Series(vals).value_counts() if vals else pd.Series(dtype=int)
    return pd.DataFrame([{"分值": k, "数量": int(vc.get(k, 0))} for k in range(1, 6)])


def pool_scores(df: pd.DataFrame, cfg: SurveyConfig, source_id: str) -> list[float]:
    spec = cfg.score_cols.get(source_id)
    if spec is None:
        vals: list[float] = []
    else:
        vals: list[float] = []
        if spec.mode == "raw":
            cols = spec.cols
            for c in cols:
                if c not in df.columns:
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                for x in s:
                    if pd.isna(x):
                        continue
                    fv = float(x)
                    if 1.0 <= fv <= 5.0:
                        vals.append(fv)
        elif spec.mode == "split":
            for c in spec.raw:
                if c not in df.columns:
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                for x in s:
                    if pd.isna(x):
                        continue
                    fv = float(x)
                    if 1.0 <= fv <= 5.0:
                        vals.append(fv)
            for c in spec.reverse:
                if c not in df.columns:
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                for x in s:
                    if pd.isna(x):
                        continue
                    fv = float(x)
                    if 1.0 <= fv <= 5.0:
                        vals.append(_reverse_low_good(fv))

    product_cfg = cfg.products.get(source_id)
    if product_cfg is not None and product_cfg.type == "coded_pair_praise_suggestion":
        for pair in product_cfg.params.get("pairs", []):
            r = score(
                df,
                ScoringSpec(
                    type="coded_pair_praise_suggestion",
                    params={"code": pair["code"], "text": pair["text"]},
                ),
            )
            vals.extend(r.scores)
    return vals


def build_multi_select_counts(df: pd.DataFrame, spec: MultiSelectSpec) -> pd.DataFrame:
    rows = []
    for idx_str in sorted(spec.items.keys(), key=lambda x: int(x)):
        col = f"{spec.base}__{idx_str}"
        if col not in df.columns:
            continue
        count = int((pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).sum())
        label = spec.items[idx_str]
        rows.append({"option": label, "count": count})
    return pd.DataFrame(rows)


def build_product_feedback_summary(df: pd.DataFrame, product_cfg: ProductGroupSpec) -> pd.DataFrame:
    if product_cfg is None or product_cfg.type != "coded_pair_praise_suggestion":
        return pd.DataFrame()
    rows = []
    for pair in product_cfg.params.get("pairs", []):
        code_col = pair["code"]
        text_col = pair["text"]
        label = pair["label"]
        praise_count = 0
        opinion_count = 0
        code_tx = (
            df[code_col].apply(is_meaningful_text)
            if code_col in df.columns
            else pd.Series(False, index=df.index)
        )
        opinion_tx = (
            df[text_col].apply(is_meaningful_text)
            if text_col in df.columns
            else pd.Series(False, index=df.index)
        )
        for idx in df.index:
            if bool(code_tx.loc[idx]):
                praise_count += 1
            if bool(opinion_tx.loc[idx]):
                opinion_count += 1
        total = praise_count + opinion_count
        rows.append(
            {
                "产品类型": label,
                "表扬量": praise_count,
                "意见量": opinion_count,
                "总反馈量": total,
            }
        )
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("总反馈量", ascending=False)
    return df_out
