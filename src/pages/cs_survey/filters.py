from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CrossFilterSpec, PillarSpec, ProductGroupSpec, ScoringSpec, SurveyConfig
from .text_utils import meaningful_text_mask


def filter_center(df: pd.DataFrame, spec: CrossFilterSpec, value: str) -> pd.DataFrame:
    if not value or value == spec.all_label:
        return df
    if spec.column not in df.columns:
        return df
    m = df[spec.column].astype(str).str.strip() == value.strip()
    return df.loc[m].copy()


def list_center_values(dfs: dict[str, pd.DataFrame], spec: CrossFilterSpec) -> list[str]:
    s: set[str] = set()
    for df in dfs.values():
        if spec.column in df.columns:
            s.update(df[spec.column].dropna().astype(str).str.strip().unique())
    return [spec.all_label] + sorted(s)


def filter_by_product(
    df: pd.DataFrame, product_cfg: ProductGroupSpec, label: str | None, all_label: str = "全部"
) -> pd.DataFrame:
    if not label or label == all_label:
        return df
    t = product_cfg.type
    p = product_cfg.params
    if t == "q_multi_items":
        base = p["base"]
        items = p.get("items", {})
        target_idx: int | None = None
        for k, v in items.items():
            if v == label:
                target_idx = int(k)
                break
        if target_idx is None:
            return df
        col = f"{base}__{target_idx}"
        if col not in df.columns:
            return df.iloc[0:0].copy()
        s = pd.to_numeric(df[col], errors="coerce")
        m = s.notna()
        return df.loc[m].copy()
    if t == "coded_pair_praise_suggestion":
        pair = next((x for x in p.get("pairs", []) if x["label"] == label), None)
        if pair is None:
            return df.iloc[0:0].copy()
        code_col = pair["code"]
        text_col = pair["text"]
        if code_col not in df.columns:
            return df.iloc[0:0].copy()
        s = df[code_col].astype(str).str.strip()
        code_numeric = s.isin(["1", "2", "表扬", "建议"]) | pd.to_numeric(
            df[code_col], errors="coerce"
        ).isin([1, 2])
        praise_txt = meaningful_text_mask(df[code_col])
        opinion_txt = (
            meaningful_text_mask(df[text_col])
            if text_col in df.columns
            else pd.Series(False, index=df.index)
        )
        m = code_numeric | praise_txt | opinion_txt
        return df.loc[m].copy()
    return df


def _pillar_score_cols(spec: ScoringSpec) -> list[str]:
    t = spec.type
    p = spec.params
    if t == "likert":
        return [pair[0] for pair in p.get("pairs", [])]
    if t == "ordinal_low_good":
        return list(p.get("cols", []))
    if t == "coded_pair_praise_suggestion":
        return [p["code"]]
    if t == "q_multi_items":
        base = p["base"]
        items = p.get("items", {})
        return [f"{base}__{i}" for i in sorted(items.keys(), key=lambda x: int(x))]
    return []


def filter_by_pillar(
    df: pd.DataFrame,
    pillar: PillarSpec | None,
    source_id: str,
) -> pd.DataFrame:
    if pillar is None:
        return df
    spec = pillar.scoring.get(source_id)
    if spec is None:
        return df
    cols = _pillar_score_cols(spec)
    if not cols:
        return df
    m = np.zeros(len(df), dtype=bool)
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        m |= s.notna().to_numpy()
    return df.loc[m].copy()


def find_pillar(cfg: SurveyConfig, name: str) -> PillarSpec | None:
    for p in cfg.pillars:
        if p.name == name:
            return p
    return None
