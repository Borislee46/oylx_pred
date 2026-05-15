from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schema import ScoringSpec
from ..text_utils import meaningful_text_mask


@dataclass
class ScoreResult:
    mean: float
    op_rate: float
    pr_rate: float
    op_n: int
    pr_n: int
    op_rids: set = field(default_factory=set)
    pr_rids: set = field(default_factory=set)
    scores: list[float] = field(default_factory=list)


def _rid_series(df: pd.DataFrame) -> pd.Series:
    if "rid" in df.columns:
        return df["rid"]
    return pd.Series(df.index, index=df.index)


def _reverse_low_good(value: float) -> float:
    return 6.0 - value


def score_likert(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    *,
    opinion_max: float = 2.0,
    praise_min: float = 4.0,
) -> ScoreResult:
    scores: list[float] = []
    op_rids: set = set()
    pr_rids: set = set()
    op_n = 0
    pr_n = 0
    rid = _rid_series(df)
    for code_col, _text_col in pairs:
        if code_col not in df.columns:
            continue
        sc = pd.to_numeric(df[code_col], errors="coerce")
        for idx in df.index:
            v = sc.loc[idx]
            if pd.isna(v):
                continue
            scores.append(float(v))
            if v <= opinion_max:
                op_n += 1
                op_rids.add(rid.loc[idx])
            elif v >= praise_min:
                pr_n += 1
                pr_rids.add(rid.loc[idx])
    n = len(scores)
    mean = float(np.mean(scores)) if scores else float("nan")
    op_r = (op_n / n * 100) if n else 0.0
    pr_r = (pr_n / n * 100) if n else 0.0
    return ScoreResult(mean, op_r, pr_r, op_n, pr_n, op_rids, pr_rids, scores)


def score_ordinal_low_good(df: pd.DataFrame, cols: list[str]) -> ScoreResult:
    aligned: list[float] = []
    op_rids: set = set()
    pr_rids: set = set()
    op_n = 0
    pr_n = 0
    rid = _rid_series(df)
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        for idx in df.index:
            x = s.loc[idx]
            if pd.isna(x):
                continue
            fv = float(x)
            aligned.append(_reverse_low_good(fv))
            if fv >= 4:
                op_n += 1
                op_rids.add(rid.loc[idx])
            elif fv <= 2:
                pr_n += 1
                pr_rids.add(rid.loc[idx])
    n = len(aligned)
    mean = float(np.mean(aligned)) if aligned else float("nan")
    op_r = (op_n / n * 100) if n else 0.0
    pr_r = (pr_n / n * 100) if n else 0.0
    return ScoreResult(mean, op_r, pr_r, op_n, pr_n, op_rids, pr_rids, aligned)


def score_coded_pair(df: pd.DataFrame, code_col: str, text_col: str) -> ScoreResult:
    if code_col not in df.columns:
        return ScoreResult(float("nan"), 0.0, 0.0, 0, 0)
    sc = pd.to_numeric(df[code_col], errors="coerce")
    code_txt = meaningful_text_mask(df[code_col])
    opinion_txt = (
        meaningful_text_mask(df[text_col])
        if text_col in df.columns
        else pd.Series(False, index=df.index)
    )
    rid = _rid_series(df)
    scores: list[float] = []
    op_rids: set = set()
    pr_rids: set = set()
    op_n = 0
    pr_n = 0
    for idx in df.index:
        v = sc.loc[idx]
        fv = float(v) if pd.notna(v) else None
        pr_num = fv == 1.0
        op_num = fv == 2.0
        pr_t = bool(code_txt.loc[idx])
        op_t = bool(opinion_txt.loc[idx])
        row_scores: list[float] = []
        if pr_num or pr_t:
            row_scores.append(5.0)
        if op_num or op_t:
            row_scores.append(2.0)
        if not row_scores:
            continue
        scores.extend(row_scores)
        if op_num or op_t:
            op_n += 1
            op_rids.add(rid.loc[idx])
        if pr_num or pr_t:
            pr_n += 1
            pr_rids.add(rid.loc[idx])
    n = len(scores)
    mean = float(np.mean(scores)) if scores else float("nan")
    op_r = (op_n / n * 100) if n else 0.0
    pr_r = (pr_n / n * 100) if n else 0.0
    return ScoreResult(mean, op_r, pr_r, op_n, pr_n, op_rids, pr_rids, scores)


def score(df: pd.DataFrame, spec: ScoringSpec) -> ScoreResult:
    t = spec.type
    p = spec.params
    if t == "likert":
        pairs = [tuple(x) for x in p.get("pairs", [])]
        return score_likert(
            df,
            pairs,
            opinion_max=float(p.get("opinion_max", 2.0)),
            praise_min=float(p.get("praise_min", 4.0)),
        )
    if t == "ordinal_low_good":
        return score_ordinal_low_good(df, list(p.get("cols", [])))
    if t == "coded_pair_praise_suggestion":
        return score_coded_pair(df, p["code"], p["text"])
    if t == "q_multi_items":
        base = p["base"]
        items = p.get("items", {})
        idxs = sorted(items.keys(), key=lambda x: int(x))
        pairs = [(f"{base}__{i}", f"{base}__{i}__open2") for i in idxs]
        return score_likert(
            df,
            pairs,
            opinion_max=float(p.get("opinion_max", 3.0)),
            praise_min=float(p.get("praise_min", 4.0)),
        )
    raise ValueError(f"unsupported scoring type: {t}")
