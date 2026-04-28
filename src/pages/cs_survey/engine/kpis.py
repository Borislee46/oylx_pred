from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import SurveyConfig
from .aggregations import pool_scores
from .scoring import score


def kpi_source(df: pd.DataFrame, cfg: SurveyConfig, source_id: str) -> dict:
    vals = pool_scores(df, cfg, source_id)
    mean = float(np.mean(vals)) if vals else float("nan")

    op_rids: set = set()
    pr_rids: set = set()
    op_n = 0
    pr_n = 0
    for pillar in cfg.pillars:
        spec = pillar.scoring.get(source_id)
        if spec is None:
            continue
        r = score(df, spec)
        op_rids |= r.op_rids
        pr_rids |= r.pr_rids
        op_n += r.op_n
        pr_n += r.pr_n
    n = len(df)
    op_r = (len(op_rids) / n * 100) if n else 0.0
    pr_r = (len(pr_rids) / n * 100) if n else 0.0
    return {
        "评分均值": round(mean, 3) if mean == mean else None,
        "评分量": len(vals),
        "意见率": round(op_r, 2),
        "表扬率": round(pr_r, 2),
        "意见量": int(op_n),
        "表扬量": int(pr_n),
    }


def build_compare_lines(dfs: dict[str, pd.DataFrame], cfg: SurveyConfig) -> pd.DataFrame:
    rows = []
    for src in cfg.sources:
        df = dfs.get(src.id)
        if df is None:
            continue
        k = kpi_source(df, cfg, src.id)
        rows.append(
            {
                "维度": src.label,
                "评分量": k["评分量"],
                "评分均值": k["评分均值"],
                "意见率": k["意见率"],
                "表扬率": k["表扬率"],
                "意见量": k["意见量"],
                "表扬量": k["表扬量"],
            }
        )
    return pd.DataFrame(rows)


def overview_kpis(dfs: dict[str, pd.DataFrame], cfg: SurveyConfig) -> dict:
    per_source = {
        src.id: kpi_source(dfs[src.id], cfg, src.id) for src in cfg.sources if src.id in dfs
    }

    sizes = {sid: len(df) for sid, df in dfs.items()}
    n_total = sum(sizes.values())
    score_total = sum(k["评分量"] for k in per_source.values())
    avg_scores_per_resp = (score_total / n_total) if n_total else 0.0

    def _weighted(key: str) -> float:
        if not n_total:
            return 0.0
        return sum(per_source[sid][key] * sizes[sid] for sid in per_source) / n_total

    opinion_rate = _weighted("意见率")
    praise_rate = _weighted("表扬率")

    def _pooled_mean() -> float:
        if not score_total:
            return float("nan")
        num = 0.0
        den = 0
        for _sid, k in per_source.items():
            m = k["评分均值"]
            q = int(k["评分量"])
            if m is None or q == 0:
                continue
            num += float(m) * q
            den += q
        return num / den if den else float("nan")

    mean_pooled = _pooled_mean()

    pillar_scores: dict[str, float] = {}
    for pillar in cfg.pillars:
        num = 0.0
        den = 0
        for sid, df in dfs.items():
            spec = pillar.scoring.get(sid)
            if spec is None:
                continue
            r = score(df, spec)
            if r.mean != r.mean:
                continue
            num += float(r.mean) * sizes.get(sid, 0)
            den += sizes.get(sid, 0)
        pillar_scores[pillar.name] = (num / den) if den else float("nan")

    return {
        "n_total": n_total,
        "score_total": score_total,
        "avg_scores_per_resp": avg_scores_per_resp,
        "opinion_rate": opinion_rate,
        "praise_rate": praise_rate,
        "mean_pooled": mean_pooled,
        "pillars": pillar_scores,
        "per_source": per_source,
        "sizes": sizes,
        "compare": build_compare_lines(dfs, cfg),
    }
