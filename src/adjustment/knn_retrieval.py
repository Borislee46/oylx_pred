from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

from src.adjustment.config import (
    DEFAULT_UNIVERSITY_DIFFICULTY_ORDER,
)
from src.adjustment.tier_calibration import build_tier_map
from src.pages.prediction.app_data import (
    load_raw_cases_data,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability
from src.utils.schools.constants import SCHOOL_LEVEL_SCORES
from src.utils.schools.level_service import get_school_level_service

logger = setup_logger("page3", "prediction")

DEFAULT_WEIGHTS: tuple[float, float, float, float] = (0.30, 0.20, 0.15, 0.35)

_LEVEL_ALIASES = {"500之后": "500+", "50-100": "51-100"}

_SCHOOL_SCORE_SPAN = 0.75
_Z_SPAN = 3.0
_SAME_TIER_DIFF_SCHOOL_PENALTY = 0.15
_UNDERDOG_MAX_TIER = 3
_LOW_GPA_CUTOFF = 2.5
_UNDERDOG_STUDENT_SCHOOL = 0.65
_UNDERDOG_STUDENT_GPA = 2.8
_UNDERDOG_GPA_CUSHION = 0.3

_SCHOOL_LEVEL_SHORT: dict[str, str] = {
    "普通本科": "二本",
    "三本/民办本科": "三本",
    "专接本": "专接本",
    "专科": "专科",
    "500+": "500+",
}

_DEDUP_COLS = [
    "target_university",
    "target_major",
    "background_university",
    "background_major",
    "gpa",
]


def _norm_key(value) -> str:
    return str(value).strip().lower()


def _school_score(uni: str) -> tuple[float, bool]:
    level = get_school_level_service().get_school_level(uni)
    level = _LEVEL_ALIASES.get(level, level)
    if level == "未知" or level not in SCHOOL_LEVEL_SCORES:
        return SCHOOL_LEVEL_SCORES["未知"], False
    return float(SCHOOL_LEVEL_SCORES[level]), True


def _precompute_stats(cases_df: pd.DataFrame) -> dict:
    stats: dict = {}

    gpa_valid = cases_df["gpa"].dropna()
    stats["gpa_mean"] = float(gpa_valid.mean()) if len(gpa_valid) else 0.0
    stats["gpa_std"] = float(gpa_valid.std()) if len(gpa_valid) > 1 else 1.0
    if math.isnan(stats["gpa_std"]):
        stats["gpa_std"] = 1.0
    lang_valid = cases_df["language_score"].dropna()
    stats["lang_mean"] = float(lang_valid.mean()) if len(lang_valid) else 0.0
    stats["lang_std"] = float(lang_valid.std()) if len(lang_valid) > 1 else 1.0
    if math.isnan(stats["lang_std"]):
        stats["lang_std"] = 1.0

    admitted = cases_df[cases_df["admitted"] == 1]
    stats["uni_counts"] = admitted["target_university"].value_counts().to_dict()
    stats["major_counts"] = admitted["target_major"].value_counts().to_dict()

    pool = admitted.drop_duplicates(subset=_DEDUP_COLS).reset_index(drop=True)
    stats["pool"] = pool

    bg_unis = pool["background_university"].astype(str).str.strip()
    pairs = {u: _school_score(u) for u in bg_unis.unique()}
    stats["score"] = bg_unis.map(lambda u: pairs[u][0]).to_numpy(dtype=float)
    stats["score_known"] = bg_unis.map(lambda u: pairs[u][1]).to_numpy(dtype=bool)
    stats["bg_uni"] = bg_unis.to_numpy()

    gpa = pool["gpa"].to_numpy(dtype=float)
    stats["gpa"] = gpa
    stats["gpa_z"] = np.where(np.isnan(gpa), np.nan, (gpa - stats["gpa_mean"]) / stats["gpa_std"])
    lang = pool["language_score"].to_numpy(dtype=float)
    stats["lang_z"] = np.where(
        np.isnan(lang), np.nan, (lang - stats["lang_mean"]) / stats["lang_std"]
    )

    codes, vocab = pd.factorize(pool["background_major"].astype(str).str.strip().str.lower())
    stats["bg_major_code"] = codes
    stats["bg_major_vocab"] = np.asarray(vocab, dtype=str)
    stats["vocab_index"] = {key: i for i, key in enumerate(vocab)}
    stats["target_major"] = pool["target_major"].to_numpy()

    full_pool = cases_df.drop_duplicates(subset=_DEDUP_COLS)
    uni_major_groups = full_pool.groupby(["target_university", "target_major"])
    stats["base_rate"] = {
        (str(uni), str(major)): float(group["admitted"].mean())
        for (uni, major), group in uni_major_groups
    }

    stats["tier_map"] = build_tier_map(DEFAULT_UNIVERSITY_DIFFICULTY_ORDER)
    stats["idx_by_uni"] = dict(pool.groupby("target_university", sort=False).indices)
    stats["idx_by_major"] = dict(pool.groupby("target_major", sort=False).indices)
    tiers = pool["target_university"].map(stats["tier_map"])
    stats["idx_by_tier"] = {int(t): g for t, g in pool.groupby(tiers).indices.items()}

    stats["major_sim_vocab"] = {}
    return stats


@st.cache_resource(show_spinner=False)
def get_knn_stats() -> dict:
    stats = _precompute_stats(load_raw_cases_data())
    return stats


def _major_name_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.5
    if a == b:
        return 1.0
    return float(fuzz.token_sort_ratio(a, b)) / 100.0


def _major_sim_vocab(stats: dict, s_bg_key: str) -> np.ndarray:
    memo = stats.setdefault("major_sim_vocab", {})
    arr = memo.get(s_bg_key)
    if arr is None:
        vocab = stats["bg_major_vocab"]
        arr = np.fromiter(
            (_major_name_sim(s_bg_key, str(bg)) for bg in vocab),
            dtype=float,
            count=len(vocab),
        )
        memo[s_bg_key] = arr
    return arr


def _zscore(val, mean: float, std: float) -> float | None:
    if val is None or pd.isna(val):
        return None
    if std == 0.0:
        return None
    return (float(val) - mean) / std


def _select_pool(stats: dict, target_uni: str, target_major: str, k: int):
    sources: list[tuple[np.ndarray, str]] = []

    idx_uni = stats["idx_by_uni"].get(target_uni)
    if idx_uni is not None:
        sources.append((idx_uni, f"同院校 |C|={len(idx_uni)}"))

    tier = stats["tier_map"].get(target_uni)
    idx_tier = stats["idx_by_tier"].get(tier) if tier else None
    if idx_tier is not None:
        sources.append((idx_tier, f"同Tier院校 |C|={len(idx_tier)}"))

    idx_major = stats["idx_by_major"].get(target_major)
    if idx_major is not None:
        sources.append((idx_major, f"同专业(不限院校) |C|={len(idx_major)}"))

    if not sources:
        logger.warning(
            "KNN 候选池为空 | target_uni=%s target_major=%s — 无索引命中",
            target_uni,
            target_major,
        )
        return None, 4, "无足够相似历史案例"

    mixed: list[int] = []
    level = 0
    for pool_idx, _label in sources:
        new_indices = np.setdiff1d(pool_idx, mixed)
        mixed.extend(new_indices.tolist())
        level += 1
        if len(mixed) >= k:
            break

    if len(mixed) >= k:
        note = f"混合候选池 ({'→'.join(s[1] for s in sources[:level])}) |C|={len(mixed)}"
    else:
        note = f"混合候选池（不足k）|C|={len(mixed)}"
    logger.debug("KNN 候选池 | %s k=%d level=%d", note, k, level)
    if len(mixed) >= k:
        return np.asarray(mixed, dtype=int), level, note
    return np.asarray(mixed, dtype=int), max(1, level), note


def _compute_distances(
    student: dict,
    idx: np.ndarray,
    stats: dict,
    weights: tuple[float, float, float, float],
    s_score: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    w_school, w_gpa, w_lang, w_major = weights

    if s_score is None:
        s_score, _ = _school_score(str(student.get("background_university", "")))
    d_school = np.minimum(np.abs(stats["score"][idx] - s_score) / _SCHOOL_SCORE_SPAN, 1.0)
    s_uni = str(student.get("background_university", "")).strip().lower()
    if s_uni:
        same_school = np.char.lower(stats["bg_uni"][idx].astype(str)) == s_uni
        same_level = np.abs(stats["score"][idx] - s_score) < 1e-12
        d_school = np.where(
            same_level & ~same_school,
            np.maximum(d_school, _SAME_TIER_DIFF_SCHOOL_PENALTY),
            d_school,
        )

    s_gpa_z = _zscore(student.get("gpa"), stats["gpa_mean"], stats["gpa_std"])
    pool_gpa_z = stats["gpa_z"][idx]
    gpa_valid = ~np.isnan(pool_gpa_z)
    if s_gpa_z is not None:
        d_gpa = np.where(
            gpa_valid,
            np.minimum(np.abs(pool_gpa_z - s_gpa_z) / _Z_SPAN, 1.0),
            1.0,
        )
    else:
        d_gpa = np.full(len(idx), 0.5)

    s_lang_z = _zscore(student.get("lang_score"), stats["lang_mean"], stats["lang_std"])
    pool_lang_z = stats["lang_z"][idx]
    lang_valid = ~np.isnan(pool_lang_z)
    if s_lang_z is not None:
        d_lang = np.where(
            lang_valid,
            np.minimum(np.abs(pool_lang_z - s_lang_z) / _Z_SPAN, 1.0),
            1.0,
        )
    else:
        d_lang = np.full(len(idx), 0.5)

    s_bg_key = _norm_key(student.get("background_major") or "")
    if not s_bg_key:
        major_sim = np.full(len(idx), 0.5)
    else:
        major_sim = _major_sim_vocab(stats, s_bg_key)[stats["bg_major_code"][idx]]
    d_major = 1.0 - major_sim

    distance = w_school * d_school + w_gpa * d_gpa + w_lang * d_lang + w_major * d_major
    components = {
        "school": d_school,
        "gpa": d_gpa,
        "lang": d_lang,
        "major": d_major,
    }
    return distance, components


def _exact_mask(student: dict, idx: np.ndarray, stats: dict) -> np.ndarray:
    mask = stats["bg_uni"][idx] == str(student.get("background_university", "")).strip()
    mask &= stats["target_major"][idx] == student["target_major"]
    s_bg_key = _norm_key(student.get("background_major") or "")
    if s_bg_key:
        code = stats["vocab_index"].get(s_bg_key, -1)
        mask &= stats["bg_major_code"][idx] == code
    return mask


def _school_short_label(uni: str) -> str:
    level = get_school_level_service().get_school_level(uni)
    level = _LEVEL_ALIASES.get(level, level)
    return _SCHOOL_LEVEL_SHORT.get(level, level)


def _underdog_pool(
    stats: dict,
    target_uni: str,
    target_major: str,
    target_tier: int,
) -> tuple[np.ndarray | None, str]:
    parts: list[np.ndarray] = []
    labels: list[str] = []

    idx_uni = stats["idx_by_uni"].get(target_uni)
    if idx_uni is not None:
        parts.append(idx_uni)
        labels.append(f"同院校|C|={len(idx_uni)}")

    idx_tier = stats["idx_by_tier"].get(target_tier)
    if idx_tier is not None:
        parts.append(idx_tier)
        labels.append(f"同Tier|C|={len(idx_tier)}")

    idx_major = stats["idx_by_major"].get(target_major)
    if idx_major is not None:
        parts.append(idx_major)
        labels.append(f"同专业|C|={len(idx_major)}")

    if not parts:
        return None, "逆袭候选池为空"

    mixed = np.unique(np.concatenate(parts))
    note = f"逆袭候选池({'→'.join(labels)})|C|={len(mixed)}"
    logger.debug("KNN %s", note)
    return mixed, note


def _find_underdogs(
    student: dict,
    idx: np.ndarray,
    stats: dict,
    distance: np.ndarray,
    exact: np.ndarray,
    target_tier: int,
    s_score: float | None = None,
) -> list[tuple[int, str]]:
    if target_tier > _UNDERDOG_MAX_TIER:
        return []

    results: list[tuple[int, str]] = []
    taken = {int(p) for p in np.flatnonzero(exact)}

    if s_score is None:
        s_score, _known = _school_score(str(student.get("background_university", "")))
    s_gpa = student.get("gpa")
    s_gpa_f = float(s_gpa) if s_gpa is not None and not pd.isna(s_gpa) else None

    logger.debug(
        "KNN 逆袭检测 | target_tier=%d student_score=%.3f student_gpa=%s",
        target_tier,
        s_score,
        s_gpa,
    )

    if s_score <= _UNDERDOG_STUDENT_SCHOOL:
        mask = (
            stats["score_known"][idx]
            & (stats["score"][idx] < s_score)
            & ~np.isnan(stats["gpa"][idx])
        )
        for p in taken:
            mask[p] = False
        if mask.any():
            masked_d = np.where(mask, distance, np.inf)
            pos = int(np.argmin(masked_d))
            case_gpa = float(stats["gpa"][idx[pos]])
            if s_gpa_f is None or case_gpa <= s_gpa_f + _UNDERDOG_GPA_CUSHION:
                case_uni = str(stats["bg_uni"][idx[pos]])
                label = f"{_school_short_label(case_uni)}逆袭"
                results.append((pos, label))
                taken.add(pos)

    if s_gpa_f is not None and s_gpa_f < _UNDERDOG_STUDENT_GPA:
        mask = (stats["gpa"][idx] < _LOW_GPA_CUTOFF) & ~np.isnan(stats["gpa"][idx])
        for p in taken:
            mask[p] = False
        if mask.any():
            masked_d = np.where(mask, distance, np.inf)
            pos = int(np.argmin(masked_d))
            results.append((pos, "低GPA逆袭"))

    if len(results) > 1:
        results.sort(key=lambda x: distance[x[0]])
        results = results[:1]

    return results


def retrieve_similar_cases(
    student: dict,
    k: int = 3,
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> tuple[list[dict], int, str]:
    stats = get_knn_stats()
    target_uni = student["target_university"]
    target_major = student.get("target_major", "?")
    logger.debug(
        "KNN 检索开始 | target=%s@%s k=%d weights=%s",
        target_major,
        target_uni,
        k,
        weights,
    )
    idx, level, note = _select_pool(stats, target_uni, target_major, k)
    if idx is None:
        logger.warning("KNN 检索失败 | target=%s@%s — 候选池为空", target_major, target_uni)
        return [], level, note

    s_score, s_score_known = _school_score(str(student.get("background_university", "")))

    distance, _ = _compute_distances(student, idx, stats, weights, s_score=s_score)
    exact = _exact_mask(student, idx, stats)

    exact_positions = np.flatnonzero(exact)
    if len(exact_positions) > 1:
        s_gpa = student.get("gpa")
        if s_gpa is not None and not pd.isna(s_gpa):
            gpa_diff = np.abs(
                np.nan_to_num(stats["gpa"][idx][exact_positions], nan=np.inf) - float(s_gpa)
            )
            exact_positions = exact_positions[np.argsort(gpa_diff)]
        else:
            exact_positions = exact_positions[np.argsort(distance[exact_positions])]

    target_tier = stats["tier_map"].get(target_uni, 99)
    underdog_global: list[
        tuple[int, int, str, float]
    ] = []
    if target_tier <= _UNDERDOG_MAX_TIER:
        _s_gpa = student.get("gpa")
        _qualifies_school = s_score <= _UNDERDOG_STUDENT_SCHOOL
        _qualifies_gpa = (
            _s_gpa is not None and not pd.isna(_s_gpa) and float(_s_gpa) < _UNDERDOG_STUDENT_GPA
        )
        if _qualifies_school or _qualifies_gpa:
            ud_idx, _ud_note = _underdog_pool(stats, target_uni, target_major, target_tier)
            if ud_idx is not None and len(ud_idx) > 0:
                ud_distance, _ = _compute_distances(
                    student, ud_idx, stats, weights, s_score=s_score
                )
                ud_exact = _exact_mask(student, ud_idx, stats)
                ud_results = _find_underdogs(
                    student, ud_idx, stats, ud_distance, ud_exact, target_tier, s_score=s_score
                )
            exact_global = {int(idx[p]) for p in exact_positions[:k]}
            for pos, label in ud_results:
                g_idx = int(ud_idx[pos])
                if g_idx not in exact_global:
                    underdog_global.append((g_idx, pos, label, float(ud_distance[pos])))
                    exact_global.add(g_idx)

    n_exact_take = min(len(exact_positions), k - len(underdog_global))
    chosen = list(exact_positions[:n_exact_take])
    match_types = ["exact"] * len(chosen)
    _ud_global_by_slot: dict[int, tuple[int, int, str, float]] = {}
    for ud_entry in underdog_global:
        if len(chosen) >= k:
            break
        chosen.append(-1)
        match_types.append("underdog")
        _ud_global_by_slot[len(chosen) - 1] = ud_entry

    if len(chosen) < k:
        taken_global: set[int] = {int(idx[p]) for p in chosen if p >= 0}
        for g_idx, _, _, _ in underdog_global:
            taken_global.add(g_idx)
        n_need = min(k + len(taken_global), len(idx))
        order = np.argpartition(distance, n_need - 1)[:n_need]
        order = order[np.argsort(distance[order])]
        for pos in order:
            if len(chosen) >= k:
                break
            if int(idx[pos]) not in taken_global:
                chosen.append(int(pos))
                match_types.append("similar")

    n_exact = match_types.count("exact")
    if n_exact:
        note += f"，完全匹配 {n_exact}"
    n_ud = match_types.count("underdog")
    if n_ud:
        note += "，含逆袭参考"

    logger.info(
        "KNN 检索完成 | n_cases=%d exact=%d underdog=%d similar=%d | %s",
        len(chosen),
        n_exact,
        n_ud,
        match_types.count("similar"),
        note,
    )

    cases = []
    for slot_idx, (pos, mt) in enumerate(zip(chosen, match_types, strict=True)):
        if mt == "underdog":
            g_idx, ud_pos, ud_label, ud_dist = _ud_global_by_slot[slot_idx]
            row = stats["pool"].iloc[g_idx]
            case = {
                "background_university": row["background_university"],
                "background_major": row["background_major"],
                "target_university": row["target_university"],
                "target_major": row["target_major"],
                "gpa": row["gpa"],
                "ielts": row["ielts"],
                "toefl": row["toefl"],
                "admitted": row["admitted"],
                "similarity": clip_probability(1.0 - ud_dist),
                "distance": ud_dist,
                "match_type": mt,
                "is_upset": True,
                "underdog_kind": ud_label,
                "base_rate": stats["base_rate"].get(
                    (str(row["target_university"]), str(row["target_major"]))
                ),
            }
        else:
            row = stats["pool"].iloc[idx[pos]]
            case = {
                "background_university": row["background_university"],
                "background_major": row["background_major"],
                "target_university": row["target_university"],
                "target_major": row["target_major"],
                "gpa": row["gpa"],
                "ielts": row["ielts"],
                "toefl": row["toefl"],
                "admitted": row["admitted"],
                "similarity": clip_probability(1.0 - distance[pos]),
                "distance": float(distance[pos]),
                "match_type": mt,
                "is_upset": False,
                "base_rate": stats["base_rate"].get(
                    (str(row["target_university"]), str(row["target_major"]))
                ),
            }
        cases.append(case)
    return cases, level, note


def reference_pool_size(target_university: str, target_major: str | None = None) -> int:
    stats = get_knn_stats()
    n = stats["uni_counts"].get(target_university, 0)
    if n > 0:
        return int(n)
    if target_major:
        return int(stats["major_counts"].get(target_major, 0))
    return 0
