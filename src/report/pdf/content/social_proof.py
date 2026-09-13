from __future__ import annotations

from src.pages.prediction.data_facts import N_SAMPLES


def build_background_social_proof(unified_results: list[dict] | None) -> str | None:
    unified = [r for r in (unified_results or []) if isinstance(r, dict)]
    samples_str = f"{N_SAMPLES / 10000:.0f} 万+"

    lines: list[str] = [
        f"<b>Signals 已比对 {samples_str} 份真实亚英申请</b>，以下雷达图展示你的背景竞争力维度。"
    ]

    baseline_total = 0
    baseline_combos = 0
    for r in unified:
        n = int(r.get("_baseline_sample_count", 0) or 0)
        if n > 0:
            baseline_total += n
            baseline_combos += 1

    seniors_total = _aggregate_reference_pool(unified)

    if seniors_total > 0:
        lines.append(
            f"推荐方案所涉院校专业，历史库中共有 <b>{seniors_total}</b> 位"
            "背景相近的学长学姐录取案例可参考。"
        )
    elif baseline_total > 0:
        lines.append(
            f"本方案 <b>{baseline_combos}</b> 个推荐组合共参考 "
            f"<b>{baseline_total}</b> 条同类历史申请记录。"
        )

    return "<br/>".join(lines)


def _aggregate_reference_pool(unified: list[dict]) -> int:
    from src.adjustment.knn_retrieval import reference_pool_size

    seen: set[tuple[str, str]] = set()
    total = 0
    for r in unified:
        uni = str(r.get("university", "") or "").strip()
        major = str(r.get("major", "") or "").strip()
        if not uni:
            continue
        key = (uni, major)
        if key in seen:
            continue
        seen.add(key)
        try:
            n = reference_pool_size(uni, major or None)
        except Exception:
            n = 0
        if n > 0:
            total += n
    return total
