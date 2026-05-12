"""DeltaCalculator for comparing prediction probabilities across submissions."""

from __future__ import annotations


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class DeltaCalculator:
    """Compares current vs previous prediction probabilities.

    Matching is by (university, major) as a compound key.
    Three output states per result: "+3.2%", "-1.5%", "NEW", "—".
    """

    @staticmethod
    def build_prob_map(results: list[dict]) -> dict[tuple[str, str], float]:
        """Build {(university, major): probability} lookup from result dicts."""
        prob_map: dict[tuple[str, str], float] = {}
        for r in results or []:
            uni = str(r.get("university", "")).strip()
            major = str(r.get("major", "")).strip()
            if uni and major:
                prob_map[(uni, major)] = _safe_float(r.get("probability"), 0.0)
        return prob_map

    @staticmethod
    def should_show_delta(
        current_target_unis: list[str],
        current_target_majors: list[str],
        current_bg_uni: str,
        current_bg_major: str,
        prev_results: list[dict] | None,
        prev_input_data: dict | None,
    ) -> tuple[bool, dict | None, bool]:
        """Return (has_previous_data, prev_prob_map, has_overlap).

        has_previous_data: True if previous results exist at all.
        has_overlap: True if >=1 (university, major) pair exists in both old and
                     new results sets.
        prev_prob_map: {(uni, major): prob} dict for matching, or None.
        """
        if not prev_results:
            return False, None, False

        prob_map = DeltaCalculator.build_prob_map(prev_results)
        if not prob_map:
            return False, None, False

        current_unis = set(current_target_unis or [])
        current_majors = set(current_target_majors or [])

        # Bulk-match mode: user didn't specify targets → all prob_map entries
        # are implicitly "in scope", so overlap always exists.
        if not current_unis and not current_majors:
            has_overlap = True
        else:
            has_overlap = any(u in current_unis and m in current_majors for u, m in prob_map)

        return True, prob_map, has_overlap

    @staticmethod
    def calculate_delta(
        result: dict,
        prev_prob_map: dict[tuple[str, str], float],
    ) -> str:
        """Return display string for the delta column.

        "+3.2%" / "-1.5%" / "NEW" / "—" (unchanged, within 0.5pp).
        """
        uni = str(result.get("university", "")).strip()
        major = str(result.get("major", "")).strip()
        key = (uni, major)

        prev = prev_prob_map.get(key)
        if prev is None:
            return "NEW"

        curr = _safe_float(result.get("probability"), 0.0)
        diff = curr - prev
        if abs(diff) < 0.005:
            return "—"
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff * 100:.1f}%"
