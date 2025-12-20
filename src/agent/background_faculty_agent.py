from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from typing import Any

import pandas as pd

from src.agent.background_faculty_prompts import build_background_faculty_prompt
from src.agent.base_agent import BaseAgent
from src.utils.app_data_loader import load_school_major_details_df

CACHE_DIR = "cache/agent_cache"
CACHE_FILE = "background_faculty_candidates.json"
PROMPT_VERSION = 2


@lru_cache(maxsize=1)
def _get_valid_faculties() -> list[str]:
    df = load_school_major_details_df()
    if df is None or df.empty or "专业大类" not in df.columns:
        return []
    series = df["专业大类"].dropna().astype(str).map(str.strip)
    values = [x for x in series.tolist() if x and x.lower() not in {"nan", "none"}]
    return sorted(set(values))


class BackgroundFacultyAgent(BaseAgent):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=10, agent_name="交叉背景学院Agent")
        self._persistent_cache = self._load_persistent_json(CACHE_DIR, CACHE_FILE)

    def _cache_key(self, background_major_original: str, base_faculty: str | None) -> str:
        key_data = {
            "v": PROMPT_VERSION,
            "background_major_original": str(background_major_original or "").strip(),
            "base_faculty": str(base_faculty or "").strip(),
            "model": self.model,
        }
        payload = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _flush_persistent_cache(self) -> None:
        self._save_persistent_json(CACHE_DIR, CACHE_FILE, self._persistent_cache)

    def resolve_background_faculties(
        self,
        background_major_original: str,
        base_faculty: str | None = None,
        max_total: int = 3,
        max_extra: int = 2,
        use_persistent_cache: bool = True,
    ) -> list[str]:
        major = str(background_major_original or "").strip()
        if not major:
            return []

        base = str(base_faculty or "").strip() or None
        max_total = int(max_total) if max_total is not None else 3
        max_extra = int(max_extra) if max_extra is not None else 2
        if max_total <= 0:
            return []
        if max_extra < 0:
            max_extra = 0

        valid_faculties = _get_valid_faculties()
        if not valid_faculties:
            if base:
                return [base]
            return []

        cache_key = self._cache_key(major, base)
        valid_set = set(valid_faculties)
        if use_persistent_cache:
            cached = self._persistent_cache.get(cache_key)
            if isinstance(cached, list):
                cached_out: list[str] = []
                for x in cached:
                    s = str(x).strip()
                    if not s or s == base:
                        continue
                    if s in valid_set and s not in cached_out:
                        cached_out.append(s)
                    if len(cached_out) >= max_total:
                        break
                if base and base in valid_set:
                    cached_out = [base] + [x for x in cached_out if x != base]
                if cached_out:
                    return cached_out[:max_total]

        prompt = build_background_faculty_prompt(
            background_major_original=major,
            base_faculty=base,
            valid_faculties=valid_faculties,
            max_extra=max_extra,
        )
        content = self._call_api(prompt, cache_prefix="bg_faculty", use_cache=True)
        if not content:
            if base:
                return [base]
            return []

        content = self._clean_json_content(content)
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            if base:
                return [base]
            return []

        if not isinstance(result, dict):
            if base:
                return [base]
            return []

        extras = result.get("extra_faculties", [])
        if not isinstance(extras, list):
            extras = []

        out: list[str] = []
        if base and base in valid_set:
            out.append(base)

        for x in extras:
            s = str(x).strip()
            if not s or s == base:
                continue
            if s in valid_set and s not in out:
                out.append(s)
            if len(out) >= max_total:
                break

        if not out and base:
            out = [base]

        if use_persistent_cache:
            self._persistent_cache[cache_key] = out
            try:
                self._flush_persistent_cache()
            except OSError:
                pass

        return out[:max_total]


def get_background_faculty_from_cases_df(
    background_major_agg: str | None, cases_df: pd.DataFrame | None
) -> str | None:
    if not background_major_agg or cases_df is None or cases_df.empty:
        return None
    if "background_major" not in cases_df.columns or "faculty" not in cases_df.columns:
        return None
    major_match = cases_df[cases_df["background_major"] == background_major_agg]
    if major_match.empty:
        return None
    s = major_match["faculty"].dropna().astype(str).map(str.strip)
    s = s[(s != "") & (~s.str.lower().isin({"nan", "none"}))]
    if s.empty:
        return None
    return s.value_counts().idxmax()
