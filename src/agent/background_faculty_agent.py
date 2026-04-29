import hashlib
import json
import os
import threading
from functools import lru_cache
from typing import Any

import pandas as pd

from src.agent.background_faculty_prompts import build_background_faculty_prompt
from src.agent.base_agent import BaseAgent
from src.utils.app_data_loader import load_school_major_details_df

CACHE_DIR = "cache/agent_cache"
CACHE_FILE = "background_faculty_candidates.json"
PROMPT_VERSION = 2
_SCHOOL_MAJOR_DETAILS_PATH = "src/machine_learning_models/data/school_major_details.feather"


@lru_cache(maxsize=4)
def _get_valid_faculties_cached(path: str, mtime_key: int) -> list[str]:
    df = load_school_major_details_df(path=path)
    if df is None or df.empty or "专业大类" not in df.columns:
        return []
    series = df["专业大类"].dropna().astype(str).map(str.strip)
    values = [x for x in series.tolist() if x and x.lower() not in {"nan", "none"}]
    return sorted(set(values))


def _get_valid_faculties() -> list[str]:
    path = _SCHOOL_MAJOR_DETAILS_PATH
    mtime_key = int(os.path.getmtime(path)) if os.path.isfile(path) else 0
    return _get_valid_faculties_cached(path, mtime_key)


class BackgroundFacultyAgent(BaseAgent):
    # Class-level persistent cache — shared across instances with locking
    _persistent_cache: dict[str, Any] | None = None
    _cache_lock = threading.Lock()
    _cache_loaded = False

    @classmethod
    def _get_persistent_cache(cls) -> dict[str, Any]:
        if not cls._cache_loaded:
            with cls._cache_lock:
                if not cls._cache_loaded:
                    cls._persistent_cache = cls._load_cache_from_disk()
                    cls._cache_loaded = True
        return cls._persistent_cache or {}

    @classmethod
    def _write_to_cache(cls, key: str, value: Any) -> None:
        with cls._cache_lock:
            cache = cls._get_persistent_cache()
            cache[key] = value

    @classmethod
    def _flush_persistent_cache(cls) -> None:
        with cls._cache_lock:
            if cls._persistent_cache is not None:
                cls._save_cache_to_disk(cls._persistent_cache)

    @classmethod
    def _load_cache_from_disk(cls) -> dict[str, Any]:
        file_path = os.path.join(CACHE_DIR, CACHE_FILE)
        if not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @classmethod
    def _save_cache_to_disk(cls, data: dict[str, Any]) -> None:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            file_path = os.path.join(CACHE_DIR, CACHE_FILE)
            tmp_path = f"{file_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, file_path)
        except OSError:
            pass

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=10, agent_name="交叉背景学院Agent")

    def _cache_key(self, background_major_original: str, base_faculty: str | None) -> str:
        key_data = {
            "v": PROMPT_VERSION,
            "background_major_original": str(background_major_original or "").strip(),
            "base_faculty": str(base_faculty or "").strip(),
            "model": self.model,
        }
        payload = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

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
            cached = BackgroundFacultyAgent._get_persistent_cache().get(cache_key)
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
        raw = self._call_api(prompt, cache_prefix="bg_faculty", use_cache=True)
        if not raw:
            if base:
                return [base]
            return []

        result = self._parse_json_response(
            raw,
            schema_hint='{"extra_faculties": [string, ...]}',
            cache_prefix="bg_faculty_json_repair",
            max_tokens=256,
        )

        if result is None:
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
            BackgroundFacultyAgent._write_to_cache(cache_key, out)
            try:
                BackgroundFacultyAgent._flush_persistent_cache()
            except OSError:
                pass

        return out[:max_total]

    def run(self, context: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Orchestrator-compatible entry point.

        Reads background_major from StudentContext or kwargs.
        Writes resolved faculties back to context.
        """
        from src.agent.context import StudentContext

        bg_major = kwargs.get("background_major_original", "")
        if not bg_major and isinstance(context, StudentContext):
            bg_major = context.background_major or context.extracted_background.get("major", "")

        base_faculty = kwargs.get("base_faculty") or (
            context.background_faculty if isinstance(context, StudentContext) else None
        )

        faculties = self.resolve_background_faculties(
            background_major_original=str(bg_major),
            base_faculty=base_faculty,
            max_total=int(kwargs.get("max_total", 3)),
            max_extra=int(kwargs.get("max_extra", 2)),
            use_persistent_cache=bool(kwargs.get("use_persistent_cache", True)),
        )

        result: dict[str, Any] = {
            "faculties": faculties,
            "primary": faculties[0] if faculties else None,
        }
        if bg_major and not faculties:
            result["_error"] = "no_faculties_resolved"
        return result


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
