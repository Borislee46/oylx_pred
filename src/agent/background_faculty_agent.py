import logging
import os
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel

from src.agent.base_agent import BaseAgent
from src.agent.persistent_cache import PersistentCache
from src.agent.prompts.background_faculty import build_background_faculty_prompt


class FacultyResult(BaseModel):
    extra_faculties: list[str] = Field(default_factory=list)


_FACULTY_SYSTEM_PROMPT = """你是研究生申请背景识别助手。
目标：判断一个"本科背景专业（原始名称）"可能对应哪些"专业大类/学院"。

要求：
- extra_faculties 只包含"在 base_faculty 之外可能还相关"的学院
- 只有在"明显跨学科/联合/双学位/多方向"或"专业与 base_faculty 明显不匹配/高度不确定"时，才输出 extra_faculties；否则必须输出空数组
- **严谨性**：如果无法判断或关联度极弱，请返回空数组。宁可少报，不可错报
- 只能从给定的 valid_faculties 里选，不要输出其它词
- 如果你认为不存在额外学院，输出空数组
- 不要解释，不要输出多余字段"""


def _build_background_faculty_agent(model: OpenAIChatModel) -> Agent:
    return Agent(
        model,
        max_concurrency=1,
        output_type=PromptedOutput(FacultyResult),
        instructions=_FACULTY_SYSTEM_PROMPT,
        retries=2,
    )


from src.pages.prediction.app_data import load_school_major_details_df

_log = logging.getLogger(__name__)

PROMPT_VERSION = 3


def _get_school_major_details_path() -> str:
    from pathlib import Path

    start = Path(__file__).resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return str(p / "src" / "ml" / "data" / "school_major_details.feather")
    return "src/ml/data/school_major_details.feather"


@lru_cache(maxsize=4)
def _get_valid_faculties_cached(path: str, mtime_key: int) -> list[str]:
    df = load_school_major_details_df(path=path)
    if df is None or df.empty or "专业大类" not in df.columns:
        _log.warning(
            "无法加载valid_faculties: df=%s columns=%s",
            type(df).__name__ if df is not None else None,
            df.columns.tolist() if df is not None else [],
        )
        return []
    series = df["专业大类"].dropna().astype(str).map(str.strip)
    values = [x for x in series.tolist() if x and x.lower() not in {"nan", "none"}]
    return sorted(set(values))


def _get_valid_faculties() -> list[str]:
    path = _get_school_major_details_path()
    mtime_key = int(os.path.getmtime(path)) if os.path.isfile(path) else 0
    return _get_valid_faculties_cached(path, mtime_key)


class BackgroundFacultyAgent(BaseAgent):
    _cache = PersistentCache("background_faculty_candidates.json")

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config, timeout=10, agent_name="交叉背景学院Agent", logger=_log)
        self._agent = _build_background_faculty_agent(self._model)

    def _cache_key(self, background_major_original: str, base_faculty: str | None) -> str:
        return self._hash_cache_key(
            PROMPT_VERSION,
            self.model,
            background_major_original=str(background_major_original or "").strip(),
            base_faculty=str(base_faculty or "").strip(),
        )

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
            _log.debug("resolve_background_faculties跳过(无专业名)")
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
            _log.warning("valid_faculties为空，无法解析学院")
            if base:
                return [base]
            return []

        _log.info(
            "学院解析开始 | major=%s base=%s valid_faculties_count=%d",
            major,
            base,
            len(valid_faculties),
        )

        cache_key = self._cache_key(major, base)
        valid_set = set(valid_faculties)
        if use_persistent_cache:
            cached = BackgroundFacultyAgent._cache.get(cache_key)
            if isinstance(cached, list):
                _log.debug("学院解析缓存命中 | major=%s cached=%s", major, cached)
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
                    _log.info("学院解析完成(缓存) | major=%s result=%s", major, cached_out)
                    return cached_out[:max_total]

        prompt = build_background_faculty_prompt(
            background_major_original=major,
            base_faculty=base,
            valid_faculties=valid_faculties,
            max_extra=max_extra,
        )
        _log.debug("学院解析API调用 | major=%s prompt_len=%d", major, len(prompt))

        try:
            result = self._agent.run_sync(prompt)
            output = result.output
            if not isinstance(output, FacultyResult):
                _log.warning("学院解析返回非预期类型: %s", type(output))
                if base:
                    return [base]
                return []
            extras = output.extra_faculties
        except Exception:
            _log.warning("学院解析API调用失败 | major=%s", major, exc_info=True)
            if base:
                return [base]
            return []

        out: list[str] = []
        filtered_count = 0
        if base and base in valid_set:
            out.append(base)

        for x in extras:
            s = str(x).strip()
            if not s or s == base:
                continue
            if s in valid_set and s not in out:
                out.append(s)
            else:
                filtered_count += 1
            if len(out) >= max_total:
                break

        if filtered_count > 0:
            _log.debug("学院解析过滤无效结果 | filtered=%d", filtered_count)

        if not out and base:
            out = [base]

        if use_persistent_cache:
            BackgroundFacultyAgent._cache.set(cache_key, out)

        _log.info(
            "学院解析完成 | major=%s result=%s (%d alternatives)",
            major,
            out,
            len(out) - (1 if base else 0),
        )
        return out[:max_total]

    def run(self, context: Any = None, **kwargs: Any) -> dict[str, Any]:
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
