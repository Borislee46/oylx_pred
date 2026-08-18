import json
from typing import Any

from pydantic_ai import RunContext

from src.agent.harness import HarnessDeps
from src.agent.tools.scope import evaluate_scope, normalize_region, supported_countries
from src.agent.tools.tool_trace import traced


@traced
def read_form(ctx: RunContext[HarnessDeps]) -> str:
    return json.dumps(ctx.deps.gateway.read(), ensure_ascii=False)


@traced
def get_form_options(ctx: RunContext[HarnessDeps], field: str) -> str:
    opts = ctx.deps.gateway.options(field)
    if not opts:
        return f"字段 {field} 暂无候选列表（可直接写入原值，系统会模糊匹配）。"
    return json.dumps(opts, ensure_ascii=False)


@traced
def write_form(
    ctx: RunContext[HarnessDeps],
    university: str = "",
    major: str = "",
    major_2: str = "",
    degree_type: str = "",
    gpa: float | None = None,
    gpa_scale: str = "",
    language_type: str = "",
    language_score: float | None = None,
    standardized_test_type: str = "",
    standardized_test_score: float | None = None,
    country: str = "",
    target_schools: list[str] | None = None,
    target_majors: list[str] | None = None,
    research: str = "",
    internship: str = "",
    paper: str = "",
    award: str = "",
) -> str:
    schools = list(target_schools or [])
    scope = evaluate_scope(country=country or "", schools=schools)
    write_country = country
    write_schools = schools
    if country:
        normalized = normalize_region(country)
        if normalized not in supported_countries():
            write_country = ""
    if scope["unsupported_schools"]:
        rejected = set(scope["unsupported_schools"])
        write_schools = [s for s in schools if str(s).strip() not in rejected]

    fields = {
        "university": university,
        "major": major,
        "major_2": major_2,
        "degree_type": degree_type,
        "gpa": gpa,
        "gpa_scale": gpa_scale,
        "language_type": language_type,
        "language_score": language_score,
        "standardized_test_type": standardized_test_type,
        "standardized_test_score": standardized_test_score,
        "country": write_country,
        "target_schools": write_schools,
        "target_majors": target_majors,
        "research": research,
        "internship": internship,
        "paper": paper,
        "award": award,
    }
    res = ctx.deps.gateway.write(fields)
    payload: dict[str, Any] = {
        "applied": res.get("applied", []),
        "missing": res.get("missing", []),
    }
    if not scope["in_scope"]:
        payload["scope_rejected"] = True
        payload["scope_issues"] = scope["issues"]
        payload["unsupported_schools"] = scope["unsupported_schools"]
    return json.dumps(payload, ensure_ascii=False)


@traced
def check_scope(
    ctx: RunContext[HarnessDeps],
    country: str = "",
    schools: list[str] | None = None,
    degree_level: str = "",
) -> str:
    return json.dumps(
        evaluate_scope(country=country, schools=schools or [], degree_level=degree_level),
        ensure_ascii=False,
    )


@traced
def submit_prediction(ctx: RunContext[HarnessDeps]) -> str:
    res = ctx.deps.gateway.submit()
    if res.get("ok"):
        return "已触发预测。"
    return "未触发：仍缺少 " + "、".join(res.get("missing", [])) + "。请补充或向用户追问后再触发。"


@traced
def expand_form(ctx: RunContext[HarnessDeps]) -> str:
    ctx.deps.gateway.expand()
    return "已展开表单供用户补充确认。"


@traced
def standardize_background_university(
    ctx: RunContext[HarnessDeps],
    university: str,
) -> str:
    from src.agent.bridges.fuzzy_match import _UNIVERSITY_ALIAS_MAP

    raw = university.strip()
    std, conf = _standardize_one(raw, _UNIVERSITY_ALIAS_MAP, ctx, field="university")
    return json.dumps({"raw": raw, "standardized": std, "confidence": conf}, ensure_ascii=False)


@traced
def standardize_background_major(
    ctx: RunContext[HarnessDeps],
    major: str,
) -> str:
    from src.agent.bridges.fuzzy_match import _BG_MAJOR_ALIAS_MAP

    raw = major.strip()
    std, conf = _standardize_one(raw, _BG_MAJOR_ALIAS_MAP, ctx, field="major")
    return json.dumps({"raw": raw, "standardized": std, "confidence": conf}, ensure_ascii=False)


@traced
def standardize_target_major(
    ctx: RunContext[HarnessDeps],
    major: str,
) -> str:
    from src.agent.bridges.fuzzy_match import _MAJOR_DISCIPLINE_MAP

    raw = major.strip()
    std, conf = _standardize_one(raw, _MAJOR_DISCIPLINE_MAP, ctx, field="target_major")
    return json.dumps({"raw": raw, "standardized": std, "confidence": conf}, ensure_ascii=False)


@traced
def standardize_fields(
    ctx: RunContext[HarnessDeps],
    university: str = "",
    major: str = "",
) -> str:
    from src.agent.bridges.fuzzy_match import _BG_MAJOR_ALIAS_MAP, _UNIVERSITY_ALIAS_MAP

    result: dict[str, dict] = {}
    if university and university.strip():
        raw = university.strip()
        std, conf = _standardize_one(raw, _UNIVERSITY_ALIAS_MAP, ctx, field="university")
        result["university"] = {"raw": raw, "standardized": std, "confidence": conf}
    if major and major.strip():
        raw = major.strip()
        std, conf = _standardize_one(raw, _BG_MAJOR_ALIAS_MAP, ctx, field="major")
        result["major"] = {"raw": raw, "standardized": std, "confidence": conf}
    return json.dumps(result, ensure_ascii=False)


def _standardize_one(
    raw: str,
    alias_map: dict[str, str],
    ctx: RunContext[HarnessDeps],
    *,
    field: str = "major",
) -> tuple[str, float]:
    from src.agent.bridges.fuzzy_match import _fuzzy_match

    query = raw
    if raw.isascii():
        probe = raw.casefold()
        if probe.startswith("the "):
            probe = probe[4:]
        for key, val in alias_map.items():
            if key.casefold() == probe:
                query = val
                break

    try:
        options_raw = ctx.deps.gateway.options(field)
    except Exception:
        options_raw = []

    if isinstance(options_raw, str):
        import json as _json

        try:
            options_raw = _json.loads(options_raw)
        except Exception:
            options_raw = []

    if not options_raw:
        return query, 0.0

    matched, conf = _fuzzy_match(query, options_raw)
    return matched, conf


FORM_TOOLS = [
    read_form,
    get_form_options,
    check_scope,
    write_form,
    submit_prediction,
    expand_form,
    standardize_background_university,
    standardize_background_major,
    standardize_target_major,
    standardize_fields,
]
