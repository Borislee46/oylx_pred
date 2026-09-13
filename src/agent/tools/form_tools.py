import json
from typing import Any

from pydantic_ai import RunContext

from src.agent.harness import HarnessDeps
from src.agent.tools.scope import evaluate_scope, normalize_region, supported_countries
from src.agent.tools.tool_trace import traced


@traced
def read_form(ctx: RunContext[HarnessDeps]) -> str:
    """读取当前表单已有哪些字段、还缺哪些必填项。

    开始处理一条新输入前先调用它，避免重复写入、避免把表单里已有的字段当成缺失。
    首轮表单为空时也可以调用，返回 missing 全量。
    """
    return json.dumps(ctx.deps.gateway.read(), ensure_ascii=False)


@traced
def get_form_options(ctx: RunContext[HarnessDeps], field: str) -> str:
    """查某个表单字段的候选名列表（用于把用户原话对齐到系统标准名）。

    只在"拿不准该写哪个标准名"时调用。能直接写原值就不要调。

    Args:
        field: 只能是 "university"（本科院校，中文）/ "major"（本科专业，中文）/
            "target_major"（目标专业，英文项目名）之一。
            像 country、gpa 这类没有候选列表的字段不要传。
    """
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
    research_count: int | None = None,
    internship: str = "",
    internship_count: int | None = None,
    paper: str = "",
    paper_count: int | None = None,
    award: str = "",
    award_count: int | None = None,
) -> str:
    """把本轮新识别到的学生背景写入表单（会改动用户可见的表单，只在确有把握时调用）。

    只传本轮新提取到的字段；已有字段无需重复传（系统自动合并保留）。
    但若表单里的旧值与当前输入明显不是同一个人，以当前输入为准覆盖。

    Args:
        university: 本科院校。中文名或简称（"中山大学""中大"）都可以，系统落地时自动对齐；
            层次别名（985/211/双一流/双非/一本）直接原样写入，不要展开成具体学校。
        major: **本科专业，中文原话**（如"金融""计算机""工商管理"）。不要翻译成英文。
        major_2: 第二专业/辅修专业，中文。仅当用户明确提到双学位或辅修时填。
        degree_type: 第二学位类型，只能是"辅修"或"双学位"。仅当填了 major_2 时才填。
        gpa: 绩点或均分的数值。85 分写 85，3.5 写 3.5（按 gpa_scale 的含义给原始值）。
        gpa_scale: gpa 的满分制，只能是 "4.0" / "4.3" / "5.0" / "10" / "100" 之一。
            85 分 → "100"；3.5/4.0 → "4.0"；3.3/4.3 → "4.3"；3.5/5.0 → "5.0"。
        language_type: 语言考试类型，只能是"雅思"或"托福"。
        language_score: 语言成绩数值（雅思 7.0 写 7.0，托福 100 写 100，不要自行换算）。
        standardized_test_type: 标化考试类型，只能是 "GRE" 或 "GMAT"。
        standardized_test_score: 标化成绩数值。
        country: 目标国家/地区，取产品支持的地区之一（见 list_supported_targets）。越界地区不要写入。
        target_schools: 目标院校列表，中文校名。可写常见简称或档位别名（港3/港5/港8），
            系统落地时会自动展开。越界院校（MIT/哈佛/LSE 等）静默丢弃，不要写进来，
            也不要在回复里特意提及。
        target_majors: 目标专业列表，**用英文项目名**（如 "Master of Finance"）；
            不确定时写用户的中文原话也可以，系统会自动对齐。
            **第 1 轮默认宽召回，不要填**；第 2 轮用户明确聚焦某专业时才填。
        research: 科研经历原文（照抄用户描述，不要改写、不要编造）。
        research_count: 科研项目数量（整数）。能从文本推断就填，否则留空。
        internship: 实习经历原文（照抄用户描述）。
        internship_count: 实习段数（整数）。"三段实习" → 3；"在字节和腾讯实习" → 2；推断不出就留空。
        paper: 论文情况原文（"一篇待发论文" → paper="待发论文", paper_count=1）。
        paper_count: 论文篇数（整数）。推断不出就留空。
        award: 获奖情况原文（照抄用户描述）。
        award_count: 获奖项数（整数）。推断不出就留空。
    """
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
        "research_count": research_count,
        "internship": internship,
        "internship_count": internship_count,
        "paper": paper,
        "paper_count": paper_count,
        "award": award,
        "award_count": award_count,
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
    """核验目标地区/院校是否在产品支持范围内。

    ⚠️ `write_form` 已内置同样的范围校验（越界院校会被自动丢弃并回报），
    所以不要为了"保险"额外调用本工具。只有用户**直接问**"能不能申 X 地区/某校"时才单独调用。

    Args:
        country: 目标国家/地区名。
        schools: 目标院校列表（中文校名）。
        degree_level: 学位层次（如"硕士"）；一般留空。
    """
    return json.dumps(
        evaluate_scope(country=country, schools=schools or [], degree_level=degree_level),
        ensure_ascii=False,
    )


@traced
def submit_prediction(ctx: RunContext[HarnessDeps]) -> str:
    """触发录取方案预测（会生成用户可见的结果）。

    四核心齐全（院校 + 专业 + GPA + 语言成绩）时才调用；缺任一项会被拒绝并返回缺失清单，
    此时应改用 expand_form 让用户补充，而不是反复重试本工具。
    """
    res = ctx.deps.gateway.submit()
    if res.get("ok"):
        return "已触发预测。"
    return "未触发：仍缺少 " + "、".join(res.get("missing", [])) + "。请补充或向用户追问后再触发。"


@traced
def expand_form(ctx: RunContext[HarnessDeps]) -> str:
    """展开表单，让用户核对已写入的字段或补充缺失项。

    需要用户补信息、或写入后希望用户确认时调用。不需要用户操作时不要调用。
    """
    ctx.deps.gateway.expand()
    return "已展开表单供用户补充确认。"


@traced
def start_new_profile(ctx: RunContext[HarnessDeps]) -> str:
    """把当前会话切换成「另一个学生」：清空表单里上一位学生的背景与目标。

    **必须调用**的情况：
    - 用户描述的本科院校/专业与当前表单快照里的**不是同一个人**
      （换了个学生、帮朋友问、又来了一位咨询者）；
    - 用户明确说「换一个学生」「重新开始」「不是刚才那个人」。

    **不要调用**的情况：
    - 用户只是在补充/修改**同一个人**的信息（缺语言成绩、想加一所目标校等）；
    - 用户只是在提问或闲聊。

    调用后本轮新识别到的字段仍然有效：先清掉旧档案，再写入新的。
    """
    work_ctx = getattr(ctx.deps, "ctx", None)
    if work_ctx is not None:
        work_ctx.extracted_background = {}
    gateway = getattr(ctx.deps, "gateway", None)
    if gateway is not None and hasattr(gateway, "_reset_requested"):
        gateway._reset_requested = True
    return json.dumps(
        {"ok": True, "note": "已清空上一位学生的档案，请继续写入本轮识别到的字段。"},
        ensure_ascii=False,
    )


@traced
def standardize_background_university(
    ctx: RunContext[HarnessDeps],
    university: str,
) -> str:
    """把本科院校名对齐到系统标准名（模糊匹配）。

    只在拿不准该写哪个标准校名时使用；拿不准也可以直接调用 write_form 写原值，
    系统落地时会自动对齐。

    Args:
        university: 用户给出的本科院校名或简称。
    """
    from src.agent.bridges.fuzzy_match import _UNIVERSITY_ALIAS_MAP

    raw = university.strip()
    std, conf = _standardize_one(raw, _UNIVERSITY_ALIAS_MAP, ctx, field="university")
    return json.dumps({"raw": raw, "standardized": std, "confidence": conf}, ensure_ascii=False)


@traced
def standardize_background_major(
    ctx: RunContext[HarnessDeps],
    major: str,
) -> str:
    """把**本科专业**（中文）对齐到系统标准名（模糊匹配）。

    一般不需要调用——write_form 里的 major 可以直接写中文原话。

    Args:
        major: 本科专业中文名（如"金融""计科"）。
    """
    from src.agent.bridges.fuzzy_match import _BG_MAJOR_ALIAS_MAP

    raw = major.strip()
    std, conf = _standardize_one(raw, _BG_MAJOR_ALIAS_MAP, ctx, field="major")
    return json.dumps({"raw": raw, "standardized": std, "confidence": conf}, ensure_ascii=False)


@traced
def standardize_target_major(
    ctx: RunContext[HarnessDeps],
    major: str,
) -> str:
    """把**目标专业**对齐到系统里的英文项目名（模糊匹配）。

    查项目要求请用 lookup_program；本工具只用于"把目标专业对齐成英文项目名"。

    Args:
        major: 目标专业，中文或英文都可以（如"数据科学"或 "Master of Finance"）。
    """
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
    """一次性对齐本科院校与本科专业（模糊匹配）。

    一般不需要调用——write_form 可以接受原值并自动对齐。

    Args:
        university: 本科院校名或简称。
        major: 本科专业中文名。
    """
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
    start_new_profile,
    standardize_background_university,
    standardize_background_major,
    standardize_target_major,
    standardize_fields,
]
