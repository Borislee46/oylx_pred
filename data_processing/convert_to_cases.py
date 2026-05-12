"""
将指南者(compassedu) / ApplySquare 爬虫数据转换为 cases.feather 格式。

用法:
  py convert_to_cases.py --source compassedu --input cases_newst_result.xlsx
  py convert_to_cases.py --source applysquare --input applysquare案例总汇.xlsx
  py convert_to_cases.py --source compassedu --input cases.xlsx --output custom.feather

依赖: pip install pandas pyarrow openpyxl openai rapidfuzz
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

# ── LLM 配置 ──
LLM_CONFIG = {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "max_tokens": 1200,  # reasoning model 需要较多 token 给推理+输出
    "temperature": 0.0,
}

# ── 目标学校列表（与 cases.feather 一致） ──
TARGET_SCHOOLS = {
    "香港大学",
    "香港中文大学",
    "香港科技大学",
    "香港理工大学",
    "香港城市大学",
    "香港浸会大学",
    "香港教育大学",
    "香港岭南大学",
    "香港都会大学",
    "香港珠海学院",
    "香港恒生大学",
    "香港中文大学 (深圳校区)",
    "香港中文大学（深圳校区）",
    "新加坡国立大学",
    "新加坡南洋理工大学",
    "新加坡管理大学",
    "澳门大学",
    "澳门科技大学",
    "澳门理工大学",
    "澳门城市大学",
    "马来亚大学",
    "马来西亚博特拉大学",
    "马来西亚国立大学",
    "马来西亚理科大学",
}

# 学校名简称/别名规范化
SCHOOL_NAME_MAP = {
    "南洋理工大学": "新加坡南洋理工大学",
    "香港中文大学（深圳）": "香港中文大学 (深圳校区)",
    "香港中文大学(深圳)": "香港中文大学 (深圳校区)",
    "香港中文大学 (深圳)": "香港中文大学 (深圳校区)",
}

# ── 专业大类 → faculty 映射 ──
FACULTY_SET = {
    "商学院",
    "工程学院",
    "文学院",
    "计算机学院",
    "理学院",
    "社会科学院",
    "法学院",
    "艺术学院",
    "医学院",
    "教育学院",
    "建筑学院",
    "设计学院",
}

# ── 基本背景正则 ──
RE_GPA = re.compile(r"GPA\s*(\d+\.?\d*)", re.IGNORECASE)
RE_GPA_SCORE = re.compile(r"均分\s*(\d+\.?\d*)")
RE_IELTS = re.compile(r"雅思\s*(\d+\.?\d*)")
RE_TOEFL = re.compile(r"托福\s*(\d+\.?\d*)")

# ── ApplySquare 录取结果 → admitted 映射 ──
APPLYSQUARE_ADMIT_MAP = {
    "OFFER": 1,
    "录取": 1,
    "被拒": 0,
    "等待": None,
    "撤回": None,
    "面试": None,
    "其他": None,
}

APPLYSQUARE_EXPERIENCE_COL = "学生心得分享"

# ── ApplySquare 本科专业回填正则 ──
RE_BG_MAJOR = re.compile(r"(?:BG|bg)[：:\s]*[^\s，。,\.]{0,10}([^\s，。,\.]{2,10})")
RE_UNDERGRAD_MAJOR = re.compile(r"(?:本科专业|本科|就读专业)[：:\s]*([^\s，。,\.]{2,10})")
_BACKFILL_BLOCKLIST = [
    "学费",
    "以下",
    "以上",
    "要求",
    "写好",
    "申请",
    "提交",
    "准备",
    "相关",
    "领域",
    "课设",
]

# ── ApplySquare 毕业学校回填正则 ──
RE_DIRECT_UNIVERSITY = re.compile(
    r"(?:就读于|毕业于|本科学校|本科院校|学校[：:])[^\n]{0,30}?([^\s，。,\.\n]{2,20}(?:大学|学院))"
)
RE_BG_UNIVERSITY = re.compile(r"(?:BG|bg)[：:\s]*([^\s，。,\.]{2,15}(?:大学|学院))")
RE_SCHOOL_TIER = re.compile(r"(双非|211|985|C9|一本|二本|末流985|末流211)")

# ── 经历分类关键词（规则优先，覆盖率预计>80%） ──
EXP_KEYWORDS = {
    "internship": ["实习", "intern", "公司", "工作", "入职", "在职"],
    "research": [
        "项目",
        "课题",
        "论文",
        "研究",
        "实验室",
        "实验",
        "调研",
        "科研",
        "学术",
        "毕设",
        "毕业设计",
    ],
    "paper": [
        "发表",
        "专利",
        "Accepted",
        "published",
        "期刊",
        "会议论文",
        "SCI",
        "EI",
        "核心期刊",
        "一作",
        "共同作者",
    ],
    "award": [
        "奖学金",
        "竞赛",
        "获奖",
        "一等奖",
        "二等奖",
        "三等奖",
        "金奖",
        "银奖",
        "铜奖",
        "优秀奖",
        "荣誉称号",
        "三好学生",
        "优秀学生",
        "国家奖学金",
        "国奖",
        "校奖",
    ],
    "activity": [
        "社团",
        "志愿者",
        "学生会",
        "支教",
        "义工",
        "公益",
        "组织",
        "举办",
        "策划",
        "主持",
    ],
}

# ── 路径推导 ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "src" / "machine_learning_models" / "data"
_OUTPUT_DIR = _DATA_DIR / "external"
_CACHE_DIR = _SCRIPT_DIR / ".llm_cache"

# ── 磁盘缓存（避免重复调用LLM） ──
import pickle


def _cache_key(prefix, text):
    return hashlib.md5(f"{prefix}:{text}".encode()).hexdigest()


def _cache_get(prefix, text):
    key = _cache_key(prefix, text)
    fpath = _CACHE_DIR / f"{prefix}" / f"{key}.pkl"
    if fpath.exists():
        try:
            with open(fpath, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def _cache_set(prefix, text, data):
    key = _cache_key(prefix, text)
    d = _CACHE_DIR / f"{prefix}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{key}.pkl", "wb") as f:
        pickle.dump(data, f)


def _resolve_data_path(filename):
    """优先使用项目 data 目录，fallback 到旧路径。"""
    p = _DATA_DIR / filename
    if p.exists():
        return str(p)
    old = (
        Path(
            "C:/Users/lijia/Desktop/0927/oylx_hk_pred_test-master/src/machine_learning_models/data"
        )
        / filename
    )
    if old.exists():
        return str(old)
    return str(p)


def get_api_key():
    """从环境变量或项目配置获取 DeepSeek API key"""
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    config_path = _PROJECT_ROOT / "config" / "app_config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            env_name = os.environ.get("APP_ENV", "test")
            env_cfg = cfg.get(env_name, cfg.get("test", {}))
            key = env_cfg.get("OPEN_AI_API_KEY", "")
            if key:
                return key
        except Exception:
            pass

    print("ERROR: 未找到 API key。请设置 DEEPSEEK_API_KEY 环境变量。")
    sys.exit(1)


def parse_basic_background(text):
    """从基本背景文本中提取 GPA、语言成绩等数值"""
    if not isinstance(text, str):
        return {}

    result = {}
    m = RE_GPA.search(text)
    if m:
        gpa_val = float(m.group(1))
        if gpa_val > 10:
            gpa_val = round(gpa_val / 25, 2)
            gpa_val = min(gpa_val, 4.0)
        result["gpa"] = gpa_val
    else:
        m = RE_GPA_SCORE.search(text)
        if m:
            score = float(m.group(1))
            result["gpa"] = min(round(score / 25, 2), 4.0)

    m = RE_IELTS.search(text)
    if m:
        result["ielts"] = float(m.group(1))

    m = RE_TOEFL.search(text)
    if m:
        result["toefl"] = float(m.group(1))

    return result


# ═══════════════════════════════════════════════════════
# Fuzzy 学校名 + 专业名 匹配
# ═══════════════════════════════════════════════════════

_school_matcher = None  # lazy init: (choices_list, scorer)
_school_choices_en = None


def _init_school_matcher():
    """懒加载 rapidfuzz school name matcher（基于 school_base）"""
    global _school_matcher, _school_choices_en
    if _school_matcher is not None:
        return

    from rapidfuzz import fuzz, process

    path = _resolve_data_path("school_base.feather")
    if not os.path.exists(path):
        _school_matcher = False
        return

    sdf = pd.read_feather(path)
    cn_names = sdf["学校名称"].dropna().tolist()
    en_names = sdf["英文名称"].dropna().tolist()

    # 合并中英文候选 + TARGET_SCHOOLS
    all_choices = set(cn_names + en_names)
    all_choices |= TARGET_SCHOOLS

    _school_choices_en = en_names
    _school_matcher = (sorted(all_choices), process, fuzz)


def normalize_school(name):
    """规范化学校名：精确匹配 → fuzzy(threshold 85) → 别名映射"""
    if not isinstance(name, str):
        return None
    name = name.strip()

    # 精确匹配
    if name in TARGET_SCHOOLS:
        return name
    if name in SCHOOL_NAME_MAP:
        return SCHOOL_NAME_MAP[name]

    # fuzzy 匹配
    _init_school_matcher()
    if _school_matcher is False:
        return None

    choices, process_mod, fuzz_mod = _school_matcher
    result = process_mod.extractOne(name, choices, scorer=fuzz_mod.ratio, score_cutoff=80)
    if result:
        matched = result[0]
        if matched in TARGET_SCHOOLS:
            return matched
        # 再查一次精确映射
        if matched in SCHOOL_NAME_MAP:
            return SCHOOL_NAME_MAP[matched]
    return None


def fuzzy_match_major(chinese_name, school_major_df):
    """Fuzzy 匹配中文录取专业名 → 英文名 + 专业大类（faculty）
    匹配对象是 school_major_details.专业中文名称（研究生项目名）。"""
    if not isinstance(chinese_name, str) or not chinese_name.strip():
        return None, None

    from rapidfuzz import fuzz, process

    if school_major_df is None or "专业中文名称" not in school_major_df.columns:
        return None, None

    lookup = school_major_df.dropna(subset=["专业中文名称"])
    if lookup.empty:
        return None, None

    choices = lookup["专业中文名称"].tolist()
    result = process.extractOne(chinese_name.strip(), choices, scorer=fuzz.ratio, score_cutoff=70)
    if result:
        match_row = lookup[lookup["专业中文名称"] == result[0]].iloc[0]
        en_name = match_row.get("专业英文名称_聚合", match_row.get("专业英文名称", None))
        faculty = match_row.get("专业大类", None)
        if faculty and faculty not in FACULTY_SET:
            faculty = None
        return en_name, faculty
    return None, None


def fuzzy_match_background_major(major_text, background_majors_list, raw_to_std=None):
    """6层匹配本科专业 → 94 standard background_major + faculty"""
    if not isinstance(major_text, str) or not major_text.strip():
        return None, None

    from rapidfuzz import fuzz, process

    mt = major_text.strip()
    mt_norm = _normalize_major_text(mt)

    # Layer 1: 精确匹配 raw→standard 查找表 (最高优先级)
    if raw_to_std and mt_norm in raw_to_std:
        bm, fac = raw_to_std[mt_norm]
        if fac is None:
            fac = _infer_faculty_from_major(bm, original_text=mt)
        return bm, fac

    # Layer 2: 精确匹配 94 标准类别名
    for bm in background_majors_list:
        if bm == mt or bm == mt_norm:
            return bm, _infer_faculty_from_major(bm, original_text=mt)

    # Layer 3: 子串匹配 (加长度 guard, min 3 chars)
    for bm in background_majors_list:
        if len(bm) >= 3:
            if bm in mt or mt in bm:
                shorter = min(len(bm), len(mt))
                longer = max(len(bm), len(mt))
                if shorter / longer >= 0.35:
                    return bm, _infer_faculty_from_major(bm, original_text=mt)

    # Layer 4: Fuzzy 匹配 raw→standard 查找表 keys (score_cutoff=80)
    if raw_to_std and raw_to_std:
        raw_keys = list(raw_to_std.keys())
        result = process.extractOne(mt_norm, raw_keys, scorer=fuzz.ratio, score_cutoff=80)
        if result:
            bm, fac = raw_to_std[result[0]]
            if fac is None:
                fac = _infer_faculty_from_major(bm, original_text=mt)
            return bm, fac

    # Layer 5: Fuzzy 匹配 94 标准类别 (score_cutoff=70, 比原来75放宽)
    result = process.extractOne(mt_norm, background_majors_list, scorer=fuzz.ratio, score_cutoff=70)
    if result:
        bm = result[0]
        return bm, _infer_faculty_from_major(bm, original_text=mt)

    return None, None


# ── 本科专业 → 学院推断规则 ──
# 顺序重要：更具体的模式在前，避免被宽泛规则吞噬
_MAJOR_FACULTY_RULES = [
    # 建筑学院 / 设计学院 (之前完全缺失)
    (re.compile(r"建筑|城乡规划|风景园林|城市规划|环境设计|建筑学"), "建筑学院"),
    (
        re.compile(
            r"设计(?!学)|工业设计|产品设计|视觉传达|服装设计|数字媒体艺术|数媒艺术|艺术设计"
        ),
        "设计学院",
    ),
    # 计算机学院
    (
        re.compile(
            r"计算机|软件工程|人工智能|AI|网络工程|物联网|数据科学|大数据|通信工程|自动化|电气|信息工程|数字媒体技术|信息安全"
        ),
        "计算机学院",
    ),
    # 商学院
    (
        re.compile(
            r"工商管理|会计|金融|经济|市场|国贸|商务|财务|管理科学|物流|供应链|酒店|旅游管理|人力资源|保险|精算|工业工程|电子商务|审计|税务"
        ),
        "商学院",
    ),
    # 工程学院
    (
        re.compile(
            r"土木|机械|材料|化工|化学工程|环境工程|能源动力|能源|测绘|交通|水利|安全工程|食品|航空航天|包装|冶金|纺织|船舶|生物工程|生物医学工程|车辆工程|测控"
        ),
        "工程学院",
    ),
    # 理学院
    (
        re.compile(
            r"数学|物理(?!治疗)|化学(?!工程)|生物(?!医学工程|工程)|统计|地理|地质|海洋|大气|天文|生态|园艺|水产|心理学|应用物理|应用化学"
        ),
        "理学院",
    ),
    # 法学院
    (re.compile(r"法学|法律|知识产权"), "法学院"),
    # 医学院
    (
        re.compile(r"医学(?!工程|技术)|临床|药学|护理|口腔|中医|公共卫生|兽医|康复|营养|医学技术"),
        "医学院",
    ),
    # 文学院
    (
        re.compile(
            r"文学|历史|哲学|语言学|英语|日语|法语|德语|翻译|新闻|传播|广告|汉语言|公共关系|文化产业|对外汉语"
        ),
        "文学院",
    ),
    # 社会科学院
    (
        re.compile(r"社会学|政治|公共管理|国际关系|人类学|社会工作|行政管理|社会保障|宗教学"),
        "社会科学院",
    ),
    # 艺术学院 (放后面，避免吞噬设计类和数字媒体)
    (
        re.compile(r"艺术(?!设计)|美术学|音乐|戏剧|影视|电影|动画|舞蹈|播音|编导|书法|摄影"),
        "艺术学院",
    ),
    # 教育学院
    (re.compile(r"教育|师范|学前|特殊教育|体育|运动训练|社会体育"), "教育学院"),
]


def _infer_faculty_from_major(major_text, original_text=None):
    """从本科专业名推断学院分类。优先对original_text做规则匹配，fallback到major_text"""
    search_texts = [original_text, major_text] if original_text else [major_text]
    for text in search_texts:
        if not isinstance(text, str):
            continue
        for pattern, faculty in _MAJOR_FACULTY_RULES:
            if pattern.search(text):
                return faculty
    return None


# ═══════════════════════════════════════════════════════
# 经历分类：规则优先 + LLM 兜底
# ═══════════════════════════════════════════════════════


def _classify_experience_item(item_text):
    """对单条经历文本做关键词分类，返回 category 或 None"""
    text = item_text.strip()
    if not text:
        return None

    # 按优先级检查（paper > internship > research，paper优先因为有"发表论文"交集）
    for cat in ["paper", "internship", "research"]:
        for kw in EXP_KEYWORDS[cat]:
            if kw.lower() in text.lower():
                return cat

    for kw in EXP_KEYWORDS["award"]:
        if kw in text:
            return "award"

    for kw in EXP_KEYWORDS["activity"]:
        if kw in text:
            return "activity"

    return None


def classify_experiences_hybrid(experiences_text, api_key, dry_run=False):
    """
    经历分类：规则优先 + LLM 兜底。
    返回 {category}_count + {category}_detail, rule_classified_pct
    """
    empty_result = {
        "research_count": 0,
        "research_detail": "",
        "paper_count": 0,
        "paper_detail": "",
        "internship_count": 0,
        "internship_detail": "",
        "activity_count": 0,
        "activity_detail": "",
        "award_count": 0,
        "award_detail": "",
        "rule_classified_pct": 0,
    }

    if not experiences_text or str(experiences_text).strip() in ("nan", "None", ""):
        return empty_result

    text = str(experiences_text)
    items = re.split(r"[\n;；。]", text)
    items = [it.strip() for it in items if len(it.strip()) >= 3]

    if not items:
        return empty_result

    rule_results = {cat: [] for cat in EXP_KEYWORDS}
    unclassified = []

    for item in items:
        cat = _classify_experience_item(item)
        if cat:
            rule_results[cat].append(item)
        else:
            unclassified.append(item)

    rule_total = sum(len(v) for v in rule_results.values())
    total = rule_total + len(unclassified)
    pct = round(rule_total / total * 100) if total > 0 else 0

    result = {}
    for cat in EXP_KEYWORDS:
        result[f"{cat}_count"] = len(rule_results[cat])
        result[f"{cat}_detail"] = "\n".join(rule_results[cat]) if rule_results[cat] else ""

    # LLM 兜底
    if unclassified and not dry_run:
        llm_text = "\n".join(unclassified)[:2000]
        llm_result = classify_experiences_with_llm(llm_text, api_key)
        for cat in EXP_KEYWORDS:
            llm_c = llm_result.get(f"{cat}_count", 0) or 0
            llm_d = llm_result.get(f"{cat}_detail", "") or ""
            result[f"{cat}_count"] += llm_c
            if llm_d:
                existing = result[f"{cat}_detail"]
                result[f"{cat}_detail"] = existing + ("\n" if existing else "") + llm_d

    result["rule_classified_pct"] = pct
    return result


def classify_experiences_with_llm(experiences_text, api_key):
    """
    用 LLM 将经历文本分类。仅作为兜底。
    """
    if not experiences_text or str(experiences_text).strip() in ("nan", "None", ""):
        return {}

    text = str(experiences_text)[:3000]

    prompt = f"""分析以下学生的课外经历文本，将其分类并计数。每项经历只归入一类。

分类规则：
- research: 科研项目、课题研究、实验室工作、学术调研
- paper: 发表论文、专利、会议论文
- internship: 实习、工作经历（公司/机构）
- activity: 社团活动、志愿者、竞赛参与（非学术类）、学生组织
- award: 奖学金、荣誉称号、竞赛获奖

输出格式（严格JSON）：
{{"research_count": 0, "research_detail": "简要列举", "paper_count": 0, "paper_detail": "简要列举", "internship_count": 0, "internship_detail": "简要列举", "activity_count": 0, "activity_detail": "简要列举", "award_count": 0, "award_detail": "简要列举"}}

经历文本：
{text}"""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=LLM_CONFIG["base_url"], timeout=30)
        resp = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=LLM_CONFIG["max_tokens"],
            temperature=LLM_CONFIG["temperature"],
        )
        content = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"  LLM error (experience): {e}")

    return {}


def standardize_major_and_faculty_hybrid(
    major_text,
    api_key,
    reference_majors,
    school_major_df,
    background_majors_list,
    raw_to_std=None,
    dry_run=False,
):
    """专业标准化：raw lookup → fuzzy 94 categories → 规则推断 faculty，LLM 兜底。"""
    if not isinstance(major_text, str) or not major_text.strip():
        return {}

    bm, faculty = fuzzy_match_background_major(major_text, background_majors_list, raw_to_std)
    if bm:
        return {"background_major": bm, "faculty": faculty}

    if dry_run:
        return {"background_major": "其他", "faculty": None}

    return standardize_major_and_faculty_with_llm(major_text, api_key, background_majors_list)


def standardize_major_and_faculty_with_llm(major_text, api_key, reference_majors):
    """LLM 标准化本科专业（兜底）"""
    if not isinstance(major_text, str) or not major_text.strip():
        return {}

    ref_list = "\n".join(f"- {m}" for m in reference_majors)
    faculty_list = "\n".join(f"- {f}" for f in sorted(FACULTY_SET))

    prompt = f"""你是留学背景识别助手。给定一个本科专业原始名称，请将其映射到标准专业分类和学院。

标准专业分类（从中选择最匹配的一个）：
{ref_list}

学院分类（从中选择最匹配的一个）：
{faculty_list}

输出格式（严格JSON，不要解释）：
{{"background_major": "标准专业名", "faculty": "学院名"}}

如果无法匹配或不确定，background_major 填 "其他"，faculty 填最接近的学院。

原始专业名称：{major_text}"""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=LLM_CONFIG["base_url"], timeout=30)
        resp = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=LLM_CONFIG["temperature"],
        )
        content = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            result = json.loads(m.group())
            if result.get("background_major") not in set(reference_majors):
                result["background_major"] = "其他"
            if result.get("faculty") not in FACULTY_SET:
                result["faculty"] = None
            return result
    except Exception as e:
        print(f"  LLM error (major): {e}")

    return {}


def translate_target_major(chinese_major, api_key, school_major_df, dry_run=False):
    """中文录取专业 → 英文：fuzzy 优先，LLM 兜底。"""
    if not isinstance(chinese_major, str) or not chinese_major.strip():
        return None

    en_name, _ = fuzzy_match_major(chinese_major, school_major_df)
    if en_name:
        return en_name

    if dry_run:
        return chinese_major

    # LLM 翻译兜底
    prompt = f"""将以下中文硕士专业名翻译为标准的英文 Master 项目名称。

输出格式（严格JSON，不要解释）：
{{"target_major": "Master of ..."}}

中文专业名：{chinese_major}"""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=LLM_CONFIG["base_url"], timeout=30)
        resp = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=LLM_CONFIG["temperature"],
        )
        content = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            return json.loads(m.group()).get("target_major")
    except Exception as e:
        print(f"  LLM error (major translate): {e}")

    return chinese_major


def backfill_undergrad_from_text(row, text_col):
    """当本科专业缺失时，尝试从心得/经历文本正则提取"""
    text = str(row.get(text_col, ""))
    if not text or len(text) < 10:
        return row

    existing = row.get("background_major_original")
    if pd.notna(existing) and str(existing).strip() not in ("", "nan", "None"):
        return row

    # 尝试多种模式
    for pattern in [RE_BG_MAJOR, RE_UNDERGRAD_MAJOR]:
        m = pattern.search(text)
        if m:
            candidate = m.group(1).strip()
            # 过滤垃圾
            if len(candidate) < 2 or len(candidate) > 10:
                continue
            if any(bad in candidate for bad in _BACKFILL_BLOCKLIST):
                continue
            row["background_major_original"] = candidate
            return row

    return row


def backfill_university_from_text(text):
    """从 ApplySquare 心得文本提取毕业学校"""
    if not isinstance(text, str) or len(text) < 20:
        return None

    for pattern in [RE_DIRECT_UNIVERSITY, RE_BG_UNIVERSITY]:
        m = pattern.search(text)
        if m:
            candidate = m.group(1).strip()
            if 3 <= len(candidate) <= 20:
                normalized = normalize_school(candidate)
                if normalized:
                    return candidate

    tier_m = RE_SCHOOL_TIER.search(text)
    if tier_m:
        return f"[{tier_m.group(1)}]"

    return None


# ═══════════════════════════════════════════════════════
# 参考数据加载
# ═══════════════════════════════════════════════════════

OUTPUT_COLS = [
    "admitted",
    "target_university",
    "target_major",
    "background_university",
    "background_major_original",
    "background_major",
    "faculty",
    "gpa",
    "research_count",
    "research_detail",
    "paper_count",
    "paper_detail",
    "internship_count",
    "internship_detail",
    "activity_count",
    "activity_detail",
    "award_count",
    "award_detail",
    "ielts",
    "toefl",
]

EXP_COUNT_COLS = [
    "research_count",
    "paper_count",
    "internship_count",
    "activity_count",
    "award_count",
]
FLOAT_COLS = ["gpa", "ielts", "toefl"]


def load_reference_data():
    """加载 school_major_details + school_base + background_majors + raw→standard 查找表"""
    school_major_df = None
    school_base_df = None
    background_majors = []
    raw_to_std = {}

    sm_path = _resolve_data_path("school_major_details.feather")
    if os.path.exists(sm_path):
        school_major_df = pd.read_feather(sm_path)
        print(f"已加载 school_major_details: {len(school_major_df)} 条")

    sb_path = _resolve_data_path("school_base.feather")
    if os.path.exists(sb_path):
        school_base_df = pd.read_feather(sb_path)
        print(f"已加载 school_base: {len(school_base_df)} 条")

    cases_path = _resolve_data_path("cases.feather")
    if os.path.exists(cases_path):
        cases_df = pd.read_feather(cases_path)
        if "background_major" in cases_df.columns:
            background_majors = sorted(cases_df["background_major"].dropna().unique().tolist())
            print(f"已加载 background_major categories: {len(background_majors)} 个")

        # 构建 raw original name → (standard_major, faculty) 查找表
        if "background_major_original" in cases_df.columns:
            has_fac = "faculty" in cases_df.columns
            raw = cases_df[["background_major_original", "background_major"]]
            if has_fac:
                raw["_fac"] = cases_df["faculty"]
            raw = raw.dropna(subset=["background_major_original", "background_major"])
            raw = raw[raw["background_major_original"].astype(str).str.strip() != ""]
            for _, r in raw.iterrows():
                key = _normalize_major_text(str(r["background_major_original"]))
                if key and key not in raw_to_std:
                    fac = r.get("_fac") if has_fac else None
                    raw_to_std[key] = (r["background_major"], fac if pd.notna(fac) else None)
            print(f"已加载 raw→standard 查找表: {len(raw_to_std)} 条映射")

    return school_major_df, school_base_df, background_majors, raw_to_std


def _normalize_major_text(text):
    """预处理专业文本用于匹配：去括号内容、学位前缀、lower、合并空白"""
    if not isinstance(text, str):
        return ""
    t = re.sub(r"[（(][^)）]*[)）]", "", text)
    t = re.sub(
        r"(?i)bachelor\s+of\s+|本科|理学学士|工学学士|文学学士|管理学学士|经济学学士|法学学士|农学学士|医学学士",
        "",
        t,
    )
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def coerce_types(result_df):
    """统一类型转换"""
    for c in EXP_COUNT_COLS:
        result_df[c] = pd.to_numeric(result_df[c], errors="coerce").fillna(0).astype(int)
    for c in FLOAT_COLS:
        result_df[c] = pd.to_numeric(result_df[c], errors="coerce")
    result_df["admitted"] = result_df["admitted"].astype(int)
    return result_df


def ensure_output_cols(df):
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = None
    return df


def save_and_report(result_df, output_file):
    """保存 feather 并打印字段完整性报告"""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df = result_df[OUTPUT_COLS].copy()
    result_df = coerce_types(result_df)
    result_df.reset_index(drop=True, inplace=True)
    out_path = _OUTPUT_DIR / output_file if not os.path.isabs(output_file) else Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_feather(str(out_path))
    print(f"\n输出: {out_path} ({len(result_df)} 条)")
    print("字段完整性:")
    for c in OUTPUT_COLS:
        valid = result_df[c].notna().sum()
        print(f"  {c}: {valid}/{len(result_df)} ({valid / len(result_df) * 100:.0f}%)")
    return result_df


# ═══════════════════════════════════════════════════════
# 主处理流程
# ═══════════════════════════════════════════════════════


def _batch_process(unique_values, process_fn, desc, show_progress=True):
    """对唯一值列表逐个处理，带磁盘缓存。返回 {value: result} 映射。"""
    cache = {}
    total = len(unique_values)
    cache_hits = 0
    llm_calls = 0
    for i, val in enumerate(unique_values):
        if show_progress and (i + 1) % 50 == 0:
            print(f"  {desc} {i + 1}/{total} (hit={cache_hits}, llm={llm_calls})...", flush=True)
        # 磁盘缓存
        cached = _cache_get(desc, val)
        if cached is not None:
            cache[val] = cached
            cache_hits += 1
            continue
        result = process_fn(val)
        cache[val] = result
        _cache_set(desc, val, result)
        if result is None or (isinstance(result, dict) and not result):
            llm_calls += 1
            time.sleep(0.3)
    if show_progress:
        print(
            f"  {desc} DONE: {total} unique, {cache_hits} cache hits, {llm_calls} LLM", flush=True
        )
    return cache


def _batch_map(df, col, cache, target_cols, default=None):
    """将缓存结果映射回 DataFrame 的多列"""
    vals = df[col].astype(str)
    if default is None:
        default = {}
    for tcol in target_cols:
        df[tcol] = vals.map(
            lambda x, _tcol=tcol: (
                cache.get(x, default).get(_tcol, None)
                if isinstance(cache.get(x, default), dict)
                else None
            )
        )


# ═══════════════════════════════════════════════════════
# 主处理流程（唯一值批量处理版）
# ═══════════════════════════════════════════════════════


def process_compassedu(
    input_file,
    output_file,
    api_key,
    school_major_df,
    background_majors,
    dry_run=False,
    raw_to_std=None,
):
    """处理指南者爬虫数据（唯一值批量处理）"""
    print(f"读取: {input_file}", flush=True)
    df = pd.read_excel(input_file)
    print(f"  输入: {len(df)} 条", flush=True)

    # 1. Fuzzy 学校匹配
    df["_target_school"] = df["录取学校"].apply(normalize_school)
    valid = df["_target_school"].notna()
    print(f"  目标学校匹配: {valid.sum()}/{len(df)} ({valid.sum() / len(df) * 100:.1f}%)")
    df = df[valid].copy()
    df["target_university"] = df["_target_school"]

    # 2. 基本背景解析 (regex, 全量快速)
    print("Parsing 基本背景...", flush=True)
    bg_parsed = df["基本背景"].apply(parse_basic_background).apply(pd.Series)
    df["gpa"] = bg_parsed.get("gpa", None)
    df["ielts"] = bg_parsed.get("ielts", None)
    df["toefl"] = bg_parsed.get("toefl", None)

    # 3. 经历分类：对唯一文本批量处理
    unique_exp = df["主要经历"].fillna("").astype(str).unique()
    print(f"Classifying 经历 (unique={len(unique_exp)}, total={len(df)})...", flush=True)

    def _classify_exp(text):
        return classify_experiences_hybrid(text, api_key, dry_run)

    exp_cache = _batch_process(unique_exp, _classify_exp, "经历")
    exp_keys = [
        "research_count",
        "research_detail",
        "paper_count",
        "paper_detail",
        "internship_count",
        "internship_detail",
        "activity_count",
        "activity_detail",
        "award_count",
        "award_detail",
    ]
    for key in exp_keys:
        df[key] = (
            df["主要经历"]
            .fillna("")
            .astype(str)
            .map(lambda x, _key=key: exp_cache.get(x, {}).get(_key, None))
        )

    # 4. 本科专业标准化：唯一值批量处理
    unique_majors = df["本科专业"].fillna("").astype(str).unique()
    print(f"Standardizing 本科专业 (unique={len(unique_majors)})...", flush=True)

    def _std_major(major):
        return standardize_major_and_faculty_hybrid(
            major,
            api_key,
            background_majors,
            school_major_df,
            background_majors,
            raw_to_std,
            dry_run,
        )

    major_cache = _batch_process(unique_majors, _std_major, "专业")
    df["background_major_original"] = df["本科专业"]
    df["background_major"] = (
        df["本科专业"]
        .fillna("")
        .astype(str)
        .map(lambda x: major_cache.get(x, {}).get("background_major", "其他"))
    )
    df["faculty"] = (
        df["本科专业"]
        .fillna("")
        .astype(str)
        .map(lambda x: major_cache.get(x, {}).get("faculty", None))
    )

    # 5. 录取专业翻译：唯一值批量处理
    unique_target = df["录取专业"].fillna("").astype(str).unique()
    print(f"Translating 录取专业 (unique={len(unique_target)})...", flush=True)

    def _translate_major(cm):
        return translate_target_major(cm, api_key, school_major_df, dry_run)

    trans_cache = _batch_process(unique_target, _translate_major, "翻译")
    df["target_major"] = df["录取专业"].fillna("").astype(str).map(trans_cache)

    # 6. 最终输出
    df["admitted"] = 1
    df["background_university"] = df["毕业学校"]
    df = ensure_output_cols(df)
    return save_and_report(df, output_file)


def process_applysquare(
    input_file,
    output_file,
    api_key,
    school_major_df,
    background_majors,
    dry_run=False,
    raw_to_std=None,
):
    """处理 ApplySquare 爬虫数据（唯一值批量处理，含本科回填）"""
    print(f"读取: {input_file}", flush=True)
    df = pd.read_excel(input_file)
    print(f"  输入: {len(df)} 条", flush=True)

    # 1. Fuzzy 学校匹配
    school_col = "申请学校" if "申请学校" in df.columns else "录取学校"
    df["_target_school"] = df[school_col].apply(normalize_school)
    valid = df["_target_school"].notna()
    print(f"  目标学校匹配: {valid.sum()}/{len(df)} ({valid.sum() / len(df) * 100:.1f}%)")
    df = df[valid].copy()
    df["target_university"] = df["_target_school"]

    # 2. admitted 映射
    if "录取结果" in df.columns:
        df["admitted"] = df["录取结果"].map(APPLYSQUARE_ADMIT_MAP)
        removed = df["admitted"].isna().sum()
        df = df[df["admitted"].notna()].copy()
        df["admitted"] = df["admitted"].astype(int)
        print(f"  admitted 分布: {dict(df['admitted'].value_counts())}")
        print(f"  过滤掉非终态: {removed} 条")
    else:
        df["admitted"] = 1

    # 3. 数值字段
    if "GPA" in df.columns:
        df["gpa"] = pd.to_numeric(df["GPA"], errors="coerce")
        mask = df["gpa"] > 10
        df.loc[mask, "gpa"] = (df.loc[mask, "gpa"] / 25).clip(upper=4.0)
    if "IELTS" in df.columns:
        df["ielts"] = pd.to_numeric(df["IELTS"], errors="coerce")
    if "TOEFL" in df.columns:
        df["toefl"] = pd.to_numeric(df["TOEFL"], errors="coerce")

    # 4. 本科专业 + 缺失回填 + 学校回填
    df["background_university"] = df.get("毕业学校", None)
    df["background_major_original"] = df.get("本科专业", df.get("本科", None))

    exp_col = APPLYSQUARE_EXPERIENCE_COL if APPLYSQUARE_EXPERIENCE_COL in df.columns else None
    if exp_col:
        # 本科专业回填
        missing_before = df["background_major_original"].isna().sum()
        df = df.apply(lambda row: backfill_undergrad_from_text(row, exp_col), axis=1)
        missing_after = df["background_major_original"].isna().sum()
        if missing_before > missing_after:
            print(f"  本科专业回填: {missing_before - missing_after}/{missing_before} 条")

        # 毕业学校回填
        uni_missing_before = (
            df["background_university"].isna().sum()
            + (df["background_university"].astype(str).isin(["nan", "None", ""])).sum()
        )
        df["background_university"] = df.apply(
            lambda row: (
                row["background_university"]
                if pd.notna(row["background_university"])
                and str(row["background_university"]).strip() not in ("", "nan", "None")
                else backfill_university_from_text(str(row.get(exp_col, "")))
            ),
            axis=1,
        )
        uni_valid_after = df["background_university"].notna().sum()
        uni_filled = uni_valid_after - (len(df) - uni_missing_before)
        if uni_filled > 0:
            print(f"  毕业学校回填: {uni_filled} 条")

    # 5. 本科专业标准化：唯一值批量处理
    unique_majors = df["background_major_original"].fillna("").astype(str).unique()
    print(f"Standardizing 本科专业 (unique={len(unique_majors)})...", flush=True)

    def _std_major(major):
        return standardize_major_and_faculty_hybrid(
            major,
            api_key,
            background_majors,
            school_major_df,
            background_majors,
            raw_to_std,
            dry_run,
        )

    major_cache = _batch_process(unique_majors, _std_major, "专业")
    df["background_major"] = (
        df["background_major_original"]
        .fillna("")
        .astype(str)
        .map(lambda x: major_cache.get(x, {}).get("background_major", "其他"))
    )
    df["faculty"] = (
        df["background_major_original"]
        .fillna("")
        .astype(str)
        .map(lambda x: major_cache.get(x, {}).get("faculty", None))
    )

    # 6. 经历分类：唯一文本批量处理
    if exp_col:
        unique_exp = df[exp_col].fillna("").astype(str).unique()
        print(f"Classifying 经历 (unique={len(unique_exp)}, total={len(df)})...", flush=True)

        def _classify_exp(text):
            return classify_experiences_hybrid(text, api_key, dry_run)

        exp_cache = _batch_process(unique_exp, _classify_exp, "经历")
        exp_keys = [
            "research_count",
            "research_detail",
            "paper_count",
            "paper_detail",
            "internship_count",
            "internship_detail",
            "activity_count",
            "activity_detail",
            "award_count",
            "award_detail",
        ]
        for key in exp_keys:
            df[key] = (
                df[exp_col]
                .fillna("")
                .astype(str)
                .map(lambda x, _key=key: exp_cache.get(x, {}).get(_key, None))
            )
    else:
        print("未找到经历列，跳过经历分类。")

    # 7. 录取专业翻译：唯一值批量处理
    major_col = "申请专业" if "申请专业" in df.columns else "录取专业"
    unique_target = df[major_col].fillna("").astype(str).unique()
    print(f"Translating 录取专业 (unique={len(unique_target)})...", flush=True)

    def _translate_major(cm):
        return translate_target_major(cm, api_key, school_major_df, dry_run)

    trans_cache = _batch_process(unique_target, _translate_major, "翻译")
    df["target_major"] = df[major_col].fillna("").astype(str).map(trans_cache)

    # 8. 最终输出
    df = ensure_output_cols(df)
    return save_and_report(df, output_file)


def main():
    parser = argparse.ArgumentParser(description="转换爬虫数据到 cases.feather 格式")
    parser.add_argument("--source", required=True, choices=["compassedu", "applysquare"])
    parser.add_argument("--input", required=True, help="输入Excel文件路径")
    parser.add_argument("--output", default=None, help="输出feather文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只分析不调用LLM")
    args = parser.parse_args()

    if args.output is None:
        base = Path(args.input).stem
        args.output = f"{base}_clean.feather"

    print(f"Source: {args.source}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Model:  {LLM_CONFIG['model']}")
    if args.dry_run:
        print("Mode:  DRY RUN (无LLM调用)")
    print()

    api_key = get_api_key() if not args.dry_run else None
    school_major_df, school_base_df, background_majors, raw_to_std = load_reference_data()

    if args.source == "compassedu":
        process_compassedu(
            args.input,
            args.output,
            api_key,
            school_major_df,
            background_majors,
            args.dry_run,
            raw_to_std,
        )
    elif args.source == "applysquare":
        process_applysquare(
            args.input,
            args.output,
            api_key,
            school_major_df,
            background_majors,
            args.dry_run,
            raw_to_std,
        )


if __name__ == "__main__":
    main()
