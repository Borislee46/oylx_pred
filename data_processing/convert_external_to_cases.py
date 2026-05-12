"""
将 2025-硕士-外部案例.xlsx 转换为 cases.feather 格式。

这个数据源已经高度结构化——实习/科研/活动/获奖的数量和详情都已拆好，
专业英文名也已翻译。只需标准化本科专业 → background_major + faculty。

用法:
  py convert_external_to_cases.py
  py convert_external_to_cases.py --input my_file.xlsx --output my_output.feather

依赖: pip install pandas pyarrow openpyxl openai rapidfuzz
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import pandas as pd

# ── LLM 配置 ──
LLM_CONFIG = {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "max_tokens": 800,
    "temperature": 0.0,
}

# ── 标准学院列表（与 cases.feather 一致） ──
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

# ── 路径推导 ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "src" / "machine_learning_models" / "data"
_OUTPUT_DIR = _DATA_DIR / "external"
_CACHE_DIR = _SCRIPT_DIR / ".llm_cache"


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


def _resolve_data_path(filename):
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
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    config_path = _PROJECT_ROOT / "config" / "app_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        env = os.environ.get("APP_ENV", "test")
        return cfg.get(env, cfg.get("test", {})).get("OPEN_AI_API_KEY", "")
    print("ERROR: 未找到 API key。")
    sys.exit(1)


def load_reference_data():
    """加载 background_major 列表 + school_major_details + raw→standard 查找表"""
    majors = []
    school_major_df = None
    raw_to_std = {}

    cases_path = _resolve_data_path("cases.feather")
    if os.path.exists(cases_path):
        df = pd.read_feather(cases_path)
        if "background_major" in df.columns:
            majors = sorted(df["background_major"].dropna().unique().tolist())
            print(f"已加载 background_major: {len(majors)} 类")

        if "background_major_original" in df.columns:
            has_fac = "faculty" in df.columns
            raw = df[["background_major_original", "background_major"]]
            if has_fac:
                raw["_fac"] = df["faculty"]
            raw = raw.dropna(subset=["background_major_original", "background_major"])
            raw = raw[raw["background_major_original"].astype(str).str.strip() != ""]
            for _, r in raw.iterrows():
                key = _normalize_major_text(str(r["background_major_original"]))
                if key and key not in raw_to_std:
                    fac = r.get("_fac") if has_fac else None
                    raw_to_std[key] = (r["background_major"], fac if pd.notna(fac) else None)
            print(f"已加载 raw→standard 查找表: {len(raw_to_std)} 条映射")

    sm_path = _resolve_data_path("school_major_details.feather")
    if os.path.exists(sm_path):
        school_major_df = pd.read_feather(sm_path)
        print(f"已加载 school_major_details: {len(school_major_df)} 条")

    return majors, school_major_df, raw_to_std


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


def normalize_gpa(gpa_val, scale):
    """将 GPA 统一到 4.0 分制"""
    if pd.isna(gpa_val) or pd.isna(scale):
        return None
    scale = float(scale)
    gpa = float(gpa_val)
    if scale == 4.0:
        return round(gpa, 2)
    elif scale == 5.0:
        return round(gpa * 4.0 / 5.0, 2)
    elif scale == 7.0:
        return round(gpa * 4.0 / 7.0, 2)
    elif scale >= 10:
        return round(min(gpa / 25, 4.0), 2)
    return round(gpa, 2)


def fuzzy_match_major(chinese_name, school_major_df):
    """Fuzzy 匹配中文录取专业名 → 英文名 + faculty（基于 school_major_details）"""
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


# ── 本科专业 → 学院推断规则 ──
# 顺序重要：更具体的模式在前，避免被宽泛规则吞噬
_MAJOR_FACULTY_RULES = [
    (re.compile(r"建筑|城乡规划|风景园林|城市规划|环境设计|建筑学"), "建筑学院"),
    (
        re.compile(
            r"设计(?!学)|工业设计|产品设计|视觉传达|服装设计|数字媒体艺术|数媒艺术|艺术设计"
        ),
        "设计学院",
    ),
    (
        re.compile(
            r"计算机|软件工程|人工智能|AI|网络工程|物联网|数据科学|大数据|通信工程|自动化|电气|信息工程|数字媒体技术|信息安全"
        ),
        "计算机学院",
    ),
    (
        re.compile(
            r"工商管理|会计|金融|经济|市场|国贸|商务|财务|管理科学|物流|供应链|酒店|旅游管理|人力资源|保险|精算|工业工程|电子商务|审计|税务"
        ),
        "商学院",
    ),
    (
        re.compile(
            r"土木|机械|材料|化工|化学工程|环境工程|能源动力|能源|测绘|交通|水利|安全工程|食品|航空航天|包装|冶金|纺织|船舶|生物工程|生物医学工程|车辆工程|测控"
        ),
        "工程学院",
    ),
    (
        re.compile(
            r"数学|物理(?!治疗)|化学(?!工程)|生物(?!医学工程|工程)|统计|地理|地质|海洋|大气|天文|生态|园艺|水产|心理学|应用物理|应用化学"
        ),
        "理学院",
    ),
    (re.compile(r"法学|法律|知识产权"), "法学院"),
    (
        re.compile(r"医学(?!工程|技术)|临床|药学|护理|口腔|中医|公共卫生|兽医|康复|营养|医学技术"),
        "医学院",
    ),
    (
        re.compile(
            r"文学|历史|哲学|语言学|英语|日语|法语|德语|翻译|新闻|传播|广告|汉语言|公共关系|文化产业|对外汉语"
        ),
        "文学院",
    ),
    (
        re.compile(r"社会学|政治|公共管理|国际关系|人类学|社会工作|行政管理|社会保障|宗教学"),
        "社会科学院",
    ),
    (
        re.compile(r"艺术(?!设计)|美术学|音乐|戏剧|影视|电影|动画|舞蹈|播音|编导|书法|摄影"),
        "艺术学院",
    ),
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


def fuzzy_match_background_major(major_text, background_majors_list, raw_to_std=None):
    """6层匹配本科专业 → 94 standard background_major + faculty"""
    if not isinstance(major_text, str) or not major_text.strip():
        return None, None

    from rapidfuzz import fuzz, process

    mt = major_text.strip()
    mt_norm = _normalize_major_text(mt)

    # Layer 1: 精确匹配 raw→standard 查找表
    if raw_to_std and mt_norm in raw_to_std:
        bm, fac = raw_to_std[mt_norm]
        if fac is None:
            fac = _infer_faculty_from_major(bm, original_text=mt)
        return bm, fac

    # Layer 2: 精确匹配 94 标准类别名
    for bm in background_majors_list:
        if bm == mt or bm == mt_norm:
            return bm, _infer_faculty_from_major(bm, original_text=mt)

    # Layer 3: 子串匹配 (加长度 guard)
    for bm in background_majors_list:
        if len(bm) >= 3:
            if bm in mt or mt in bm:
                shorter = min(len(bm), len(mt))
                longer = max(len(bm), len(mt))
                if shorter / longer >= 0.35:
                    return bm, _infer_faculty_from_major(bm, original_text=mt)

    # Layer 4: Fuzzy 匹配 raw→standard 查找表 keys
    if raw_to_std:
        raw_keys = list(raw_to_std.keys())
        result = process.extractOne(mt_norm, raw_keys, scorer=fuzz.ratio, score_cutoff=80)
        if result:
            bm, fac = raw_to_std[result[0]]
            if fac is None:
                fac = _infer_faculty_from_major(bm, original_text=mt)
            return bm, fac

    # Layer 5: Fuzzy 匹配 94 标准类别
    result = process.extractOne(mt_norm, background_majors_list, scorer=fuzz.ratio, score_cutoff=70)
    if result:
        bm = result[0]
        return bm, _infer_faculty_from_major(bm, original_text=mt)

    return None, None


def standardize_major_hybrid(
    major_text, api_key, reference_majors, school_major_df, raw_to_std=None
):
    """专业标准化：raw lookup → fuzzy 94 categories → 规则 faculty，LLM 兜底"""
    if not isinstance(major_text, str) or not major_text.strip():
        return {}

    bm, faculty = fuzzy_match_background_major(major_text, reference_majors, raw_to_std)
    if bm:
        return {"background_major": bm, "faculty": faculty}

    # 2. LLM 兜底
    ref_list = "\n".join(f"- {m}" for m in reference_majors)
    faculty_list = "\n".join(f"- {f}" for f in sorted(FACULTY_SET))

    prompt = f"""你是留学背景识别助手。给定一个本科专业原始名称，请映射到标准专业分类和学院。

标准专业分类（从中选一个最匹配的）：
{ref_list}

学院分类（从中选一个最匹配的）：
{faculty_list}

输出严格JSON（不要解释）：
{{"background_major": "标准专业名", "faculty": "学院名"}}
若无法匹配，background_major 填"其他"。

原始专业名称：{major_text}"""

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
            result = json.loads(m.group())
            if result.get("background_major") not in set(reference_majors):
                result["background_major"] = "其他"
            if result.get("faculty") not in FACULTY_SET:
                result["faculty"] = None
            return result
    except Exception as e:
        print(f"  LLM error: {e}")

    return {}


def merge_experience(count1, detail1, count2, detail2):
    """合并两组经历（如实习+工作 → internship）"""
    c1 = int(count1) if pd.notna(count1) else 0
    c2 = int(count2) if pd.notna(count2) else 0
    parts = []
    if pd.notna(detail1) and str(detail1).strip():
        parts.append(str(detail1).strip())
    if pd.notna(detail2) and str(detail2).strip():
        parts.append(str(detail2).strip())
    return c1 + c2, "\n".join(parts) if parts else None


def convert(input_file, output_file, api_key, reference_majors, school_major_df, raw_to_std=None):
    print(f"读取: {input_file}", flush=True)
    df = pd.read_excel(input_file)
    print(f"  输入: {len(df)} 条", flush=True)

    result = pd.DataFrame(index=df.index)

    # ── 直接映射字段 ──
    result["admitted"] = 1
    result["target_university"] = df["申请院校"]
    result["target_major"] = df["专业英文名称修正"]
    result["background_university"] = df["就读院校1"]
    result["background_major_original"] = df["就读专业1"]

    # GPA 统一到 4.0
    gpa_normalized = []
    for _, row in df.iterrows():
        gpa_normalized.append(normalize_gpa(row.get("GPA1"), row.get("GPA分制1")))
    result["gpa"] = gpa_normalized

    # 语言成绩
    result["ielts"] = pd.to_numeric(df["IELTS分数"], errors="coerce")
    result["toefl"] = pd.to_numeric(df["TOEFL分数"], errors="coerce")

    # ── 经历字段：实习+工作合并为 internship ──
    int_counts, int_details = [], []
    for _, row in df.iterrows():
        c, d = merge_experience(
            row.get("实习数量"), row.get("实习经历"), row.get("工作数量"), row.get("工作经历")
        )
        int_counts.append(c)
        int_details.append(d)
    result["internship_count"] = int_counts
    result["internship_detail"] = int_details

    result["research_count"] = df["科研数量"].fillna(0).astype(int)
    result["research_detail"] = df["科研经历"]
    result["paper_count"] = df["发表数量"].fillna(0).astype(int)
    result["paper_detail"] = df["发表经历"]
    result["activity_count"] = df["活动数量"].fillna(0).astype(int)
    result["activity_detail"] = df["活动经历"]
    result["award_count"] = df["获奖数量"].fillna(0).astype(int)
    result["award_detail"] = df["获奖经历"]

    # ── 本科专业标准化：fuzzy + LLM ──
    unique_majors = df["就读专业1"].dropna().astype(str).unique()
    print(
        f"标准化本科专业 → background_major + faculty ({len(unique_majors)} unique)...", flush=True
    )
    major_cache = {}

    for i, major in enumerate(unique_majors):
        if not major or major == "nan":
            major_cache[major] = {}
            continue
        # 磁盘缓存
        cached = _cache_get("major", major)
        if cached is not None:
            major_cache[major] = cached
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(unique_majors)} (cache hit)...", flush=True)
            continue
        r = standardize_major_hybrid(major, api_key, reference_majors, school_major_df, raw_to_std)
        major_cache[major] = r
        _cache_set("major", major, r)
        if not r:
            time.sleep(0.3)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(unique_majors)} unique majors done", flush=True)

    print(f"  专业标准化完成，映射回 {len(df)} 行...", flush=True)
    df["_bg_major_raw"] = df["就读专业1"].astype(str)
    result["background_major"] = df["_bg_major_raw"].map(
        lambda x: major_cache.get(x, {}).get("background_major", "其他")
    )
    result["faculty"] = df["_bg_major_raw"].map(
        lambda x: major_cache.get(x, {}).get("faculty", None)
    )

    # ── 类型转换 ──
    for c in EXP_COUNT_COLS:
        result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0).astype(int)
    result["gpa"] = pd.to_numeric(result["gpa"], errors="coerce")
    result["ielts"] = pd.to_numeric(result["ielts"], errors="coerce")
    result["toefl"] = pd.to_numeric(result["toefl"], errors="coerce")
    result["admitted"] = result["admitted"].astype(int)

    # ── 保存 ──
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.reset_index(drop=True, inplace=True)
    out_path = _OUTPUT_DIR / output_file if not os.path.isabs(output_file) else Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = result[OUTPUT_COLS]
    result.to_feather(str(out_path))
    print(f"\n输出: {out_path} ({len(result)} 条)", flush=True)

    print("字段完整性:")
    for c in OUTPUT_COLS:
        valid = result[c].notna().sum()
        print(f"  {c}: {valid}/{len(result)} ({valid / len(result) * 100:.0f}%)")

    return result


def main():
    parser = argparse.ArgumentParser(description="转换外部案例数据到 cases.feather")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.input is None:
        args.input = str(_SCRIPT_DIR / "2025-硕士-外部案例.xlsx")
    if args.output is None:
        args.output = f"{Path(args.input).stem}_clean.feather"

    api_key = get_api_key()
    reference_majors, school_major_df, raw_to_std = load_reference_data()

    convert(args.input, args.output, api_key, reference_majors, school_major_df, raw_to_std)


if __name__ == "__main__":
    main()
