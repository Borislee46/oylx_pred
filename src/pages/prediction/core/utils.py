"""
预测核心工具函数 + 院校-专业数据管理器。

数据管理器（SchoolMajorDataManager）：
  加载 school_major_details 表（院校-专业-要求映射），构建索引。
  这是全系统的基础数据源之一，被多个模块依赖。

语言分数归一化/反归一化：
  normalize_language_score: 原始分数 → [0,1] （IELTS / 9, TOEFL / 120）
  denormalize_language_score: [0,1] → 原始分数（含雅思 0.5 步进取整）
"""

from functools import lru_cache

import pandas as pd

from src.utils.app_data_loader import load_school_major_details_df


def normalize_language_score(score, language_type):
    """原始语言分数归一化到 [0,1] 区间。"""
    score = float(score)
    if language_type == "托福":
        return score / 120
    else:
        return score / 9


def denormalize_language_score(normalized_score, language_type, round_to_half=False):
    """归一化语言分数反归一化为原始分数。

    round_to_half=True（雅思）：反归一化后取整到最近的 0.5（如 5.5, 6.0, 6.5）。
    这是为了显示友好的雅思分数（而非 5.83 这样的奇怪数字）。
    """
    normalized_score = float(normalized_score)
    if language_type == "托福":
        return normalized_score * 120
    else:
        score = normalized_score * 9
        if round_to_half:
            return round(score * 2) / 2
        return score


# ── 院校-专业数据管理器 ──────────────────────────────────
class SchoolMajorDataManager:
    """院校-专业详细信息的数据管理器（单例模式，模块级 _data_manager）。

    数据来源：school_major_details 表（config 目录下的 Excel/CSV）
    内容：每个(院校, 专业)组合的详细信息——学习年限、学费、语言要求、GPA要求、面试等。

    索引结构：
      _details_index[(学校, 专业英文名称)] = {字段: 值, _lang_reqs: LanguageRequirement}

    延迟加载：首次访问任何属性时自动加载数据 + 构建索引。
    warm_up() 在 resource_loader 中预调用，避免首次用户请求时的延迟峰值。
    """

    def __init__(self):
        self._details_df = None          # 原始 DataFrame
        self._valid_combinations = None  # {(学校, 专业), ...} 有效组合集合
        self._valid_universities = None  # {学校, ...} 有效院校集合
        self._valid_majors = None        # {专业, ...} 有效专业集合
        self._details_version = None     # 数据版本哈希
        self._details_index = None       # {(学校, 专业): row_dict} 查询索引

    @property
    def details_df(self):
        """延迟加载原始数据（首次访问时触发加载+索引构建）。"""
        if self._details_df is None:
            self._details_df = load_school_major_details_df()
            self._build_indexes()
        return self._details_df

    def _build_indexes(self):
        """从原始 DataFrame 构建所有查询索引。

        关键操作：
        - 为每行预提取语言要求（LanguageRequirementPenalty.extract_requirement）
        - 构建 (学校, 专业) 二元组索引
        - 导出有效院校、有效专业集合
        """
        df = self._details_df
        if df is None or df.empty:
            self._details_index = {}
            self._valid_combinations = set()
            self._valid_universities = set()
            self._valid_majors = set()
            return

        temp_df = df.copy()
        temp_df["学校"] = temp_df["学校"].astype(str)
        temp_df["专业英文名称"] = temp_df["专业英文名称"].astype(str)

        from src.pages.prediction.result_modifier.language_penalty import (
            LanguageRequirementPenalty,
        )

        self._details_index = {}
        for row_dict in temp_df.to_dict("records"):
            row_dict["_lang_reqs"] = LanguageRequirementPenalty.extract_requirement(row_dict)
            self._details_index[(row_dict["学校"], row_dict["专业英文名称"])] = row_dict

        self._valid_combinations = set(self._details_index.keys())
        self._valid_universities = set(temp_df["学校"].unique())
        self._valid_majors = set(temp_df["专业英文名称"].unique())

    @property
    def valid_universities(self) -> set[str]:
        if self._details_df is None:
            _ = self.details_df
        return self._valid_universities

    @property
    def valid_majors(self) -> set[str]:
        if self._details_df is None:
            _ = self.details_df
        return self._valid_majors

    @property
    def details_version(self) -> int:
        """数据版本：基于 (学校, 专业英文名称) 列的哈希值。

        用于缓存失效检测：版本变化 → 所有依赖缓存自动失效。
        """
        if self._details_version is None:
            self._details_version = self._compute_details_version()
        return self._details_version

    def _compute_details_version(self) -> int:
        df = self.details_df
        if df is None or df.empty:
            return 0
        from pandas.util import hash_pandas_object

        return int(hash_pandas_object(df[["学校", "专业英文名称"]]).sum())

    @property
    def valid_combinations(self) -> set[tuple[str, str]]:
        if self._details_df is None:
            _ = self.details_df
        return self._valid_combinations

    def get_row(self, university: str, major: str):
        """按 (院校, 专业) 查询详情行。"""
        if self._details_index is None:
            _ = self.details_df
        return self._details_index.get((university, major))

    def _is_new_major_cached(self, university: str, major: str, version: int) -> bool:
        row = self.get_row(university, major)
        if row is not None:
            is_new = row.get("新增专业")
            return is_new is not None and str(is_new).strip() != ""
        return False

    def warm_up(self):
        """预热：触发数据加载（在 resource_loader 中预调用）。"""
        _ = self.details_df


# ── 模块级单例 ──────────────────────────────────────────
_data_manager = SchoolMajorDataManager()


# ── 背景专业 → 学部映射（缓存）────────────────────────
@lru_cache(maxsize=1)
def _get_faculty_map(df_hash: int, df_id: int):
    """从 cases_df 构建背景专业 → 学部映射。

    参数 df_hash + df_id 作为缓存键（确保 cases_df 变化时重新计算）。
    @lru_cache(maxsize=1)：只保留最近一次调用结果。
    """
    from src.utils.app_data_loader import load_raw_cases_data

    df = load_raw_cases_data()
    if df.empty or "background_major" not in df.columns:
        return {}
    return df.groupby("background_major")["faculty"].first().to_dict()


def get_background_faculty(background_major: str, cases_df: pd.DataFrame | None) -> str | None:
    """获取背景专业的学部归属。

    cases_df 的 groupby 用于缓存键（df_hash, df_id），
    确保不同版本或不同数据集的缓存隔离。
    """
    if not background_major or cases_df is None:
        return None

    faculty_map = _get_faculty_map(len(cases_df), id(cases_df))
    return faculty_map.get(background_major)


# ── 格式化工具 ──────────────────────────────────────────
def format_list_field(field_list: list[str]) -> str:
    """列表格式化：≤3 项直接拼接，>3 项前 3 项 + 等 N 项。"""
    if not field_list:
        return ""
    return (
        ", ".join(field_list)
        if len(field_list) <= 3
        else f"{', '.join(field_list[:3])} 等 {len(field_list) - 3} 项"
    )


def format_field(value) -> str:
    if value is None or value == "":
        return ""
    return str(value)


def format_float(value, decimals: int = 2):
    if value is None or value == "":
        return ""
    return round(float(value), decimals)


def get_cached_major_similarity(
    target_major: str = None,
    background_major: str = None,
    cache=None,
    major1: str = None,
    major2: str = None,
) -> float:
    """从预计算相似度缓存中查询两个专业的相似度。

    支持两种参数格式：
    - (target_major, background_major)：目标 vs 背景
    - (major1, major2)：任意两个专业
    """
    bg = background_major or major2
    target = target_major or major1

    if cache is None:
        return 0.0

    bg_key = str(bg).strip().lower()
    target_key = str(target).strip().lower()

    return float(cache.get((bg_key, target_key), 0.0))


# ── 院校-专业详情格式化 ─────────────────────────────────
def format_school_major_details_from_row(row: dict) -> str:
    """将一行院校-专业数据格式化为可读的多行文本。

    包含字段：专业名称、学习年限、学费、语言要求、面试/笔试、申请注意事项等 20 个字段。
    空值字段跳过，专业中文名称不显示前缀（作为标题行）。
    """
    if not row:
        return "无详细信息"

    key_fields = [
        "专业中文名称", "学习年限", "学费", "授课语言", "录取要求",
        "专业背景要求", "GPA要求", "IELTS", "TOEFL", "CET-6",
        "考试要求", "特殊要求", "申请方式", "推荐信方式",
        "成绩送分要求", "是否面试", "是否笔试", "考核形式",
        "申请注意事项", "专业网址",
    ]

    details = []
    for field in key_fields:
        field_value = row.get(field)
        if pd.notna(field_value) and str(field_value).strip():
            prefix = "专业网址: " if field == "专业网址" else f"{field}: "
            details.append(f"{prefix}{field_value}")

    if not details:
        return "无详细信息"

    # 专业中文名称作为首行，不显示字段名前缀
    if details and details[0].startswith("专业中文名称:"):
        details[0] = details[0].replace("专业中文名称: ", "", 1).strip()

    return "\n".join(details)


def get_school_major_details(
    university: str | None, major: str | None, return_df: bool = False
) -> str | pd.DataFrame | None:
    """查询(院校, 专业)的详细信息（格式化文本 or 原始 DataFrame）。"""
    if return_df:
        return _data_manager.details_df

    return _get_school_major_details_cached(university, major, _data_manager.details_version)


def _get_school_major_details_cached(university: str, major: str, version: int) -> str:
    row = _data_manager.get_row(university, major)
    if row is not None:
        return format_school_major_details_from_row(row)
    return "无详细信息"


def get_valid_school_major_set() -> set[tuple[str, str]]:
    return _data_manager.valid_combinations


def has_school_major_details(university: str, major: str) -> bool:
    return (str(university), str(major)) in _data_manager.valid_combinations


@lru_cache(maxsize=1000)
def is_new_major(university: str, major: str) -> bool:
    """检查是否为新增专业（用于 UI 标记"新专业"标签）。"""
    return _data_manager._is_new_major_cached(university, major, _data_manager.details_version)
