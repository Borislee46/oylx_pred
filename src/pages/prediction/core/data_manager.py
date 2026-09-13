import logging
from functools import lru_cache

import pandas as pd

from src.pages.prediction.app_data import load_school_major_details_df

logger = logging.getLogger(__name__)


class SchoolMajorDataManager:
    def __init__(self):
        self._details_df: pd.DataFrame | None = None
        self._valid_combinations: set[tuple[str, str]] | None = None
        self._valid_universities: set[str] | None = None
        self._valid_majors: set[str] | None = None
        self._details_version: int | None = None
        self._details_index: dict[tuple[str, str], dict] | None = None

    @property
    def details_df(self) -> pd.DataFrame | None:
        if self._details_df is None:
            logger.debug("首次加载 school_major_details 数据...")
            self._details_df = load_school_major_details_df()
            if self._details_df is not None and not self._details_df.empty:
                self._build_indexes()
                logger.info(
                    "school_major_details 加载完成 | %d 行 | %d 个唯一组合",
                    len(self._details_df),
                    len(self._valid_combinations) if self._valid_combinations else 0,
                )
            else:
                logger.warning("school_major_details 数据为空或加载失败")
        return self._details_df

    def _build_indexes(self):
        df = self._details_df
        if df is None or df.empty:
            logger.warning("DataFrame 为空，跳过索引构建")
            self._details_index = {}
            self._valid_combinations = set()
            self._valid_universities = set()
            self._valid_majors = set()
            return

        temp_df = df.copy()
        temp_df["学校"] = temp_df["学校"].astype(str)
        temp_df["专业英文名称"] = temp_df["专业英文名称"].astype(str)

        from src.adjustment.language_penalty import (
            LanguageRequirementPenalty,
        )

        self._details_index = {}
        for row_dict in temp_df.to_dict("records"):
            row_dict["_lang_reqs"] = LanguageRequirementPenalty.extract_requirement(row_dict)
            self._details_index[(row_dict["学校"], row_dict["专业英文名称"])] = row_dict

        self._valid_combinations = set(self._details_index.keys())
        self._valid_universities = set(temp_df["学校"].unique())
        self._valid_majors = set(temp_df["专业英文名称"].unique())

        logger.debug(
            "索引构建完成 | 院校=%d | 专业=%d | 组合=%d",
            len(self._valid_universities),
            len(self._valid_majors),
            len(self._valid_combinations),
        )

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
    def valid_combinations(self) -> set[tuple[str, str]]:
        if self._details_df is None:
            _ = self.details_df
        return self._valid_combinations

    @property
    def details_version(self) -> int:
        if self._details_version is None:
            self._details_version = self._compute_details_version()
            logger.debug("details_version=%d", self._details_version)
        return self._details_version

    def _compute_details_version(self) -> int:
        df = self.details_df
        if df is None or df.empty:
            return 0
        from pandas.util import hash_pandas_object

        return int(hash_pandas_object(df[["学校", "专业英文名称"]]).sum())

    def get_row(self, university: str, major: str) -> dict | None:
        if self._details_index is None:
            _ = self.details_df
        if self._details_index is None:
            return None
        return self._details_index.get((university, major))

    def warm_up(self):
        logger.debug("SchoolMajorDataManager 预热开始...")
        _ = self.details_df
        logger.info(
            "SchoolMajorDataManager 预热完成 | %d 个组合就绪",
            len(self._valid_combinations) if self._valid_combinations else 0,
        )


_data_manager = SchoolMajorDataManager()


def format_school_major_details_from_row(row: dict) -> str:
    if not row:
        return "无详细信息"

    key_fields = [
        "专业中文名称",
        "学习年限",
        "学费",
        "授课语言",
        "录取要求",
        "专业背景要求",
        "GPA要求",
        "IELTS",
        "TOEFL",
        "CET-6",
        "考试要求",
        "特殊要求",
        "申请方式",
        "推荐信方式",
        "成绩送分要求",
        "是否面试",
        "是否笔试",
        "考核形式",
        "申请注意事项",
        "专业网址",
    ]

    details: list[str] = []
    for field in key_fields:
        field_value = row.get(field)
        if pd.notna(field_value) and str(field_value).strip():
            prefix = "专业网址: " if field == "专业网址" else f"{field}: "
            details.append(f"{prefix}{field_value}")

    if not details:
        return "无详细信息"

    if details and details[0].startswith("专业中文名称:"):
        details[0] = details[0].replace("专业中文名称: ", "", 1).strip()

    return "\n".join(details)


def get_school_major_details(
    university: str | None, major: str | None, return_df: bool = False
) -> str | pd.DataFrame | None:
    if return_df:
        return _data_manager.details_df
    if university is None or major is None:
        return "无详细信息"
    return _get_school_major_details_cached(university, major, _data_manager.details_version)


@lru_cache(maxsize=512)
def _get_school_major_details_cached(university: str, major: str, version: int) -> str:
    row = _data_manager.get_row(university, major)
    return format_school_major_details_from_row(row) if row else "无详细信息"


def get_valid_school_major_set() -> set[tuple[str, str]]:
    return _data_manager.valid_combinations


def has_school_major_details(university: str, major: str) -> bool:
    return (str(university), str(major)) in _data_manager.valid_combinations
