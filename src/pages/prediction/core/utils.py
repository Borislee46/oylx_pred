from functools import lru_cache

import pandas as pd

from src.pages.prediction.school_details_config import school_details_key_fields
from src.utils.app_data_loader import load_school_major_details_df
from src.utils.logger import setup_logger


def normalize_language_score(score, language_type):
    try:
        score = float(score)
        if language_type == "托福":
            return score / 120
        else:
            return score / 9
    except Exception:
        return score


def denormalize_language_score(normalized_score, language_type, round_to_half=False):
    try:
        normalized_score = float(normalized_score)
        if language_type == "托福":
            return normalized_score * 120
        else:
            score = normalized_score * 9
            if round_to_half:
                return round(score * 2) / 2
            return score
    except Exception:
        return normalized_score


_utils_logger = setup_logger("page3", "prediction")


class SchoolMajorDataManager:
    def __init__(self):
        self._details_df = None
        self._valid_combinations = None
        self._valid_universities = None
        self._valid_majors = None
        self._details_version = None
        self._details_index = None

    @property
    def details_df(self):
        if self._details_df is None:
            self._details_df = load_school_major_details_df()
            self._build_indexes()
        return self._details_df

    def _build_indexes(self):
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

        self._details_index = {
            (row["学校"], row["专业英文名称"]): row for _, row in temp_df.iterrows()
        }
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
        if self._details_df is None:
            _ = self.details_df
        return self._details_index.get((str(university), str(major)))

    def warm_up(self):
        _ = self.details_df


_data_manager = SchoolMajorDataManager()


@lru_cache(maxsize=1)
def _get_faculty_map(df_hash: int, df_id: int):
    from src.utils.app_data_loader import load_raw_cases_data

    df = load_raw_cases_data()
    if df.empty or "background_major" not in df.columns:
        return {}
    return df.groupby("background_major")["faculty"].first().to_dict()


def get_background_faculty(background_major: str, cases_df: pd.DataFrame | None) -> str | None:
    if not background_major or cases_df is None:
        return None

    faculty_map = _get_faculty_map(len(cases_df), id(cases_df))
    return faculty_map.get(background_major)


def format_list_field(field_list: list[str]) -> str:
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
    try:
        if value is None or value == "":
            return ""
        return round(float(value), decimals)
    except Exception:
        return value


def _create_major_similarity_key(major1: str, major2: str) -> str:
    key_pair = tuple(sorted([major1, major2]))
    return f"{key_pair[0]}|{key_pair[1]}"


def get_cached_major_similarity(
    target_major: str = None,
    background_major: str = None,
    cache: dict = None,
    major1: str = None,
    major2: str = None,
) -> float:
    first_major = target_major or major1
    second_major = background_major or major2

    if not first_major or not second_major or cache is None:
        return 0.0

    key = _create_major_similarity_key(first_major, second_major)
    return cache.get(key, 0.0)


def get_cached_major_similarities_batch(
    pairs: list[tuple[str, str]], cache: dict = None
) -> list[float]:
    if not pairs or cache is None:
        return [0.0] * len(pairs)

    return [
        get_cached_major_similarity(major1=target, major2=background, cache=cache)
        for target, background in pairs
    ]


def format_school_major_details_from_row(row: pd.Series) -> str:
    if row is None or row.empty:
        return "无详细信息"

    key_fields = school_details_key_fields

    details = []
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
    return _is_new_major_cached(university, major, _data_manager.details_version)


def _is_new_major_cached(university: str, major: str, version: int) -> bool:
    row = _data_manager.get_row(university, major)
    if row is not None:
        is_new = row.get("新增专业")
        return pd.notna(is_new) and str(is_new).strip() != ""
    return False
