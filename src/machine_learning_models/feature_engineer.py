# =============================================================================
# 特征工程 (Feature Engineer)
# ─────────────────────────────────────────────────────────────────────────────
# 将原始录取数据转化为 XGBoost 可用的特征矩阵。
# 设计遵循 scikit-learn fit/transform 范式，保证训练和推理时特征一致。
#
# 关键设计决策：
# 1. log1p 变换处理右偏计数特征
# 2. 99分位数 capping 处理极端值
# 3. TOEFL/IELTS 统一归一化到 [0,1]
# 4. 中位数填充缺失值（而非均值或 0）
# 5. fit/transform 分离的 categorical 对齐（防止推理时出现新类别）
# =============================================================================

import numpy as np
import pandas as pd
from data_config import (
    CATEGORICAL_COLUMNS,
    COUNT_COLUMNS_FOR_LOG_TRANSFORM,
    IRRELEVANT_COLUMNS,
    TEXT_COLUMNS,
)


class FeatureEngineer:
    def __init__(self):
        self.numeric_medians = {}
        self.cap_values = {}
        self.existing_categorical_columns = []
        self.categorical_levels = {}

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data = data.drop(columns=IRRELEVANT_COLUMNS, errors="ignore")

        for col in TEXT_COLUMNS:
            if col in data.columns and data[col].isnull().any():
                data[col] = data[col].fillna("")

        return data

    # ─────────────────────────────────────────────────────────────────────────
    # 缺失值处理：中位数填充 (Median Imputation)
    # ─────────────────────────────────────────────────────────────────────────
    # 为什么是中位数而不是均值或 0？
    # - 均值：对偏态分布（如实习经历右偏）失真严重，一个 10 段实习的学生
    #   能拉高均值 20%，所有缺失值都被过高估计
    # - 0：0 对 GPA 不具备任何"中性"意义，会引入严重的负偏置
    # - 中位数：对偏态稳健，50% 分位不受极值影响，是"典型值"的最佳估计
    #   在录取数据中，中位数 ≈ 保守估计，不会把缺失值的学生抬高到不合理水平
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_numeric_missing(self, data: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in TEXT_COLUMNS]

        if is_fit:
            for col in numeric_columns:
                self.numeric_medians[col] = data[col].median()
        for col in numeric_columns:
            if col in self.numeric_medians and data[col].isnull().any():
                data[col] = data[col].fillna(self.numeric_medians[col])

        return data

    # ─────────────────────────────────────────────────────────────────────────
    # 分类特征对齐：防止推理时出现训练未见过的类别
    # ─────────────────────────────────────────────────────────────────────────
    # XGBoost 的 enable_categorical=True 在遇到新类别时会报错或用默认分支。
    # fit 阶段记录所有训练类别，transform 阶段强制对齐：
    #   推理中出现新类别 → 设为 NaN → 被树模型的默认分支处理
    # 这在"新学生来自一所冷门学校"这类场景下是安全的 fallback。
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_categorical_alignment(
        self, data: pd.DataFrame, is_fit: bool = False
    ) -> pd.DataFrame:
        for col in self.existing_categorical_columns:
            if col not in data.columns:
                continue

            if is_fit:
                if not pd.api.types.is_categorical_dtype(data[col]):
                    data[col] = data[col].astype("category")
                self.categorical_levels[col] = data[col].cat.categories
            else:
                data[col] = pd.Categorical(
                    data[col], categories=self.categorical_levels.get(col), ordered=False
                )
        return data

    # ─────────────────────────────────────────────────────────────────────────
    # 计数特征的 log1p 变换 + 99分位数 capping
    # ─────────────────────────────────────────────────────────────────────────
    # 经历计数（科研、奖项、实习、论文）的分布：大部分 0-2，极少数 5-10+
    # 问题1：右偏分布 → XGBoost (基于分位数的 tree) 对极端值不友好
    # 问题2：log(0) 不存在 → log1p(x) = log(1+x) 在 x=0 时 =0，合理
    # 问题3：可能有人填了"999段实习" → 99分位数 cap 防止数据污染
    #
    # 为什么是 99 分位而不是 95 或 99.9？
    #   95: 太严格，5% 的正常多段经历也会被切掉 → 信息损失
    #   99.9: 太宽松，接近 max → 几乎等于没 cap
    #   99: 保留 99% 的真实数据，去掉 top 1% 的异常值/输入错误
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_count_columns(self, data: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        for col in COUNT_COLUMNS_FOR_LOG_TRANSFORM:
            if col in data.columns:
                numeric_col_series = pd.to_numeric(data[col], errors="coerce")

                if is_fit and numeric_col_series.notna().any():
                    cap_value = numeric_col_series.quantile(0.99)
                    if pd.notna(cap_value):
                        self.cap_values[col] = float(cap_value)

                if col in self.cap_values:
                    data[col] = numeric_col_series.clip(upper=self.cap_values[col])

                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
                data[col] = np.log1p(data[col])

        return data

    # ─────────────────────────────────────────────────────────────────────────
    # 语言成绩统一归一化：TOEFL/120 与 IELTS/9 → 统一到 [0,1]
    # ─────────────────────────────────────────────────────────────────────────
    # 为什么合并为单一特征 language_score 而不是保留两个独立特征？
    # 1. 学生只考一种，两个特征大部分是缺失值 → 稀疏性问题
    # 2. 信息等价：TOEFL 和 IELTS 衡量同一能力（英语水平），
    #    只是量纲不同。归一化后语义统一。
    # 3. 两个都有的学生用 max（取最有利的成绩）— 对应实际审核中的
    #    "最好成绩"原则，学校通常接受最高分。
    #
    # 为什么除以 120/9 而不是更细的归一化？
    #   这是满分比例归一化（percentage of max），最直观：
    #   100/120 = 0.833, 7/9 = 0.778 → 直接可比
    #   z-score 归一化也可以，但会丢失绝对含金量信息
    #   (100/120 的 TOEFL 和 7/9 的 IELTS 在实际录取中确实有差异)
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_language_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        toefl_exists = "toefl" in data.columns
        ielts_exists = "ielts" in data.columns

        if toefl_exists and ielts_exists:
            toefl_norm = pd.to_numeric(data["toefl"], errors="coerce").fillna(0) / 120
            ielts_norm = pd.to_numeric(data["ielts"], errors="coerce").fillna(0) / 9
            data["language_score"] = np.maximum(toefl_norm, ielts_norm)
            data.drop(columns=["toefl", "ielts"], inplace=True, errors="ignore")
        elif toefl_exists:
            data["language_score"] = pd.to_numeric(data["toefl"], errors="coerce").fillna(0) / 120
            data.drop(columns=["toefl"], inplace=True, errors="ignore")
        elif ielts_exists:
            data["language_score"] = pd.to_numeric(data["ielts"], errors="coerce").fillna(0) / 9
            data.drop(columns=["ielts"], inplace=True, errors="ignore")

        return data

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        data = self._preprocess_data(df)
        data = self._handle_numeric_missing(data, is_fit=True)
        data = self._handle_count_columns(data, is_fit=True)
        data = self._handle_language_scores(data)
        self.existing_categorical_columns = [
            col for col in CATEGORICAL_COLUMNS if col in data.columns
        ]
        self._handle_categorical_alignment(data, is_fit=True)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = self._preprocess_data(df)
        data = self._handle_numeric_missing(data, is_fit=False)
        data = self._handle_count_columns(data, is_fit=False)
        data = self._handle_language_scores(data)
        data = self._handle_categorical_alignment(data, is_fit=False)
        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_state(self) -> dict:
        return {
            "numeric_medians": {k: float(v) for k, v in self.numeric_medians.items()},
            "cap_values": {k: float(v) for k, v in self.cap_values.items()},
            "existing_categorical_columns": list(self.existing_categorical_columns),
            "categorical_levels": {
                col: [str(level) for level in levels.tolist()]
                for col, levels in self.categorical_levels.items()
            },
        }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineer = FeatureEngineer()
    return engineer.fit_transform(df)
