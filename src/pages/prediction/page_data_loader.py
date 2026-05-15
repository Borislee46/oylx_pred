"""
模型资源加载器 — 全局单例，所有用户共享。

machine_learning_model 是整个预测系统的"数据总线"：
  - cases_df:         61,716 行历史案例（训练数据）
  - prediction_model: XGBoost 模型实例
  - loaded_feature_names: 模型期望的特征名列表（顺序敏感）
  - cases_df_fingerprint: 数据版本哈希（用于一致性校验）
  - background_universities: 模型可识别的院校白名单
  - target_base_df: 目标院校-专业基础表
  - university_country_map: 院校 → 国家/地区映射

resource_loader() 使用 @st.cache_resource + scope="global"：
  - 首次调用：加载模型 + 数据（~3-5s）
  - 后续调用：返回缓存实例（~0ms）
  - scope="global"：所有用户的 session 共享同一实例（内存节省 ~200MB/user）
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.modeling.model import PredictionModel
from src.utils.app_data_loader import (
    load_global_categories_dataframe,
    load_raw_cases_data,
)


def get_prediction_model(model_name):
    """加载 XGBoost 模型实例（未缓存，每次调用重新加载）。"""
    global_categories_df_instance = load_global_categories_dataframe()
    model_instance = PredictionModel(model_name, global_categories_df=global_categories_df_instance)
    return model_instance


@st.cache_resource(show_spinner=False, scope="global")
def cached_get_prediction_model(model_name):
    """Streamlit 缓存的模型加载（全局共享 + 不显示 spinner）。"""
    return get_prediction_model(model_name)


@dataclass
class machine_learning_model:
    """预测系统的数据总线 dataclass。

    字段：
    - prediction_model: XGBoost 模型实例（含 predict_proba 方法）
    - loaded_feature_names: 模型训练时的特征列名（顺序必须完全一致）
    - cases_df: 完整历史案例 DataFrame（用于院校/专业查找、fallback）
    - cases_df_fingerprint: cases_df 的哈希指纹（跨模块一致性校验）
    - background_universities: 已知背景院校集合（用于归一化阶段的模糊匹配）
    - target_base_df: 目标院校-专业基础信息表
    - university_country_map: 院校名 → 地区映射（用于海本语言豁免等）
    """
    prediction_model: Any
    loaded_feature_names: list[str]
    cases_df: pd.DataFrame
    cases_df_fingerprint: int
    background_universities: set[str]
    target_base_df: pd.DataFrame
    university_country_map: dict[str, str]

    @classmethod
    @st.cache_resource(show_spinner=False, scope="global")
    def resource_loader(cls) -> "machine_learning_model":
        """全局单例加载器。

        加载顺序（不可变）：
        1. Warm-up: SchoolMajorDataManager + 背景-目标相似度缓存
        2. 模型加载: XGBoost .ubj 模型 + 特征名校验
        3. 数据加载: cases_df + 院校白名单
        4. 目标基础表构建: 院校-专业组合 + 国家映射

        为什么预热（warm_up）？
          SchoolMajorDataManager 的首次访问涉及大量 pandas 操作（groupby、索引构建）。
          在资源加载阶段预热可避免首次用户请求时的延迟峰值。
        """
        from src.pages.prediction.core.utils import _data_manager
        from src.pages.prediction.input_form_components.target_options_service import (
            build_target_base_df,
        )
        from src.pages.prediction.modeling import validate_model_and_features
        from src.pages.prediction.prediction_preparation import (
            compute_df_fingerprint,
        )
        from src.utils.app_data_loader import (
            load_bg_target_similarity_cache,
            load_school_major_details_df,
        )

        # 1. Warm-up 预热
        _data_manager.warm_up()
        load_bg_target_similarity_cache()  # 高峰用户预加载相似度矩阵

        # 2. 模型加载
        model = cached_get_prediction_model("xgboost")
        features = validate_model_and_features(model)
        cases = load_raw_cases_data()
        feature_list = features if features is not None else []

        # 3. 数据指纹 + 院校白名单
        fingerprint = compute_df_fingerprint(cases)
        bg_unis = (
            set(cases["background_university"].dropna().astype(str).unique())
            if "background_university" in cases.columns
            else set()
        )

        # 4. 目标基础表
        details_df = load_school_major_details_df()
        unique_targets_df = None
        if cases is not None and not cases.empty:
            cols = [c for c in ["target_university", "target_major"] if c in cases.columns]
            if cols:
                unique_targets_df = cases[cols].drop_duplicates()

        target_base_df, uni_country_map = build_target_base_df(unique_targets_df, details_df)

        return cls(
            prediction_model=model,
            loaded_feature_names=feature_list,
            cases_df=cases,
            cases_df_fingerprint=fingerprint,
            background_universities=bg_unis,
            target_base_df=target_base_df,
            university_country_map=uni_country_map,
        )
