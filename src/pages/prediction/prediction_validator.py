from typing import Optional

import streamlit as st

from src.pages.prediction.prediction_model import PredictionModel
from src.utils.logger import setup_logger

prediction_handler_logger = setup_logger("page3", "prediction")


def validate_model_and_features(prediction_model: Optional[PredictionModel]) -> Optional[list[str]]:
    if prediction_model is None:
        st.error("关键配置错误：无法加载预测模型。预测功能无法启动。请检查应用日志并联系管理员。")
        prediction_handler_logger.critical("预测函数无法继续：prediction_model 为 None。")
        return None

    if hasattr(prediction_model, "feature_names") and prediction_model.feature_names is not None:
        return prediction_model.feature_names
    else:
        prediction_handler_logger.error("模型特征列表为空，但模型已加载。这通常表示模型配置问题。")
        st.error("模型配置错误：特征列表为空。")
        return None

