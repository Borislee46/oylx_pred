"""
输入表单组件包 — 预测页面表单的完整 UI + 状态 + 校验基础设施。

架构分层（由底向上）:
  Layer 0: form_config        — 静态配置常量（GPA分制、语言分数范围、国家-院校映射）
  Layer 1: gpa_converter      — GPA异制转换规则引擎（JSON配置驱动）
  Layer 1: language_score_converter — 托福↔雅思互转查找表
  Layer 2: form_state         — 表单状态机（session_state 读写、变更检测、自动保存快照）
  Layer 2: form_validator     — 提交前校验（GPA归一化、标化分范围、经历count/detail一致性）
  Layer 2: widget_helpers     — SelectBox/Multiselect 封装（缓存选项生成、防御性回退）
  Layer 3: *_ui               — 各表单区块的 Streamlit 渲染函数（gpa_ui, language_ui, ...）
  Layer 4: form_ui            — 薄编排层，组装所有区块 + 提交按钮

数据流:
  用户交互 → st.widget (on_change callback)
    → FormStateManager.on_form_change()     # 标记 dirty、清除锁定、写自动保存快照
    → SessionManager.set() / batch_set()    # 持久化到 st.session_state
    → 提交时 FormValidator.validate_form_data()  # 校验 → 通过后进入预测流水线

设计约束:
  - Streamlit 无响应式数据绑定 → 所有状态变更必须显式写 session_state
  - on_change 回调中不能用 st.toast（会丢）→ 警告逻辑放在 render 阶段
  - widget key 全局唯一 → 通过 handler_config.DEFAULT_WIDGET_KEYS 集中管理
"""

from src.pages.prediction.input_form_components.form_config import (
    DEFAULT_GPA_SCALE,
    GPA_SCALES,
    LANGUAGE_SCORE_RANGES,
    LANGUAGE_TYPES,
)
from src.pages.prediction.input_form_components.form_state import FormStateManager
from src.pages.prediction.input_form_components.form_ui import FormUIComponents
from src.pages.prediction.input_form_components.form_validator import FormValidator, ValidationError
from src.pages.prediction.input_form_components.gpa_converter import GPAConverter

__all__ = [
    "GPA_SCALES",
    "DEFAULT_GPA_SCALE",
    "LANGUAGE_TYPES",
    "LANGUAGE_SCORE_RANGES",
    "FormStateManager",
    "FormValidator",
    "FormUIComponents",
    "GPAConverter",
    "ValidationError",
]
