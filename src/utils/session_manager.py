"""
SessionManager — Streamlit session_state 的类型安全封装。

设计动机：
  Streamlit 的 st.session_state 是一个无类型的全局 dict。
  直接使用存在三个问题：
  1. key 拼写错误在运行时报错，IDE 无法检测
  2. 无法追踪"哪些 key 被哪些模块读写"
  3. 没有默认值机制，每次 get 都要写 default=xxx

解决方案：
  UserDataModel dataclass 定义已知字段（强类型），other_states dict 容纳动态扩展。
  SessionManager 是统一的读写接口：set/get/delete/prune/clear。

两层存储：
  UserDataModel 字段   → 预定义的、常用的状态（has_predicted, prediction_results 等）
  UserDataModel.other_states → 动态扩展的状态（所有未预定义的 key 都存在这里）

设计取舍：
  - 为什么不用全字段 dataclass？预测 pipeline 中不断需要新增临时 key，
    每次加 key 都要改 UserDataModel 会非常繁琐。
  - 为什么不用纯 dict？丢失类型安全，IDE 无法补全预定义字段的 set/get。
  - 当前方案是两种模式的折中：高频核心字段在 dataclass 中有类型提示，
    临时/实验性 key 走 other_states。
"""

import uuid
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

import streamlit as st


# ── PredictionResultModel：预测结果的结构化容器 ───────────
# 预测结果的完整结构：三个来源 + 合并后的统一结果 + 元数据。
# 使用 dataclass 而非 dict 是为了字段级别的类型提示。
@dataclass
class PredictionResultModel:
    similarity_results: list[dict] | None = None       # 相似度匹配的结果列表
    cross_major_results: list[dict] | None = None      # 跨专业匹配的结果列表
    user_specified_results: list[dict] | None = None   # 用户指定组合的结果列表
    unified_results: list[dict] | None = None          # 去重合并后的最终结果列表
    meta: dict[str, Any] | None = None                 # 元数据（运行时间、模型版本等）


# ── UserDataModel：用户会话数据的结构化容器 ───────────────
# 每个 Streamlit session 对应一个 UserDataModel 实例。
# 存储在 st.session_state["user_data_model"] 中。
#
# 字段分为三级：
#   Tier 1（预定义 dataclass 字段）— 高频核心状态，含类型提示
#   Tier 2（other_states dict）     — 动态扩展的临时/实验性 key
# 新增临时 key 时走 Tier 2，稳定后可以提升到 Tier 1。
@dataclass
class UserDataModel:
    # session_id 在每次创建时自动生成 UUID，用于日志关联和用户追踪
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_logged_in: bool = False                         # E2 登录状态
    user_info: dict[str, Any] = field(default_factory=dict)  # E2 返回的用户信息
    input_data: dict[str, Any] | None = None           # 当前表单输入数据
    prediction_results: PredictionResultModel = field(default_factory=PredictionResultModel)  # 预测结果容器
    prediction_submit_lock: bool = False               # 提交互斥锁
    other_states: dict[str, Any] = field(default_factory=dict)  # 动态扩展状态桶


# ── SessionManager ─────────────────────────────────────────
class SessionManager:
    """Streamlit session_state 的类型安全封装。

    用法：
        sm = SessionManager()
        sm.set(hk_ui_phase="running", has_predicted=True)
        phase = sm.get("hk_ui_phase", "idle")
        sm.delete("hk_last_error")
        sm.prune_states(["pending_", "last_"])

    特性：
    - 幂等实例化：SessionManager() 可安全地多次调用，多个实例共享同一底层 store
    - 分层读写：预定义字段走 getattr/setattr，动态 key 走 other_states
    - 默认值：get() 支持 default 参数（与 dict.get 语义一致）
    - 批量清理：prune_states() 按前缀批量删除 other_states 中的 key
    """

    _STATE_KEY = "user_data_model"                     # st.session_state 中的存储 key
    _MODEL_FIELDS = {f.name for f in fields(UserDataModel)}  # 预定义字段名集合（O(1) 查找）

    # ── 字段默认值表 ──────────────────────────────────────
    # 在类加载时计算一次，delete() 时用于恢复默认值。
    # __func__ 绕过 staticmethod descriptor 取原始函数。
    @staticmethod
    def _build_field_defaults() -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for f in fields(UserDataModel):
            if f.default is not MISSING:
                defaults[f.name] = f.default
            elif f.default_factory is not MISSING:
                defaults[f.name] = f.default_factory()
            else:
                defaults[f.name] = None
        return defaults

    _FIELD_DEFAULTS = _build_field_defaults.__func__()  # type: ignore[attr-defined]

    def __init__(self) -> None:
        # 幂等：只在 Streamlit session 首次创建时初始化 UserDataModel
        if self._STATE_KEY not in st.session_state:
            st.session_state[self._STATE_KEY] = UserDataModel()
            st.session_state["session_id"] = self.session_id_short

    @property
    def _model(self) -> UserDataModel:
        """内部访问：直接返回 UserDataModel 实例（非防御性拷贝，调用方负责不滥用）"""
        return st.session_state[self._STATE_KEY]

    @property
    def session_id_short(self) -> str:
        """返回 session UUID 的前 8 位，用于日志显示。"""
        return self._model.session_id[:8]

    # ── 读写 API ──────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """读取 key 对应的值。

        查找顺序：UserDataModel 预定义字段 → other_states dict → default。
        与 dict.get 语义一致：不存在时返回 default 而非抛异常。
        """
        if key in self._MODEL_FIELDS:
            return getattr(self._model, key)
        return self._model.other_states.get(key, default)

    def set(self, **kwargs: Any) -> None:
        """写入一个或多个 key-value 对。

        key 在 UserDataModel 预定义字段中 → setattr
        key 不在预定义字段中 → 存入 other_states dict
        支持批量写入：sm.set(a=1, b=2, c=3)
        """
        for key, value in kwargs.items():
            if key in self._MODEL_FIELDS:
                setattr(self._model, key, value)
            else:
                self._model.other_states[key] = value

    def batch_set(self, updates: dict[str, Any]) -> None:
        """dict 形式的批量写入，等价于 set(**updates)。"""
        self.set(**updates)

    def delete(self, key: str) -> None:
        """删除 key 对应的值。

        other_states 中的 key → 真删除（del）
        预定义字段（session_id 除外）→ 恢复为默认值（而非删除，因为 dataclass 字段必须存在）
        session_id → 不可删除（忽略）
        """
        if key in self._model.other_states:
            del self._model.other_states[key]
        elif key in self._MODEL_FIELDS and key != "session_id":
            setattr(self._model, key, self._FIELD_DEFAULTS.get(key))

    def clear_session(self) -> None:
        """清除当前会话的所有状态（保留 session_id）。

        用于"重新开始"场景：清除所有预测、表单、UI 状态，
        但保持登录状态和 session_id 不变。
        """
        current_id = self._model.session_id
        st.session_state[self._STATE_KEY] = UserDataModel(session_id=current_id)
        st.session_state["session_id"] = current_id[:8]

    def prune_states(self, prefixes: list[str]) -> int:
        """删除 other_states 中所有以指定前缀开头的 key。

        用于批量清理临时状态。例如：
          sm.prune_states(["pending_", "last_", "lead_in_"])
        清除所有 pending_*、last_*、lead_in_* key。

        Returns:
            int: 删除的 key 数量。
        """
        to_delete = [k for k in self._model.other_states if any(k.startswith(p) for p in prefixes)]
        for k in to_delete:
            del self._model.other_states[k]
        return len(to_delete)

    def get_widget_value(self, key: str, default: Any = None) -> Any:
        """直接读取 st.session_state 中的 widget value（不走 UserDataModel）。

        Streamlit widget 的值通过 widget key 直接存在 st.session_state 中，
        不在 UserDataModel 内。此方法提供统一的读取入口。
        """
        return st.session_state.get(key, default)

    def ensure_user_info(self) -> dict[str, Any]:
        """返回缓存的用户信息，未同步时返回空 dict。

        Auth→model 同步在 init_page() 中完成一次。
        此方法是纯读取，不会触发重新认证。
        判断条件：user_info 中是否有 username 字段（而非整个 dict 是否为空）。
        """
        return self._model.user_info if self._model.user_info.get("username") else {}
