"""
预测结果管理 + 会话初始化 + 结果合并去重。

职责：
  1. initialize_session_states — 首次页面加载时初始化所有 session_state（幂等）
  2. reset_prediction_results   — 重置预测结果（新提交前调用）
  3. clear_pending_prediction_state — 清理跨学部确认/重试的临时状态
  4. combine_and_deduplicate_results — 合并三个来源的预测结果并去重
"""

from src.pages.prediction.handler_config import DEFAULT_UI_KEYS
from src.utils.session_manager import PredictionResultModel, SessionManager


# ── 预测结果重置 ───────────────────────────────────────────
def reset_prediction_results(session_manager: SessionManager):
    """清空预测结果和相关状态，准备新一轮预测。

    调用时机：用户点击"开始预测"或"AI 提取"触发自动提交前。
    prune_states 清除所有 pending_/last_/lead_in_ 前缀的临时 key，
    确保上次预测的中间状态不会污染新一轮。
    """
    session_manager.set(
        has_predicted=False,
        prediction_results=PredictionResultModel(),  # 空模型实例
        prediction_submit_lock=False,
        processing_lock=False,
        lock_start_time=0,
    )
    session_manager.prune_states(["pending_", "last_", "lead_in_"])


# ── 跨学部状态清理 ─────────────────────────────────────────
def clear_pending_prediction_state(
    session_manager: SessionManager,
    *,
    reset_cross_faculty_confirmed: bool = False,
    reset_cross_faculty_cancelled: bool = False,
):
    """清除跨学部确认流程的临时状态。

    跨学部确认流程会产生以下临时状态：
    - pending_cross_faculty_prediction: 标记"有等待用户确认的跨学部预测"
    - pending_prediction_data: 保存原始提交数据（用户确认后重新提交用）
    - cross_faculty_confirmed: 用户已确认
    - cross_faculty_cancelled: 用户已取消

    reset_* 参数允许选择性清除确认/取消 flag：
    - 正常新提交：两者都不 reset（保留确认状态不影响新提交）
    - 异常恢复/跨学院重试：根据场景 reset 对应 flag
    """
    updates = {
        DEFAULT_UI_KEYS.pending_cross_faculty_prediction: False,
        DEFAULT_UI_KEYS.pending_prediction_data: None,
    }
    if reset_cross_faculty_confirmed:
        updates[DEFAULT_UI_KEYS.cross_faculty_confirmed] = False
    if reset_cross_faculty_cancelled:
        updates[DEFAULT_UI_KEYS.cross_faculty_cancelled] = False
    session_manager.set(**updates)


# ── 多源结果合并去重 ───────────────────────────────────────
def combine_and_deduplicate_results(sim_results, cross_results, user_specified_results):
    """合并三个预测来源的结果，按优先级去重。

    三个来源及优先级（数字越小 → 越优先）：
    1. similarity（相似度匹配）：pipeline 推的最相似专业组合
    2. cross_major（跨专业）：用户选的跨专业组合（仅 admitted=1 的）
    3. user_specified（用户指定）：用户手动指定的组合

    去重逻辑：
    - 同一个 (university, major) 组合 → 保留优先级最高的
    - 同优先级 → 保留概率更高的
    - 最终输出时移除内部 _source / _priority 标记字段
    """
    # (数据, 来源标签, 优先级, 过滤函数)
    # cross_major 的 filter_fn 只保留已有录取记录的组合
    sources = [
        (sim_results, "similarity", 1, None),
        (cross_results, "cross_major", 2, lambda r: r.get("admitted", 0) == 1),
        (user_specified_results, "user_specified", 3, None),
    ]

    unique_results = {}

    for results, source, priority, filter_fn in sources:
        for result in results or []:
            if filter_fn and not filter_fn(result):
                continue

            # 去重 key：(院校名, 专业名) 二元组
            key = (result.get("university"), result.get("major"))
            new_prob = result.get("probability", 0.0) or 0.0

            existing = unique_results.get(key)
            if not existing:
                # 首次出现 → 直接加入，附带来源标记
                unique_results[key] = {**result, "_source": source, "_priority": priority}
            else:
                existing_priority = existing.get("_priority", 0)
                existing_prob = existing.get("probability", 0.0) or 0.0

                # 更高优先级 或 同优先级更高概率 → 替换
                if priority > existing_priority or (
                    priority == existing_priority and new_prob > existing_prob
                ):
                    unique_results[key] = {**result, "_source": source, "_priority": priority}

    # 移除内部标记字段后返回
    return [
        {k: v for k, v in res.items() if not k.startswith("_")} for res in unique_results.values()
    ]


# ── 会话状态初始化（幂等）─────────────────────────────────
def initialize_session_states(session_manager: SessionManager):
    """首次页面加载时初始化所有 session_state 默认值。

    幂等设计：app_initialized flag 确保只执行一次。
    即使 prediction_page_content 每次重跑都调用此函数，
    也只有第一次会真正执行初始化。

    初始化的值覆盖：
    - 预测状态（has_predicted, processing_lock, prediction_submit_lock）
    - 输入缓存（input_data）
    - 跨学部状态（cross_faculty_* flags）
    - 状态机（hk_ui_phase="idle", hk_run_id=None, hk_last_error=None）
    """
    if session_manager.get(DEFAULT_UI_KEYS.app_initialized):
        return

    session_manager.set(
        has_predicted=False,
        is_school_selection_submit=False,
        processing_lock=False,
        prediction_submit_lock=False,
        input_data=None,
        cross_faculty_confirmed=False,
        cross_faculty_cancelled=False,
        pending_cross_faculty_prediction=False,
        pending_prediction_data=None,
        hk_ui_phase="idle",
        hk_run_id=None,
        hk_last_error=None,
        app_initialized=True,
    )
