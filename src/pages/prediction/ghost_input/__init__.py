"""
Ghost Text Autocomplete — 浏览器端 DeepSeek 前缀续写组件。

架构：
  这是一个自定义 Streamlit 组件（st.components.v1.declare_component）。
  前端代码在 frontend/ 目录中（HTML/JS/CSS），后端此文件是 Python 桥接层。

  数据流（与普通 Streamlit widget 不同）：
    用户在 textarea 中输入 → 前端直接调 DeepSeek API（绕过 Streamlit server）
    → 返回续写建议（前端渲染 ghost text）→ 用户接受/忽略
    → 仅当用户完成编辑（blur）或点击 AI 按钮（analyze）时，才回传结果到 Python

  为什么前端直接调 DeepSeek？
    1. 延迟：每个字符都走 Streamlit → DeepSeek → Streamlit 会有 300ms+ RTT
    2. 成本：每次重跑 Streamlit 脚本会触发不必要的重渲染
    3. 用户体验：自动补全需要 <100ms 响应，只有浏览器直连能做到

  限流：
    rate_max + rate_window_ms：前端在时间窗口内最多发起 rate_max 次 API 请求
    rate_cooldown_ms：触发限流后的冷却时间

  Action 类型：
    - blur：用户离开 textarea（失焦）
    - accept：用户接受了 ghost 建议
    - sync：定期同步（仅更新文本，不触发后续处理）
    - analyze：用户在 iframe 内点击了 AI 提取按钮
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

_FRONTEND_DIR = Path(__file__).parent / "frontend"


# ── 目标院校白名单（来源：prediction_rules.json）──────────
def _load_allowed_universities() -> list[str]:
    """从配置文件加载认可的院校列表 + 常用别名。

    白名单用于前端的院校名匹配（ghost 续写时高亮/联想）。
    别名覆盖用户常用的非标准写法（如"港大"而非"香港大学"）。
    """
    try:
        rules_path = Path(__file__).parent.parent.parent.parent / "config" / "prediction_rules.json"
        if rules_path.exists():
            import json

            rules = json.loads(rules_path.read_text(encoding="utf-8"))
            unis = rules.get("UNIVERSITY_DISPLAY_ORDER", [])
            aliases = [
                "港大", "港中文", "港科大", "港理工", "港城市", "港浸会", "港岭南",
                "港教育", "港都会", "港恒生", "港珠海",
                "新国立", "南洋理工", "新管理",
                "NUS", "NTU", "HKU", "CUHK", "HKUST", "CityU", "PolyU", "SMU",
            ]
            return unis + aliases
    except Exception:
        pass
    return []


# ── 组件声明（模块加载时执行一次）────────────────────────
# declare_component 通过 path 参数指向 frontend/ 目录，
# Streamlit 将目录中的 index.html 作为 iframe 渲染。
# 前端通过 Streamlit.component_api API 与 Python 双向通信。
_component = st.components.v1.declare_component(
    "ghost_text_area",
    path=str(_FRONTEND_DIR),
)
_log = logging.getLogger("ghost_input")


def ghost_text_area(
    api_key: str = "",
    api_base_url: str = "https://api.deepseek.com/beta",
    api_model: str = "deepseek-v4-flash",
    placeholder: str = "",
    initial_text: str = "",
    height: int = 100,
    rate_limit_max: int = 30,
    rate_limit_window_seconds: int = 60,
    rate_limit_cooldown_seconds: int = 15,
    key: str | None = None,
) -> str:
    """渲染 ghost-autocomplete textarea 组件。

    组件生命周期：
    1. 首次渲染：创建 iframe，传入 config（含 API key、白名单等）
    2. 用户输入：前端直接调 DeepSeek 获取续写建议（无 Streamlit 往返）
    3. 用户动作（blur/accept/analyze）：前端通过 Streamlit.setComponentValue 回传结果
    4. Python 侧收到结果 → 更新 session_state → 返回当前文本

    Args:
        api_key: DeepSeek API key（从 app_config 读取）
        api_base_url: API 端点（默认 DeepSeek beta）
        api_model: 模型名（deepseek-v4-flash 用于快速补全）
        placeholder: textarea 占位符文本
        initial_text: 初始文本
        height: 组件高度（px）
        rate_limit_max: 时间窗口内最大 API 请求数
        rate_limit_window_seconds: 限流窗口（秒）
        rate_limit_cooldown_seconds: 限流触发后冷却时间（秒）
        key: Streamlit 组件 key（用于 session_state 标识）

    Returns:
        用户编辑的文本（仅当 action 为 blur/accept/analyze/sync 时返回非空）
        返回 "" 表示无用户动作需要处理（避免不必要的重跑）
    """
    config = {
        "api_key": api_key,
        "api_base_url": api_base_url.rstrip("/"),
        "api_model": api_model,
        "initial_text": initial_text,
        "placeholder": placeholder,
        "rows": max(3, height // 24),   # 行数 = 高度 / 行高（约 24px）
        "height": height + 40,          # 额外 40px 给 ghost suggestion bar
        "rate_max": rate_limit_max,
        "rate_window_ms": rate_limit_window_seconds * 1000,     # 前端用毫秒
        "rate_cooldown_ms": rate_limit_cooldown_seconds * 1000,
        "allowed_universities": _load_allowed_universities(),
        "allowed_regions": ["香港", "新加坡", "澳门", "马来西亚"],
    }

    result = _component(
        config=config,
        default=None,
        key=key,
    )

    _log.debug(
        "COMPONENT_RAW | result_type=%s result=%s",
        type(result).__name__,
        repr(result)[:200] if result is not None else "None",
    )

    # 处理组件返回的动作
    if isinstance(result, dict) and result.get("action") in ("blur", "accept", "sync", "analyze"):
        # 去重：同一个 event_id 不处理两次（Streamlit 重跑可能重复触发）
        event_id = str(result.get("event_id") or "")
        if event_id:
            last_event_key = f"_ghost_last_event_id:{key or 'default'}"
            if st.session_state.get(last_event_key) == event_id:
                return ""
            st.session_state[last_event_key] = event_id

        # sync 动作不算用户交互，不记录 telemetry（避免日志噪音）
        if result.get("action") not in ("sync",):
            _log_telemetry(result)
        text = str(result.get("text") or "")
        # analyze 动作：iframe 内的 AI 按钮被点击
        # → 存储文本到 _ghost_analyze_text → render_lead_in_actions 消费
        if result.get("action") == "analyze" and text:
            st.session_state["_ghost_analyze_text"] = text
            _log.info("GHOST_ANALYZE | text_len=%d text=%s", len(text), text[:120])
        return text
    return ""


def _log_telemetry(result: dict) -> None:
    """记录 ghost_input 前端遥测数据。

    指标包括：
    - fetch_attempt/ok/fail/retry：DeepSeek API 请求统计
    - cache_hit/cache_set：前端缓存命中率
    - rule_hit/rule_miss：规则匹配（院校名联想）
    - suggestion_shown/accepted/dismissed：用户对建议的接受率
    - rate_limited/dedup_blocked：限流和去重触发次数

    高错误率（>50%）时发出 warning，帮助排查 API 问题。
    """
    telemetry = result.get("telemetry")
    if not telemetry:
        return
    counters = telemetry.get("counters", {})
    _log.info(
        "TELEMETRY | session=%s duration=%ss "
        "attempts=%d ok=%d fail=%d retry=%d "
        "cache_hit=%d cache_set=%d "
        "rule_hit=%d rule_miss=%d "
        "shown=%d accepted=%d dismissed=%d "
        "rate_limited=%d dedup=%d",
        telemetry.get("session", "?"),
        telemetry.get("duration_s", "?"),
        counters.get("fetch_attempt", 0),
        counters.get("fetch_ok", 0),
        counters.get("fetch_fail", 0),
        counters.get("fetch_retry", 0),
        counters.get("cache_hit", 0),
        counters.get("cache_set", 0),
        counters.get("rule_hit", 0),
        counters.get("rule_miss", 0),
        counters.get("suggestion_shown", 0),
        counters.get("suggestion_accepted", 0),
        counters.get("suggestion_dismissed", 0),
        counters.get("rate_limited", 0),
        counters.get("dedup_blocked", 0),
    )
    ok = counters.get("fetch_ok", 0)
    fail = counters.get("fetch_fail", 0)
    total = ok + fail
    if total > 0 and fail / total > 0.5:
        _log.warning(
            "TELEMETRY | high error rate session=%s ok=%d fail=%d",
            telemetry.get("session", "?"),
            ok,
            fail,
        )
