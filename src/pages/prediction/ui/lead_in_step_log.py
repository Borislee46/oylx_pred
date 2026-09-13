"""原生「步骤日志」：把工具调用/思考步骤增量渲染进一个 st.status。

## 为什么用原生而不是现有 React 组件

见 `docs/agent_module_review_20260911.md` §11。要点：

- 自定义组件每次**属性变化都会重挂载**（iframe 重建），WebGL/粒子/轨道动画随之重置。
  它原本的设计是"挂载一次 + SSE 自更新"，而 SSE 从未启用（`LEAD_IN_SSE_SAME_ORIGIN` 没配），
  于是只能靠重挂载驱动 → 动画反复重置，比静止更难看。
- 原生元素可以**原地增量追加**，且进度写入现在发生在脚本线程上
  （`runtime.event_loop.submit(on_tick=...)`），所以原生路径是安全且正确的。
- 主题、暗色、可访问性、文本复制、`st.json`/`st.code` 展示参数与返回，全部免费。

职责划分：**这个类只负责"发生了什么"（高频、可读、可展开）**；
"正在为你解读"那种氛围留给组件（挂载一次、不随进度重渲染）。
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

_log = logging.getLogger(__name__)

_DEFAULT_TITLE = "AI 正在解读"
_MAX_RAW_CHARS = 1200


class StepLog:
    """增量步骤日志。

    用法（必须在**脚本线程**上创建与绘制）：

        log = StepLog()
        log.draw(["正在解析学生背景", "已写入：院校 中山大学"])
        log.label("正在检索院校专业库…")
        log.complete()
    """

    def __init__(self, *, title: str = _DEFAULT_TITLE, expanded: bool = True) -> None:
        self._status = st.status(title, expanded=expanded, type="step")
        # container 可以追加子元素；st.empty() 只能替换，不能追加
        self._box = st.container()
        # 模型正在生成的回复：整体替换，所以用 empty
        self._live = st.empty()
        self._drawn: list[str] = []
        self._title = title

    # ── 绘制 ──────────────────────────────────────────────────────

    def draw(self, details: list[str] | None) -> int:
        """把 details 里**新增的**行追加到日志里，返回本次追加条数。

        `details` 来自 `entries_to_detail_lines`（每步工具调用一行，含工具标签）。
        **只画人话，不 dump 原始参数与返回** —— 用户不会展开看，摆出来只会像测试日志。
        """
        lines = [str(x).strip() for x in (details or []) if str(x).strip()]
        new_lines = lines[len(self._drawn) :]
        if not new_lines:
            return 0
        for line in new_lines:
            with self._box:
                st.markdown(f"- {line}")
        self._drawn = lines
        return len(new_lines)

    def live(self, text: str) -> None:
        """模型正在生成的回复（整体替换，不是追加）。"""
        clean = (text or "").strip()
        if clean:
            self._live.markdown(clean)

    def raw_details(self, entries: list[Any] | None) -> None:
        """**折叠**的原始调用明细：每次调用的函数名、参数、返回。

        极客层（`LEAD_IN_GEEK_UI=0` 可整体关闭）。默认折叠 —— 终端用户不会点开，
        但对排障、对"我到底调了什么"很有用，所以保留而不是删掉。
        """
        from src.agent.step_text import geek_enabled

        if not entries or not geek_enabled():
            return

        def _get(raw: Any, key: str, default: Any = None) -> Any:
            return raw.get(key, default) if isinstance(raw, dict) else getattr(raw, key, default)

        rows = [e for e in entries if _get(e, "tool") != "router_stage"]
        if not rows:
            return

        with self._box, st.expander(f"原始调用明细（{len(rows)} 次）", expanded=False):
            for i, raw in enumerate(rows, 1):
                tool = str(_get(raw, "tool", "") or "")
                ms = float(_get(raw, "duration_ms", 0.0) or 0.0)
                ok = bool(_get(raw, "ok", True))
                head = f"{i}. `{tool}`" + (f" · {ms:.0f}ms" if ms >= 1 else "")
                if not ok:
                    head += " · 失败"
                st.caption(head)
                args = dict(_get(raw, "args_preview", {}) or {})
                if args:
                    st.json(args)
                result = str(_get(raw, "result_preview", "") or "")
                if result:
                    st.code(result[:_MAX_RAW_CHARS], language="json")

    # ── 状态 ──────────────────────────────────────────────────────

    def label(self, text: str, *, state: str = "running") -> None:
        self._status.update(label=text or self._title, state=state, expanded=True)

    def complete(self, label: str | None = None) -> None:
        n = len(self._drawn)
        final = label or (f"处理完成 · {n} 步" if n else "处理完成")
        self._status.update(label=final, state="complete", expanded=False)

    def fail(self, label: str | None = None) -> None:
        self._status.update(label=label or "处理过程中出现问题", state="error", expanded=True)

    @property
    def step_count(self) -> int:
        return len(self._drawn)
