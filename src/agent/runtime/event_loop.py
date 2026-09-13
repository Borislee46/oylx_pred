"""进程级常驻事件循环。

## 为什么需要它

`runtime.model_factory` 把 `OpenAIChatModel`（内含一个 `httpx.AsyncClient`）缓存在进程级
`_model_cache` 里，连接池中的 keep-alive 连接绑定在"首次使用它的事件循环"上。

如果每次调用都用 `asyncio.run(...)`（每轮新建并关闭一个 loop），下一轮复用连接池时会抛
`RuntimeError: Event loop is closed`。已实测：同一 client 连续三次 `asyncio.run`，
第 1、3 次成功、第 2 次失败 —— 表现为"偶发失败、重试就好"。

因此所有 agent 异步调用都提交到同一个常驻 loop，让 client 的连接池永远只有一个宿主。

## 用法

    from src.agent.runtime.event_loop import submit

    result = submit(coro, timeout=60.0)     # 阻塞当前线程，跑完返回

`timeout` 触发时抛内置 `TimeoutError`（3.11+ 起 `concurrent.futures.TimeoutError`
就是内置 `TimeoutError`），与仓库既有 `except TimeoutError` 捕获点兼容。
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_guard = threading.Lock()


def _run_forever(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            _log.debug("AGENT_LOOP | drain pending tasks failed", exc_info=True)
        try:
            loop.close()
        except Exception:
            _log.debug("AGENT_LOOP | close failed", exc_info=True)


def ensure_loop() -> asyncio.AbstractEventLoop:
    """返回常驻循环，必要时创建（线程安全）。"""
    global _loop, _thread
    with _guard:
        if (
            _loop is not None
            and not _loop.is_closed()
            and _thread is not None
            and _thread.is_alive()
        ):
            return _loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_run_forever,
            args=(loop,),
            name="agent-event-loop",
            daemon=True,
        )
        thread.start()
        _loop = loop
        _thread = thread
        _log.info("AGENT_LOOP | started")
        return loop


def submit(
    coro: Coroutine[Any, Any, T],
    *,
    timeout: float | None = None,
    on_tick: Callable[[], None] | None = None,
    tick_interval: float = 0.2,
) -> T:
    """在常驻循环上执行协程并阻塞当前线程等待结果。

    超时会把循环上的任务取消（`async with` / `finally` 正常收尾），再抛 `TimeoutError`。

    ## 为什么要有 on_tick

    常驻循环跑在**独立线程**上，所以协程里的回调（进度、日志）也都在那个线程执行。
    从那里写 `st.*` / `st.session_state` 会被 Streamlit 静默丢弃（没有 ScriptRunContext）。
    做法：循环线程只把进度写进普通数据结构，**等待方（脚本线程）在分片等待的空隙里回调**，
    由调用线程负责渲染 —— 这样 UI 写入永远发生在脚本线程上。

    `on_tick` 抛异常时会取消循环上的任务并把异常抛给调用方
    （UI 的"取消"就是靠这个：进度回调抛 `LeadInCancelled`）。
    """
    loop = ensure_loop()
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception:
        # 调度失败时协程从未被 await，主动关闭避免 "coroutine was never awaited"
        coro.close()
        raise

    deadline = None if timeout is None else time.monotonic() + timeout
    interval = max(0.05, float(tick_interval))

    while True:
        wait_for = interval
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                raise TimeoutError(f"agent run exceeded {timeout}s")
            wait_for = min(interval, remaining)

        try:
            return future.result(timeout=wait_for)
        except TimeoutError:
            # 区分"分片等待超时"和"协程自己抛的 TimeoutError"：
            # future 已完成说明是后者，原样抛出。
            if future.done():
                raise
            if deadline is not None and time.monotonic() >= deadline:
                future.cancel()
                raise
            if on_tick is not None:
                try:
                    on_tick()
                except BaseException:
                    future.cancel()
                    raise


def shutdown() -> None:
    """停止常驻循环（测试与进程退出用）。"""
    global _loop, _thread
    with _guard:
        loop, thread = _loop, _thread
        _loop = None
        _thread = None
    if loop is None:
        return
    try:
        loop.call_soon_threadsafe(loop.stop)
    except Exception:
        _log.debug("AGENT_LOOP | stop failed", exc_info=True)
    if thread is not None and thread.is_alive():
        thread.join(timeout=3.0)
    _log.info("AGENT_LOOP | stopped")


atexit.register(shutdown)
