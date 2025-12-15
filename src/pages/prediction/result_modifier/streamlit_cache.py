"""
惰性加载，测试用
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def _noop_decorator[P: ParamSpec, R](func: Callable[P, R]) -> Callable[P, R]:
    return func


def _has_streamlit_runtime(st: Any) -> bool:
    runtime = getattr(st, "runtime", None)
    exists = getattr(runtime, "exists", None)
    return bool(callable(exists) and exists())


def cache_data(*args: Any, **kwargs: Any):
    try:
        import streamlit as st

        if _has_streamlit_runtime(st):
            return st.cache_data(*args, **kwargs)
    except ImportError:
        pass

    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        return _noop_decorator(func)

    return decorator


def cache_resource(*args: Any, **kwargs: Any):
    try:
        import streamlit as st

        if _has_streamlit_runtime(st):
            return st.cache_resource(*args, **kwargs)
    except ImportError:
        pass

    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        return _noop_decorator(func)

    return decorator
