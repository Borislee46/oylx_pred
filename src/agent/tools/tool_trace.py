import functools
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic_ai import RunContext

from src.agent.harness import HarnessDeps

F = TypeVar("F", bound=Callable[..., Any])


def traced(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(ctx: RunContext[HarnessDeps], *args: Any, **kwargs: Any):
        trace = getattr(ctx.deps, "trace", None)
        args_preview = {k: v for k, v in kwargs.items() if v not in (None, "", [])}
        entry = None
        if trace is not None:
            entry = trace.record(fn.__name__, args_preview=args_preview, result_preview="", ok=True)
        try:
            result = fn(ctx, *args, **kwargs)
        except Exception as exc:
            if entry is not None:
                entry.ok = False
                entry.result_preview = f"Error: {exc}"[:200]
            raise
        if entry is not None:
            text = result if isinstance(result, str) else str(result)
            entry.result_preview = text[:500]
        return result

    return wrapper
