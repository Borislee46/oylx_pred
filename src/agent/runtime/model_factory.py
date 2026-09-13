import atexit
import hashlib
import logging
import threading

import httpx
from pydantic_ai.models import get_user_agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config.settings import current_env, load_app_config

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "deepseek-v4-flash"
_DEFAULT_BASE = "https://api.deepseek.com"
DEFAULT_MODEL_TIMEOUT = 60
# 兜底的单轮预算：必须 >= 模型自身 HTTP 超时，否则外层会先掐断内层请求，
# 让 FallbackModel 连"换模型重试"的机会都没有。真实预算优先取
# HarnessConfig.turn_budget_ms（见 tool_agent._effective_turn_timeout）。
LEAD_IN_TOOL_STREAM_TIMEOUT = 60

_DEFAULT_TIMEOUT = DEFAULT_MODEL_TIMEOUT

_model_cache: dict[tuple[str, str, str, int], OpenAIChatModel] = {}
_cache_lock = threading.Lock()


def resolve_openai_base_url(raw: str) -> str:
    base = (raw or "").rstrip("/")
    suffix = "/chat/completions"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base or _DEFAULT_BASE


def build_production_model(*, timeout: int = _DEFAULT_TIMEOUT) -> OpenAIChatModel:
    cfg = load_app_config()
    model_name = cfg.get("OPEN_AI_MODEL", _DEFAULT_MODEL)
    api_base = resolve_openai_base_url(cfg.get("OPEN_AI_BASE_URL", ""))
    api_key = cfg.get("OPEN_AI_API_KEY", "")
    return _build_model(model_name, api_base, api_key, timeout=timeout)


def build_fallback_models(*, timeout: int = _DEFAULT_TIMEOUT) -> list[OpenAIChatModel]:
    cfg = load_app_config()
    models: list[OpenAIChatModel] = []

    fallback_url = cfg.get("OPEN_AI_BASE_URL_FALLBACK", "")
    fallback_key = cfg.get("OPEN_AI_API_KEY_FALLBACK", "")
    if fallback_url and fallback_key:
        model_name = cfg.get("OPEN_AI_MODEL_FALLBACK") or cfg.get("OPEN_AI_MODEL", _DEFAULT_MODEL)
        models.append(
            _build_model(
                model_name,
                resolve_openai_base_url(fallback_url),
                fallback_key,
                timeout=timeout,
            )
        )

    if current_env() != "test":
        alt_model = (cfg.get("OPEN_AI_ALT_MODEL") or "").strip()
        if alt_model:
            api_base = resolve_openai_base_url(cfg.get("OPEN_AI_BASE_URL", ""))
            api_key = cfg.get("OPEN_AI_API_KEY", "")
            if api_base and api_key and not any(m.model_name == alt_model for m in models):
                models.append(_build_model(alt_model, api_base, api_key, timeout=timeout))

    return models


def _build_model(
    model_name: str,
    api_base: str,
    api_key: str,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
    connect_timeout: int = 10,
) -> OpenAIChatModel:
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    cache_key = (model_name, api_base, api_key_hash, timeout)

    with _cache_lock:
        cached = _model_cache.get(cache_key)
        if cached is not None:
            _log.debug("BUILD_MODEL | cache hit | model=%s key_hash=%s", model_name, api_key_hash)
            return cached

    _log.info(
        "BUILD_MODEL | model=%s timeout=%ds connect=%ds",
        model_name,
        timeout,
        connect_timeout,
    )
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=timeout, connect=connect_timeout),
        headers={"User-Agent": get_user_agent()},
    )
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=api_base, api_key=api_key, http_client=http_client),
    )

    with _cache_lock:
        _model_cache[cache_key] = model

    return model


def build_model_with_fallback(*, timeout: int = _DEFAULT_TIMEOUT) -> FallbackModel:
    primary = build_production_model(timeout=timeout)
    fallbacks = build_fallback_models(timeout=timeout)
    return FallbackModel(primary, *fallbacks)


async def _close_client(client) -> None:
    try:
        await client.aclose()
    except Exception:
        _log.debug("BUILD_MODEL | aclose failed", exc_info=True)


def _clear_model_cache() -> None:
    from src.agent.runtime.event_loop import submit

    with _cache_lock:
        clients = []
        for m in _model_cache.values():
            try:
                client = getattr(getattr(m, "provider", None), "client", None)
                if client is not None and hasattr(client, "aclose"):
                    clients.append(client)
            except Exception:
                pass
        _model_cache.clear()

    # 连接池绑定在常驻循环上，必须回到同一个循环里关闭；
    # 在别的循环上 aclose 会撞上 "Event loop is closed"。
    for client in clients:
        try:
            submit(_close_client(client), timeout=5.0)
        except Exception:
            _log.debug("BUILD_MODEL | client close skipped", exc_info=True)
    _log.debug("BUILD_MODEL | cache cleared | clients=%d", len(clients))


atexit.register(_clear_model_cache)
