from __future__ import annotations

import atexit
import json
import logging
import os
import re
import secrets
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import httpx

from config.settings import load_app_config_raw

logger = logging.getLogger(__name__)

_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_DONE_TTL_SECONDS = 120
_STALE_TTL_SECONDS = 900
_IDLE_HEARTBEAT_SECONDS = 15


def configured_sse_port() -> int:
    try:
        return int(os.environ.get("LEAD_IN_SSE_PORT", "") or 0)
    except (TypeError, ValueError):
        return 0


def same_origin_sse_enabled() -> bool:
    return os.environ.get("LEAD_IN_SSE_SAME_ORIGIN", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@lru_cache(maxsize=1)
def _app_origin_host() -> str:
    """应用自身配置的公网 base URL 主机名（用于同源/反代直连的 Origin 放行）。"""
    try:
        raw = load_app_config_raw().get("STREAMLIT_APP_BASE_URL", "") or ""
        return (urlparse(str(raw)).hostname or "").lower()
    except Exception:
        return ""


def _origin_allowed(origin: str | None) -> bool:
    if not origin:
        return True
    allowed = os.environ.get("LEAD_IN_SSE_ALLOW_ORIGIN", "").strip()
    if allowed and origin == allowed:
        return True
    try:
        host = (urlparse(origin).hostname or "").lower()
    except Exception:
        return False
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        import ipaddress

        if ipaddress.ip_address(host).is_private:
            return True
    except ValueError:
        pass
    # 应用配置的 base URL 主机（生产环境经反代同源访问时 Origin 为公网域名）
    return bool(_app_origin_host() and host == _app_origin_host())


_GHOST_TOKEN_TTL_SECONDS = 6 * 3600
_GHOST_TOKEN_MAX_REQS = 30
_GHOST_TOKEN_WINDOW_SECONDS = 60.0
_GHOST_MAX_BODY_BYTES = 100 * 1024
_GHOST_UPSTREAM_TIMEOUT = 20.0

_ghost_tokens: dict[str, tuple[float, list[float]]] = {}


def issue_ghost_token() -> str:
    now = time.monotonic()
    with _registry_lock:
        expired = [
            t for t, (issued, _) in _ghost_tokens.items() if now - issued > _GHOST_TOKEN_TTL_SECONDS
        ]
        for t in expired:
            _ghost_tokens.pop(t, None)
        token = secrets.token_hex(16)
        _ghost_tokens[token] = (now, [])
        return token


def _check_ghost_token(token: str) -> bool:
    now = time.monotonic()
    with _registry_lock:
        state = _ghost_tokens.get(token)
        if state is None:
            return False
        issued, hits = state
        if now - issued > _GHOST_TOKEN_TTL_SECONDS:
            _ghost_tokens.pop(token, None)
            return False
        hits[:] = [t for t in hits if now - t < _GHOST_TOKEN_WINDOW_SECONDS]
        if len(hits) >= _GHOST_TOKEN_MAX_REQS:
            return False
        hits.append(now)
        return True


@lru_cache(maxsize=1)
def _ghost_upstream_config() -> tuple[str, str]:
    from src.utils.env_config_loader import load_app_config

    cfg = load_app_config()
    return (
        str(cfg.get("GHOST_API_KEY", "") or ""),
        str(cfg.get("GHOST_API_BASE_URL", "https://api.deepseek.com/beta") or "").rstrip("/"),
    )


def _forward_chat_completion(body: bytes, content_type: str) -> tuple[int, str, bytes]:
    api_key, base_url = _ghost_upstream_config()
    if not api_key or not base_url:
        return (
            503,
            "application/json",
            json.dumps({"error": {"message": "GHOST_API_KEY 未配置"}}).encode("utf-8"),
        )
    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            content=body,
            headers={
                "Content-Type": content_type or "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=_GHOST_UPSTREAM_TIMEOUT,
        )
    except httpx.RequestError:
        logger.warning("ghost proxy upstream request failed", exc_info=True)
        return (
            502,
            "application/json",
            json.dumps({"error": {"message": "upstream_unreachable"}}).encode("utf-8"),
        )
    out_ctype = resp.headers.get("content-type", "application/json")
    logger.info("ghost proxy | upstream_status=%d bytes=%d", resp.status_code, len(resp.content))
    return resp.status_code, out_ctype, resp.content


class _RunStream:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.text = ""
        self.stage = ""
        self.stage_index = 0
        self.stage_count = 4
        self.details: list[str] = []
        self.retry = 0
        self.retry_max = 3
        self._meta_dirty = False
        self.done = False
        self.ok = True
        self.cancel_requested = False
        self.created = time.monotonic()
        self.last_activity = self.created
        self.cond = threading.Condition(threading.Lock())

    def publish_text(self, text: str) -> None:
        with self.cond:
            if text != self.text:
                self.text = text
                self.last_activity = time.monotonic()
                self.cond.notify_all()

    def publish_meta(
        self,
        *,
        stage: str = "",
        stage_index: int = 0,
        stage_count: int = 4,
        details: list[str] | None = None,
        retry: int = 0,
        retry_max: int = 3,
    ) -> None:
        with self.cond:
            new_details = list(details or [])
            changed = (
                stage != self.stage
                or stage_index != self.stage_index
                or stage_count != self.stage_count
                or new_details != self.details
                or retry != self.retry
                or retry_max != self.retry_max
            )
            if not changed:
                return
            self.stage = stage
            self.stage_index = stage_index
            self.stage_count = stage_count
            self.details = new_details
            self.retry = retry
            self.retry_max = retry_max
            self._meta_dirty = True
            self.last_activity = time.monotonic()
            self.cond.notify_all()

    def meta_snapshot(self) -> dict:
        return {
            "stage": self.stage,
            "stage_index": self.stage_index,
            "stage_count": self.stage_count,
            "details": self.details,
            "retry": self.retry,
            "retry_max": self.retry_max,
        }

    def close(self, ok: bool = True) -> None:
        with self.cond:
            if not self.done:
                self.done = True
                self.ok = bool(ok)
                self.last_activity = time.monotonic()
                self.cond.notify_all()


_registry: dict[str, _RunStream] = {}
_registry_lock = threading.Lock()
_server: ThreadingHTTPServer | None = None
_server_lock = threading.Lock()


def new_run_id() -> str:
    return secrets.token_hex(16)


def _prune() -> None:
    now = time.monotonic()
    with _registry_lock:
        stale = [
            rid
            for rid, s in _registry.items()
            if (s.done and now - s.last_activity > _DONE_TTL_SECONDS)
            or now - s.created > _STALE_TTL_SECONDS
        ]
        for rid in stale:
            _registry.pop(rid, None)


def open_run(run_id: str) -> _RunStream:
    _prune()
    with _registry_lock:
        stream = _registry.get(run_id)
        if stream is None:
            stream = _RunStream(run_id)
            _registry[run_id] = stream
        return stream


def close_run(run_id: str, ok: bool = True) -> None:
    with _registry_lock:
        stream = _registry.get(run_id)
    if stream is not None:
        stream.close(ok=ok)


def request_cancel(run_id: str) -> bool:
    with _registry_lock:
        stream = _registry.get(run_id)
    if stream is None:
        return False
    with stream.cond:
        stream.cancel_requested = True
        stream.last_activity = time.monotonic()
    return True


def is_cancel_requested(run_id: str) -> bool:
    with _registry_lock:
        stream = _registry.get(run_id)
    return bool(stream is not None and stream.cancel_requested)


def ensure_sse_server() -> int:
    global _server
    with _server_lock:
        if _server is None:
            port = configured_sse_port()
            # 默认只绑本机：浏览器直连走同机反代/同源模式；
            # 明确配置 LEAD_IN_SSE_PORT 时按部署意图放开到所有网卡。
            host = "0.0.0.0" if port else "127.0.0.1"
            server = ThreadingHTTPServer((host, port), _make_handler())
            server.daemon_threads = True
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            _server = server
            logger.info("lead_in_wait SSE server on port %d", server.server_address[1])
        return int(_server.server_address[1])


def shutdown_sse_server() -> None:
    global _server
    with _server_lock:
        if _server is None:
            return
        try:
            _server.shutdown()
            _server.server_close()
        except Exception:
            logger.warning("lead_in_wait SSE server shutdown failed", exc_info=True)
        _server = None
        logger.info("lead_in_wait SSE server shut down")


atexit.register(shutdown_sse_server)


class _SseHandler(BaseHTTPRequestHandler):
    server_version = ""
    sys_version = ""

    def do_GET(self) -> None:
        if not _origin_allowed(self.headers.get("Origin")):
            self.send_error(403)
            return
        path = self.path.split("?", 1)[0]
        if not path.startswith("/sse/"):
            self.send_error(404)
            return
        run_id = path[len("/sse/") :]
        if not _RUN_ID_RE.match(run_id):
            self.send_error(400)
            return
        with _registry_lock:
            stream = _registry.get(run_id)
        if stream is None:
            self.send_error(404)
            return
        _serve_stream(self, stream)

    def do_POST(self) -> None:
        if not _origin_allowed(self.headers.get("Origin")):
            self.send_error(403)
            return
        path = self.path.split("?", 1)[0]

        if path == "/ghost/chat/completions":
            self._handle_ghost_proxy()
            return

        if path.startswith("/cancel/"):
            run_id = path[len("/cancel/") :]
            if _RUN_ID_RE.match(run_id) and request_cancel(run_id):
                self.send_response(204)
                origin = self.headers.get("Origin")
                if origin:
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")
                self.end_headers()
                return
        self.send_error(404)

    def do_OPTIONS(self) -> None:
        """CORS preflight：浏览器跨端口调 /ghost 前必须通过（否则 501）。"""
        origin = self.headers.get("Origin")
        if origin and not _origin_allowed(origin):
            self.send_error(403)
            return
        self.send_response(204)
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def _handle_ghost_proxy(self) -> None:
        origin = self.headers.get("Origin")

        def _send_json(status: int, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(payload)

        auth = self.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        if not token or not _check_ghost_token(token):
            _send_json(401, b'{"error":{"message":"unauthorized"}}')
            return

        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            _send_json(400, b'{"error":{"message":"empty_body"}}')
            return
        if length > _GHOST_MAX_BODY_BYTES:
            _send_json(413, b'{"error":{"message":"body_too_large"}}')
            return

        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "application/json")
        status, out_ctype, payload = _forward_chat_completion(body, content_type)
        self.send_response(status)
        self.send_header("Content-Type", out_ctype)
        self.send_header("Cache-Control", "no-store")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


def _make_handler():
    return _SseHandler


def _emit(handler: BaseHTTPRequestHandler, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False)
    handler.wfile.write(f"data: {data}\n\n".encode())
    handler.wfile.flush()


def _serve_stream(handler: BaseHTTPRequestHandler, stream: _RunStream) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache, no-transform")
    handler.send_header("Connection", "keep-alive")
    origin = handler.headers.get("Origin")
    if origin:
        handler.send_header("Access-Control-Allow-Origin", origin)
        handler.send_header("Vary", "Origin")
    handler.end_headers()

    try:
        with stream.cond:
            sent = stream.text
            sent_meta = stream.meta_snapshot()
        _emit(handler, {"reset": True})
        if sent:
            _emit(handler, {"text": sent})
        _emit(handler, {"meta": sent_meta})
        while True:
            with stream.cond:
                while not stream.done and stream.text == sent and not stream._meta_dirty:
                    if not stream.cond.wait(timeout=_IDLE_HEARTBEAT_SECONDS):
                        break
                new = stream.text
                meta_dirty = stream._meta_dirty
                meta = stream.meta_snapshot()
                done = stream.done
                ok = stream.ok
                if meta_dirty:
                    stream._meta_dirty = False
                    sent_meta = meta
            if new != sent:
                if new.startswith(sent) and len(new) > len(sent):
                    _emit(handler, {"token": new[len(sent) :]})
                else:
                    _emit(handler, {"reset": True})
                    _emit(handler, {"text": new})
                sent = new
            elif meta_dirty:
                _emit(handler, {"meta": meta})
            if done:
                _emit(handler, {"done": True, "ok": ok})
                handler.close_connection = True
                return
            if new == sent:
                handler.wfile.write(b": ping\n\n")
                handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
