from __future__ import annotations

import html
import threading
import time
import urllib.parse

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

from src.utils.auth.access_code import verify_access_code
from src.utils.auth.session import (
    CSRF_COOKIE_NAME,
    SESSION_COOKIE_NAME,
    csrf_matches,
    make_clear_session_cookie_header,
    make_csrf_cookie_header,
    make_set_session_cookie_header,
    new_csrf_token,
    validate_session_token,
)

LOGIN_PATH = "/login"
LOGOUT_PATH = "/logout"

DISPLAY_USERNAME = "demo"
DISPLAY_NICKNAME = "演示用户"

_PUBLIC_EXACT = ("/_stcore/health", "/healthz", "/favicon.ico")
_PUBLIC_PREFIXES = ("/login", "/logout", "/_stcore/static")
_SAFE_NEXT_PREFIXES = ("/", "/hk")

_MAX_FAILURES_PER_IP = 20
_IP_WINDOW_SECONDS = 60 * 60


def _sanitize_next(raw: str | None) -> str:
    if not raw:
        return "/"
    if raw.startswith("/") and not raw.startswith("//") and "\\" not in raw:
        stripped = raw.strip("/")
        first = "/" + stripped.split("/", 1)[0] if stripped else "/"
        if first in _SAFE_NEXT_PREFIXES:
            return raw
    return "/"


def _client_ip(request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "127.0.0.1"


class _AccessCodeLimiter:
    """按 IP 限速，避免访问码被暴力枚举。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ip_failures: dict[str, list[float]] = {}

    def is_locked(self, ip: str) -> bool:
        with self._lock:
            stamps = [
                s for s in self._ip_failures.get(ip, []) if time.time() - s < _IP_WINDOW_SECONDS
            ]
            self._ip_failures[ip] = stamps
            return len(stamps) >= _MAX_FAILURES_PER_IP

    def record_failure(self, ip: str) -> None:
        with self._lock:
            self._ip_failures.setdefault(ip, []).append(time.time())

    def reset(self, ip: str) -> None:
        with self._lock:
            self._ip_failures.pop(ip, None)


_limiter = _AccessCodeLimiter()


def _login_html(*, csrf_token: str, error: str = "", next_path: str = "/") -> str:
    # 访问码一律不在此页面出现：一旦显示，"门禁"就退化成了"贴在门上的密码"。
    # 本地演示需要的默认码写在项目 README 里，不该由登录页代劳。
    hint_html = (
        f'<div class="alert">{html.escape(error)}</div>'
        if error
        else '<div class="alert muted">请输入访问码（由部署方提供，不在此页面显示）。</div>'
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>访问验证 · Signals 演示环境</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
    background: #0b1220; color: #cbd5e1;
    min-height: 100vh; display: flex; align-items: center; justify-content: center;
  }}
  .card {{
    width: 360px; background: #111a2c; border: 1px solid #1e293b;
    border-radius: 16px; padding: 36px 32px; box-shadow: 0 20px 60px rgba(0,0,0,.45);
  }}
  .logo {{ color: #06b6d4; font-size: 22px; font-weight: 700; letter-spacing: .5px; }}
  .sub {{ color: #64748b; font-size: 13px; margin: 6px 0 26px; }}
  label {{ display: block; font-size: 13px; color: #94a3b8; margin: 14px 0 6px; }}
  input {{
    width: 100%; padding: 11px 12px; border-radius: 8px;
    border: 1px solid #26334a; background: #0f172a; color: #e2e8f0;
    font-size: 14px; outline: none; letter-spacing: .08em;
  }}
  input:focus {{ border-color: #06b6d4; }}
  button {{
    width: 100%; margin-top: 22px; padding: 12px; border: 0; border-radius: 8px;
    background: #06b6d4; color: #04121a; font-size: 15px; font-weight: 700; cursor: pointer;
  }}
  button:hover {{ background: #22d3ee; }}
  .alert {{ font-size: 13px; padding: 10px 12px; border-radius: 8px; margin-bottom: 6px; }}
  .alert {{ background: rgba(244,63,94,.12); color: #fda4af; border: 1px solid rgba(244,63,94,.35); }}
  .alert.muted {{ background: rgba(148,163,184,.08); color: #94a3b8; border-color: #1e293b; }}
  code {{ color: #67e8f9; }}
  .hint {{ font-size: 12px; color: #475569; margin-top: 18px; text-align: center; }}
</style>
</head>
<body>
  <form class="card" method="post" action="/login" autocomplete="off">
    <input type="hidden" name="next" value="{html.escape(next_path)}">
    <input type="hidden" name="csrf_token" value="{html.escape(csrf_token)}">
    <div class="logo">Signals</div>
    <div class="sub">留学择校预测 · 开源演示环境</div>
    {hint_html}
    <label for="access_code">访问码</label>
    <input type="password" id="access_code" name="access_code"
           placeholder="请输入访问码" required autofocus autocomplete="off">
    <button type="submit">进 入</button>
    <div class="hint">演示环境仅需访问码，无需注册账号</div>
  </form>
</body>
</html>
"""


async def login_page(request) -> Response:
    next_path = _sanitize_next(request.query_params.get("next"))
    error = request.query_params.get("error", "")
    messages = {
        "1": "访问码不正确，请重试。",
        "2": "尝试次数过多，请 1 小时后再试。",
        "3": "请求校验失败，请刷新页面后重试。",
    }
    csrf_token = new_csrf_token()
    response = HTMLResponse(
        _login_html(
            csrf_token=csrf_token,
            error=messages.get(error, ""),
            next_path=next_path,
        )
    )
    response.headers["Set-Cookie"] = make_csrf_cookie_header(csrf_token, request=request)
    return response


async def login_submit(request) -> Response:
    try:
        form = await request.form()
    except Exception:
        return RedirectResponse(f"{LOGIN_PATH}?error=3", status_code=302)

    cookie_csrf = request.cookies.get(CSRF_COOKIE_NAME)
    form_csrf = str(form.get("csrf_token", "") or "")
    if not csrf_matches(cookie_csrf, form_csrf):
        return RedirectResponse(f"{LOGIN_PATH}?error=3", status_code=302)

    next_path = _sanitize_next(str(form.get("next", "/") or "/"))
    access_code = str(form.get("access_code", "") or "")
    ip = _client_ip(request)

    if _limiter.is_locked(ip):
        return RedirectResponse(
            f"{LOGIN_PATH}?next={urllib.parse.quote(next_path)}&error=2", status_code=302
        )

    if not verify_access_code(access_code):
        _limiter.record_failure(ip)
        return RedirectResponse(
            f"{LOGIN_PATH}?next={urllib.parse.quote(next_path)}&error=1", status_code=302
        )

    _limiter.reset(ip)
    response = RedirectResponse(next_path, status_code=302)
    response.headers["Set-Cookie"] = make_set_session_cookie_header(
        DISPLAY_USERNAME, DISPLAY_NICKNAME, request=request
    )
    return response


async def logout(request) -> Response:
    response = RedirectResponse(LOGIN_PATH, status_code=302)
    response.headers["Set-Cookie"] = make_clear_session_cookie_header()
    return response


async def healthz(request) -> Response:
    return Response("ok", media_type="text/plain")


def auth_routes() -> list[Route]:
    return [
        Route(LOGIN_PATH, login_page, methods=["GET"]),
        Route(LOGIN_PATH, login_submit, methods=["POST"]),
        Route(LOGOUT_PATH, logout, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
    ]


def _is_public(path: str) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES)


class AuthEnforcementMiddleware(BaseHTTPMiddleware):
    """Block every route except login/health/static unless a valid signed
    session cookie is present."""

    async def dispatch(self, request, call_next):
        path = request.url.path
        if _is_public(path):
            return await call_next(request)

        raw_cookie = request.cookies.get(SESSION_COOKIE_NAME)
        if raw_cookie and validate_session_token(raw_cookie):
            return await call_next(request)

        if path in ("/", "/index.html"):
            return RedirectResponse(LOGIN_PATH, status_code=302)
        return Response("未登录或会话已过期", status_code=403)
