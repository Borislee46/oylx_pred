from __future__ import annotations

import html
import threading
import time
import urllib.parse

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

from src.utils.auth.password import get_user, verify_password, verify_totp
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

_PUBLIC_EXACT = ("/_stcore/health", "/healthz", "/favicon.ico")
_PUBLIC_PREFIXES = ("/login", "/logout", "/_stcore/static")
_SAFE_NEXT_PREFIXES = ("/", "/hk")

_MAX_FAILURES_PER_USER = 5
_LOCKOUT_SECONDS = 15 * 60
_MAX_FAILURES_PER_IP = 30
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


class _LoginLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures: dict[tuple[str, str], list[float]] = {}
        self._ip_failures: dict[str, list[float]] = {}

    def is_locked(self, ip: str, username: str) -> bool:
        with self._lock:
            key = (ip, username)
            stamps = self._failures.get(key, [])
            stamps = [s for s in stamps if time.time() - s < _LOCKOUT_SECONDS]
            self._failures[key] = stamps
            if len(stamps) >= _MAX_FAILURES_PER_USER:
                return True
            ip_stamps = [
                s for s in self._ip_failures.get(ip, []) if time.time() - s < _IP_WINDOW_SECONDS
            ]
            self._ip_failures[ip] = ip_stamps
            return len(ip_stamps) >= _MAX_FAILURES_PER_IP

    def record_failure(self, ip: str, username: str) -> None:
        now = time.time()
        with self._lock:
            self._failures.setdefault((ip, username), []).append(now)
            self._ip_failures.setdefault(ip, []).append(now)

    def reset(self, ip: str, username: str) -> None:
        with self._lock:
            self._failures.pop((ip, username), None)


_limiter = _LoginLimiter()


def _login_html(*, csrf_token: str, error: str = "", next_path: str = "/") -> str:
    error_html = (
        f'<div class="alert">{html.escape(error)}</div>'
        if error
        else '<div class="alert muted">请使用部署时创建的账号登录</div>'
    )
    totp_field = (
        '<label for="totp_code">动态验证码（TOTP，未开启则留空）</label>'
        '<input type="text" id="totp_code" name="totp_code" autocomplete="one-time-code" '
        'placeholder="6 位动态码" inputmode="numeric" pattern="[0-9]{6}" maxlength="6">'
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>登录 · Signals 演示环境</title>
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
    font-size: 14px; outline: none;
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
  .hint {{ font-size: 12px; color: #475569; margin-top: 18px; text-align: center; }}
</style>
</head>
<body>
  <form class="card" method="post" action="/login" autocomplete="off">
    <input type="hidden" name="next" value="{html.escape(next_path)}">
    <input type="hidden" name="csrf_token" value="{html.escape(csrf_token)}">
    <div class="logo">Signals</div>
    <div class="sub">留学择校预测 · 脱敏演示环境</div>
    {error_html}
    <label for="username">用户名</label>
    <input type="text" id="username" name="username" autocomplete="username" required autofocus>
    <label for="password">密码</label>
    <input type="password" id="password" name="password" autocomplete="current-password" required>
    {totp_field}
    <button type="submit">登 录</button>
    <div class="hint">账号由部署脚本创建，密码仅以 argon2id 哈希保存</div>
  </form>
</body>
</html>
"""


async def login_page(request) -> Response:
    next_path = _sanitize_next(request.query_params.get("next"))
    error = request.query_params.get("error", "")
    messages = {
        "1": "用户名或密码错误，请重试。",
        "2": "登录尝试过多，请 15 分钟后再试。",
        "3": "动态验证码错误或已过期。",
        "4": "登录会话无效或已过期，请重新登录。",
        "5": "请求校验失败，请刷新页面后重试。",
    }
    error_text = messages.get(error, "")
    csrf_token = new_csrf_token()
    response = HTMLResponse(
        _login_html(csrf_token=csrf_token, error=error_text, next_path=next_path)
    )
    response.headers["Set-Cookie"] = make_csrf_cookie_header(csrf_token, request=request)
    return response


async def login_submit(request) -> Response:
    try:
        form = await request.form()
    except Exception:
        return RedirectResponse("/login?error=5", status_code=302)

    cookie_csrf = request.cookies.get(CSRF_COOKIE_NAME)
    form_csrf = str(form.get("csrf_token", "") or "")
    if not csrf_matches(cookie_csrf, form_csrf):
        return RedirectResponse("/login?error=5", status_code=302)

    next_path = _sanitize_next(str(form.get("next", "/") or "/"))
    username = str(form.get("username", "") or "").strip()
    password = str(form.get("password", "") or "")
    totp_code = str(form.get("totp_code", "") or "").strip()
    ip = _client_ip(request)

    if _limiter.is_locked(ip, username):
        return RedirectResponse(
            f"/login?next={urllib.parse.quote(next_path)}&error=2", status_code=302
        )

    user = get_user(username)
    if user is None or not verify_password(str(user.get("password_hash", "")), password):
        _limiter.record_failure(ip, username)
        return RedirectResponse(
            f"/login?next={urllib.parse.quote(next_path)}&error=1", status_code=302
        )

    if not verify_totp(str(user.get("totp_secret") or ""), totp_code):
        _limiter.record_failure(ip, username)
        error_code = "3" if user.get("totp_secret") else "1"
        return RedirectResponse(
            f"/login?next={urllib.parse.quote(next_path)}&error={error_code}", status_code=302
        )

    _limiter.reset(ip, username)
    nickname = str(user.get("nickname", username))
    response = RedirectResponse(next_path, status_code=302)
    response.headers["Set-Cookie"] = make_set_session_cookie_header(
        username, nickname, request=request
    )
    return response


async def logout(request) -> Response:
    response = RedirectResponse("/login", status_code=302)
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
    """Block every route except the login/health/static endpoints
    unless a valid signed session cookie is present."""

    async def dispatch(self, request, call_next):
        path = request.url.path
        if _is_public(path):
            return await call_next(request)

        raw_cookie = request.cookies.get(SESSION_COOKIE_NAME)
        if raw_cookie and validate_session_token(raw_cookie):
            return await call_next(request)

        if path in ("/", "/index.html"):
            return RedirectResponse("/login", status_code=302)
        return Response("未登录或会话已过期", status_code=403)
