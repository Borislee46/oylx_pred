from __future__ import annotations

import hmac
import os
import secrets
import time
from typing import Any

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SESSION_COOKIE_NAME = "oylx_demo_session"
CSRF_COOKIE_NAME = "oylx_demo_csrf"
_COOKIE_SALT = "oylx-demo-session-v1"
SESSION_TTL_SECONDS = 24 * 3600


def session_secret() -> str:
    """Return the mandatory session-signing secret.

    Fails closed: if no secret is configured the app must not start serving.
    Use the same value for Streamlit's own cookie secret on the server.
    """
    secret = os.environ.get("DEMO_COOKIE_SECRET") or os.environ.get(
        "STREAMLIT_SERVER_COOKIE_SECRET"
    )
    if not secret or len(secret) < 32:
        raise RuntimeError(
            "未配置会话签名密钥。请在环境变量中设置 DEMO_COOKIE_SECRET"
            "（至少 32 位随机字符串，部署时由 install_ubuntu.sh 自动生成）。"
        )
    return secret


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(session_secret(), salt=_COOKIE_SALT)


def make_session_token(username: str, nickname: str = "") -> str:
    payload = {
        "username": username,
        "nickname": nickname or username,
        "iat": int(time.time()),
    }
    return _serializer().dumps(payload)


def validate_session_token(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        payload = _serializer().loads(raw, max_age=SESSION_TTL_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    except Exception:
        return None
    if not isinstance(payload, dict) or not payload.get("username"):
        return None
    return payload


def _secure_cookie(request) -> bool:
    secure = request.url.scheme == "https"
    trust_proxy = os.environ.get("DEMO_TRUST_PROXY", "").strip() == "1"
    if trust_proxy:
        fwd = request.headers.get("x-forwarded-proto", "")
        if "https" in fwd.lower():
            secure = True
    return secure


def make_set_session_cookie_header(
    username: str,
    nickname: str = "",
    *,
    request=None,
    max_age: int = SESSION_TTL_SECONDS,
) -> str:
    value = make_session_token(username, nickname)
    parts = [
        f"{SESSION_COOKIE_NAME}={value}",
        "Path=/",
        "HttpOnly",
        "SameSite=Lax",
        f"Max-Age={max_age}",
    ]
    if request is None or _secure_cookie(request):
        parts.append("Secure")
    return "; ".join(parts)


def make_clear_session_cookie_header() -> str:
    return f"{SESSION_COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"


def make_csrf_cookie_header(token: str, *, request=None) -> str:
    parts = [
        f"{CSRF_COOKIE_NAME}={token}",
        "Path=/",
        "HttpOnly",
        "SameSite=Strict",
        "Max-Age=3600",
    ]
    if request is None or _secure_cookie(request):
        parts.append("Secure")
    return "; ".join(parts)


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_matches(cookie_token: str | None, form_token: str | None) -> bool:
    if not cookie_token or not form_token:
        return False
    return hmac.compare_digest(cookie_token, form_token)

