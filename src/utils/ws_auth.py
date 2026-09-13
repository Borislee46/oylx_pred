from __future__ import annotations

import logging

from streamlit.web.server.starlette import starlette_websocket

from src.utils.auth.session import SESSION_COOKIE_NAME, validate_session_token

_log = logging.getLogger(__name__)

_original_create_handler = None
_installed = False


def ws_auth_enabled() -> bool:
    """The desensitized app never disables the WebSocket auth gate."""
    return True


def _has_valid_session_cookie(websocket) -> bool:
    raw = websocket.cookies.get(SESSION_COOKIE_NAME)
    return raw is not None and validate_session_token(raw) is not None


def _make_gated_handler(runtime):
    original = _original_create_handler(runtime)

    async def _gated_endpoint(websocket):
        if not _has_valid_session_cookie(websocket):
            _log.warning("Rejecting WS handshake: missing/invalid %s", SESSION_COOKIE_NAME)
            await websocket.close(code=1008)
            return
        await original(websocket)

    return _gated_endpoint


def install_ws_auth_gate() -> None:
    global _original_create_handler, _installed
    if _installed:
        return
    _installed = True
    if not ws_auth_enabled():
        _log.warning("WS auth gate disabled by configuration")
        return
    _original_create_handler = starlette_websocket.create_websocket_handler
    starlette_websocket.create_websocket_handler = _make_gated_handler
    _log.info("WS auth gate installed on /_stcore/stream")
