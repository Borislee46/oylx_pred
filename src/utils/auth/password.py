from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError, VerificationError

_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)

USERS_PATH_ENV = "DEMO_USERS_PATH"
_DEFAULT_USERS_PATH = "config/demo_users.json"


def hash_password(plaintext: str) -> str:
    return _hasher.hash(plaintext)


def verify_password(encoded_hash: str, plaintext: str) -> bool:
    if not encoded_hash or not plaintext:
        return False
    try:
        return _hasher.verify(encoded_hash, plaintext)
    except (VerifyMismatchError, InvalidHashError, VerificationError):
        return False
    except Exception:
        return False


def _users_path() -> str:
    return os.environ.get(USERS_PATH_ENV, _DEFAULT_USERS_PATH)


def load_users(path: str | None = None) -> dict[str, dict[str, Any]]:
    resolved = path or _users_path()
    try:
        raw = json.loads(Path(resolved).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    users = raw.get("users", {}) if isinstance(raw, dict) else {}
    return users if isinstance(users, dict) else {}


def get_user(username: str) -> dict[str, Any] | None:
    users = load_users()
    if not users:
        return None
    for key, value in users.items():
        if key == username:
            return value
    return None


def verify_totp(secret: str | None, code: str | None) -> bool:
    if not secret:
        return True
    if not code:
        return False
    try:
        import pyotp

        totp = pyotp.TOTP(secret)
        return totp.verify(code.strip(), valid_window=1)
    except Exception:
        return False

