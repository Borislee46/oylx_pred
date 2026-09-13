from __future__ import annotations


def check_module_permission(_email: str | None = None, _module: str | None = None) -> bool:
    return True


def check_user_access_permission(_email: str | None = None) -> bool:
    return True


def get_user_accessible_modules(_email: str = "") -> dict[str, bool]:
    return {"hk": True}


def is_admin(_email: str | None = None) -> bool:
    return False

