from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAYOUT_PATH = _REPO_ROOT / "config" / "cs_survey" / "layout.json"


def _read_layout_file() -> dict[str, Any]:
    if not _LAYOUT_PATH.exists():
        return {}
    with _LAYOUT_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def layout_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def load_layout_config() -> dict[str, Any]:
    raw = _read_layout_file()
    pages = raw.get("pages")
    return {
        "global": raw.get("global") if isinstance(raw.get("global"), dict) else {},
        "pages": pages if isinstance(pages, list) else [],
    }


def get_global_layout() -> dict[str, Any]:
    return load_layout_config()["global"]


def get_page_layout(page_key: str) -> dict[str, Any]:
    for page in load_layout_config()["pages"]:
        if not isinstance(page, dict) or page.get("key") != page_key:
            continue
        layout = page.get("layout")
        return layout if isinstance(layout, dict) else {}
    return {}
