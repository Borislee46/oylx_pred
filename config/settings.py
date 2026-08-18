from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.config_models import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_CONFIG_PATH = PROJECT_ROOT / "config" / "app_config.json"


def current_env() -> str:
    return (os.environ.get("APP_ENV", "prod") or "prod").strip()


@lru_cache(maxsize=1)
def _load_all_configs() -> dict:
    with APP_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_app_config_raw() -> dict:
    return dict(_load_all_configs().get(current_env(), {}))


def load_app_config() -> AppConfig:
    from src.utils.config_models import AppConfig

    return AppConfig.model_validate(load_app_config_raw())


def clear_app_config_cache() -> None:
    _load_all_configs.cache_clear()
