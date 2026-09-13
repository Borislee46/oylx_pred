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
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "config" / "app_config.json.example"

# 环境变量优先于配置文件，便于容器/CI 注入密钥而不落地到磁盘。
_ENV_OVERRIDES = {
    "OPENAI_API_KEY": "OPEN_AI_API_KEY",
    "OPENAI_BASE_URL": "OPEN_AI_BASE_URL",
    "OPENAI_MODEL": "OPEN_AI_MODEL",
    "GHOST_API_KEY": "GHOST_API_KEY",
    "STREAMLIT_APP_BASE_URL": "STREAMLIT_APP_BASE_URL",
}


def current_env() -> str:
    return (os.environ.get("APP_ENV", "test") or "test").strip()


def _env_overrides() -> dict:
    return {
        cfg_name: value
        for env_name, cfg_name in _ENV_OVERRIDES.items()
        if (value := os.environ.get(env_name))
    }


@lru_cache(maxsize=1)
def _load_all_configs() -> dict:
    source = APP_CONFIG_PATH if APP_CONFIG_PATH.is_file() else EXAMPLE_CONFIG_PATH
    data: dict = {}
    if source.is_file():
        try:
            loaded = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}

    overrides = _env_overrides()
    for env_name in ("test", "prod"):
        section = data.get(env_name)
        if not isinstance(section, dict):
            section = {}
            data[env_name] = section
        if overrides:
            section.update(overrides)
    return data


def load_app_config_raw() -> dict:
    return dict(_load_all_configs().get(current_env(), {}))


def load_app_config() -> AppConfig:
    from src.utils.config_models import AppConfig

    return AppConfig.model_validate(load_app_config_raw())


def clear_app_config_cache() -> None:
    _load_all_configs.cache_clear()
