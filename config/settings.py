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

# 允许用环境变量覆盖 AI 凭据/端点，避免把个人 key 写进任何文件。
# 部署时在 .env（systemd EnvironmentFile）中设置即可，优先级高于 app_config.json。
_ENV_OVERRIDES: dict[str, str] = {
    "OPEN_AI_API_KEY": "OPENAI_API_KEY",
    "OPEN_AI_BASE_URL": "OPENAI_BASE_URL",
    "OPEN_AI_MODEL": "OPENAI_MODEL",
    "OPEN_AI_API_KEY_FALLBACK": "OPENAI_API_KEY_FALLBACK",
    "OPEN_AI_BASE_URL_FALLBACK": "OPENAI_BASE_URL_FALLBACK",
    "OPEN_AI_ALT_MODEL": "OPENAI_ALT_MODEL",
    "GHOST_API_KEY": "GHOST_API_KEY",
    "GHOST_API_BASE_URL": "GHOST_API_BASE_URL",
    "GHOST_API_MODEL": "GHOST_API_MODEL",
}


def current_env() -> str:
    return (os.environ.get("APP_ENV", "prod") or "prod").strip()


@lru_cache(maxsize=1)
def _load_all_configs() -> dict:
    with APP_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_app_config_raw() -> dict:
    cfg = dict(_load_all_configs().get(current_env(), {}))
    for key, env_name in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value:
            cfg[key] = value
    # 统一 key：GHOST（幽灵补全）未单独配置时复用主 AI key
    if not cfg.get("GHOST_API_KEY") and cfg.get("OPEN_AI_API_KEY"):
        cfg["GHOST_API_KEY"] = cfg["OPEN_AI_API_KEY"]
    return cfg


def load_app_config() -> AppConfig:
    from src.utils.config_models import AppConfig

    return AppConfig.model_validate(load_app_config_raw())


def clear_app_config_cache() -> None:
    _load_all_configs.cache_clear()
