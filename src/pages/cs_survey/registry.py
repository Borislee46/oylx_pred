from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st
import yaml

from .schema import SurveyConfig, parse_survey_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _REPO_ROOT / "config" / "cs_survey"


def _config_dir() -> Path:
    return _CONFIG_DIR


@lru_cache(maxsize=1)
def _scan_configs() -> dict[str, SurveyConfig]:
    out: dict[str, SurveyConfig] = {}
    for path in sorted(_config_dir().glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = parse_survey_config(raw, source_path=str(path))
        out[cfg.id] = cfg
    return out


def list_surveys() -> list[SurveyConfig]:
    return list(_scan_configs().values())


def get_survey(survey_id: str) -> SurveyConfig | None:
    return _scan_configs().get(survey_id)


def reset_cache() -> None:
    _scan_configs.cache_clear()


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_data_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return _REPO_ROOT / rel_or_abs


@st.cache_data(show_spinner=False)
def _cached_survey_ids() -> list[str]:
    return [cfg.id for cfg in list_surveys()]
