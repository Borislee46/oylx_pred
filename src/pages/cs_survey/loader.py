from __future__ import annotations

import pandas as pd
import streamlit as st

from .registry import resolve_data_path
from .schema import SurveyConfig


@st.cache_data(show_spinner=False)
def _load_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str, encoding="utf-8-sig")


def load_source(cfg: SurveyConfig, source_id: str) -> pd.DataFrame:
    src = cfg.source(source_id)
    path = resolve_data_path(src.path)
    return _load_csv(str(path))


def load_all_sources(cfg: SurveyConfig) -> dict[str, pd.DataFrame]:
    return {s.id: load_source(cfg, s.id) for s in cfg.sources}
