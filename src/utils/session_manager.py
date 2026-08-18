import uuid
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

import streamlit as st

@dataclass
class PredictionResultModel:
    similarity_results: list[dict] | None = None
    cross_major_results: list[dict] | None = None
    user_specified_results: list[dict] | None = None
    unified_results: list[dict] | None = None
    meta: dict[str, Any] | None = None

@dataclass
class UserDataModel:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_logged_in: bool = False
    user_info: dict[str, Any] = field(default_factory=dict)
    input_data: dict[str, Any] | None = None
    prediction_results: PredictionResultModel = field(default_factory=PredictionResultModel)
    prediction_submit_lock: bool = False
    other_states: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    _STATE_KEY = "user_data_model"
    _MODEL_FIELDS = {f.name for f in fields(UserDataModel)}

    @staticmethod
    def _build_field_defaults() -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for f in fields(UserDataModel):
            if f.default is not MISSING:
                defaults[f.name] = f.default
            elif f.default_factory is not MISSING:
                defaults[f.name] = f.default_factory()
            else:
                defaults[f.name] = None
        return defaults

    _FIELD_DEFAULTS = _build_field_defaults.__func__()

    def __init__(self) -> None:
        if self._STATE_KEY not in st.session_state:
            st.session_state[self._STATE_KEY] = UserDataModel()
            st.session_state["session_id"] = self.session_id_short
            try:
                from src.utils.analytics import track as _track

                udm = st.session_state[self._STATE_KEY]
                info = getattr(udm, "user_info", {}) or {}
                _track("session_start", contract_tier=info.get("contract_tier", "unknown"))
            except Exception:
                pass

    @property
    def _model(self) -> UserDataModel:
        return st.session_state[self._STATE_KEY]

    @property
    def session_id(self) -> str:
        return self._model.session_id

    @property
    def session_id_short(self) -> str:
        return self._model.session_id[:8]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._MODEL_FIELDS:
            return getattr(self._model, key)
        return self._model.other_states.get(key, default)

    def set(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key in self._MODEL_FIELDS:
                setattr(self._model, key, value)
            else:
                self._model.other_states[key] = value

    def batch_set(self, updates: dict[str, Any]) -> None:
        self.set(**updates)

    def delete(self, key: str) -> None:
        if key in self._model.other_states:
            del self._model.other_states[key]
        elif key in self._MODEL_FIELDS and key != "session_id":
            setattr(self._model, key, self._FIELD_DEFAULTS.get(key))

    def pop(self, key: str, default: Any = None) -> Any:
        value = self.get(key, default)
        self.delete(key)
        return value

    def clear_session(self) -> None:
        current_id = self._model.session_id
        st.session_state[self._STATE_KEY] = UserDataModel(session_id=current_id)
        st.session_state["session_id"] = current_id[:8]

    def prune_states(self, prefixes: list[str]) -> int:
        to_delete = [k for k in self._model.other_states if any(k.startswith(p) for p in prefixes)]
        for k in to_delete:
            del self._model.other_states[k]
        return len(to_delete)

    def clear_prefix(self, prefix: str, exclude: "set[str] | None" = None) -> int:
        exclude = exclude or set()
        removed = 0

        to_delete = [
            k for k in self._model.other_states if k.startswith(prefix) and k not in exclude
        ]
        for k in to_delete:
            del self._model.other_states[k]
        removed += len(to_delete)

        top_keys = [k for k in st.session_state if k.startswith(prefix) and k not in exclude]
        for k in top_keys:
            del st.session_state[k]
        removed += len(top_keys)

        return removed

    def get_widget_value(self, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    def ensure_user_info(self) -> dict[str, Any]:
        return self._model.user_info if self._model.user_info.get("username") else {}
