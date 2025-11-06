import uuid
from dataclasses import dataclass, field
from typing import Any

import streamlit as st


@dataclass
class PredictionResultModel:
    similarity_results: list[dict] | None = None
    cross_major_results: list[dict] | None = None
    user_specified_results: list[dict] | None = None
    unified_results: list[dict] | None = None


@dataclass
class UserDataModel:
    session_id: str | None = None
    is_logged_in: bool = False
    user_info: dict[str, Any] = field(default_factory=dict)
    input_data: dict[str, Any] | None = None
    prediction_results: PredictionResultModel = field(default_factory=PredictionResultModel)
    prediction_submit_lock: bool = False
    other_states: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    _STATE_KEY = "user_data_model"

    def __init__(self) -> None:
        if self._STATE_KEY not in st.session_state:
            st.session_state[self._STATE_KEY] = UserDataModel()

        if self._model.session_id is None:
            self._model.session_id = str(uuid.uuid4())

    @property
    def _model(self) -> UserDataModel:
        if self._STATE_KEY not in st.session_state:
            st.session_state[self._STATE_KEY] = UserDataModel()
        return st.session_state[self._STATE_KEY]

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self._model, key):
            return getattr(self._model, key)
        return self._model.other_states.get(key, default)

    def set(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self._model, key):
                setattr(self._model, key, value)
            else:
                self._model.other_states[key] = value

    def batch_set(self, updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if hasattr(self._model, key):
                setattr(self._model, key, value)
            else:
                self._model.other_states[key] = value

    def delete(self, key: str) -> None:
        if key in self._model.other_states:
            del self._model.other_states[key]

    def clear_session(self) -> None:
        st.session_state[self._STATE_KEY] = UserDataModel()

    def get_widget_value(self, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    def get_current_user_info(self) -> dict[str, Any]:
        e2_email = st.session_state.get("e2_user_email", "")
        e2_nickname = st.session_state.get("e2_user_nickname", e2_email)
        if e2_email:
            return {"username": e2_email, "nickname": e2_nickname}

        if st.session_state.get("is_authenticated", False):
            username = st.session_state.get("username", "")
            if username:
                stored_user_info = self._model.user_info.copy()
                if stored_user_info and stored_user_info.get("username") == username:
                    return stored_user_info
                return {"username": username, "nickname": username}

        stored_user_info = self._model.user_info.copy()
        if stored_user_info.get("username"):
            return stored_user_info

        return {}
