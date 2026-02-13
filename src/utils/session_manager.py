import uuid
from dataclasses import dataclass, field, fields
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

    def __init__(self) -> None:
        if self._STATE_KEY not in st.session_state:
            st.session_state[self._STATE_KEY] = UserDataModel()

        if "session_id" not in st.session_state:
            st.session_state["session_id"] = self._model.session_id[:8]

    @property
    def _model(self) -> UserDataModel:
        return st.session_state[self._STATE_KEY]

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
        for key, value in updates.items():
            if key in self._MODEL_FIELDS:
                setattr(self._model, key, value)
            else:
                self._model.other_states[key] = value

    def delete(self, key: str) -> None:
        if key in self._model.other_states:
            del self._model.other_states[key]
        elif key in self._MODEL_FIELDS and key != "session_id":
            default_model = UserDataModel()
            default_value = getattr(default_model, key)
            setattr(self._model, key, default_value)

    def clear_session(self) -> None:
        current_id = self._model.session_id
        st.session_state[self._STATE_KEY] = UserDataModel(session_id=current_id)

    def get_widget_value(self, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    def get_current_user_info(self) -> dict[str, Any]:
        if self._model.user_info.get("username"):
            return self._model.user_info

        e2_email = st.session_state.get("e2_user_email")
        if e2_email:
            info = {
                "username": e2_email,
                "nickname": st.session_state.get("e2_user_nickname", e2_email),
            }
            self.set(user_info=info, is_logged_in=True)
            return info

        if st.session_state.get("is_authenticated", False):
            username = st.session_state.get("username", "")
            if username:
                info = {"username": username, "nickname": username}
                self.set(user_info=info, is_logged_in=True)
                return info

        return {}
