from collections.abc import Callable
from typing import Any, Literal

import streamlit as st

from src.pages.prediction.config.ui_messages import FORM_PLACEHOLDERS

_FilterMode = Literal["fuzzy", "contains", "prefix"] | None

_FILTER_MODE_MIN_OPTIONS = 15


def _filter_mode_for_option_count(n: int) -> _FilterMode:
    return "fuzzy" if n > _FILTER_MODE_MIN_OPTIONS else None


class SelectBoxHelper:
    def __init__(self, session_manager, form_state_manager, logger):
        self.session_manager = session_manager
        self.form_state_manager = form_state_manager
        self.logger = logger

    def render_cached_selectbox(
        self,
        label: str,
        widget_key: str,
        cache_key: str,
        history_key: str,
        options_generator_func: Callable[[], Any],
        on_change_callback: Callable,
        options_path_in_cache: str = None,
    ) -> Any:
        try:
            if self.session_manager.get(cache_key) is None:
                options_data = options_generator_func()
                self.session_manager.set(**{cache_key: options_data})

            cached_data = self.session_manager.get(cache_key)

            if options_path_in_cache:
                all_options = cached_data.get(options_path_in_cache, []) if cached_data else []
            else:
                all_options = cached_data if cached_data else []

            user_history_data = self.session_manager.get("user_history_data", {})
            saved_value = user_history_data.get(history_key)

            default_index = 0
            if saved_value and all_options and saved_value in all_options:
                try:
                    default_index = all_options.index(saved_value)
                except (ValueError, TypeError):
                    default_index = 0

            if not all_options:
                default_index = 0
            elif default_index >= len(all_options):
                default_index = 0

            fm = _filter_mode_for_option_count(len(all_options))
            return st.selectbox(
                label,
                all_options,
                index=default_index,
                on_change=on_change_callback,
                placeholder=FORM_PLACEHOLDERS["selectbox_default"],
                key=widget_key,
                filter_mode=fm,
            )
        except Exception as e:
            self.logger.error(f"渲染selectbox失败 ({widget_key}): {e}", exc_info=True)
            toast_key = f"_selectbox_error_toast_{widget_key}"
            err_sig = f"{type(e).__name__}:{e}"
            if st.session_state.get(toast_key) != err_sig:
                st.toast(FORM_PLACEHOLDERS["selectbox_error"])
                st.session_state[toast_key] = err_sig
            return st.selectbox(
                label,
                [],
                index=0,
                on_change=on_change_callback,
                placeholder=FORM_PLACEHOLDERS["selectbox_default"],
                key=widget_key,
            )

    def render_multiselect(
        self,
        label: str,
        options: list,
        default_selections: list,
        widget_key: str,
        on_change_callback: Callable,
    ) -> list:
        fm = _filter_mode_for_option_count(len(options))
        return st.multiselect(
            label,
            options=options,
            default=default_selections,
            key=widget_key,
            on_change=on_change_callback,
            placeholder="",
            filter_mode=fm,
        )
