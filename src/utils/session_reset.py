from __future__ import annotations

import streamlit as st

def clear_session_by_prefix(prefix: str, exclude: set[str] | None = None) -> int:
    exclude = exclude or set()
    keys_to_del = [k for k in st.session_state if k.startswith(prefix) and k not in exclude]
    for k in keys_to_del:
        del st.session_state[k]
    return len(keys_to_del)
