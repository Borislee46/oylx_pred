import streamlit as st

from src.utils.logger import setup_logger

page_components_logger = setup_logger("page3", "prediction")


def display_feedback_section(session_id: str) -> None:
    key = f"feedback_{session_id}"
    prev_key = f"{key}_prev"
    toast_key = f"{key}_toast_for"
    current = st.feedback("thumbs", key=key)

    def _coerce_feedback(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if s in ("0", "1"):
                return int(s)
        return v

    current_v = _coerce_feedback(current)
    prev_v = _coerce_feedback(st.session_state.get(prev_key))

    if prev_key not in st.session_state or current_v != prev_v:
        if current_v is None and prev_v is not None:
            prev_text = "满意" if prev_v == 1 else "不满意"
            page_components_logger.info(f"用户取消反馈: {prev_text}, session_id: {session_id}")
        elif current_v is not None and current_v != prev_v:
            current_text = "满意" if current_v == 1 else "不满意"
            page_components_logger.info(f"用户反馈: {current_text}, session_id: {session_id}")

        st.session_state[prev_key] = current_v

    if current_v is None:
        st.session_state.pop(toast_key, None)
        return

    if st.session_state.get(toast_key) == current_v:
        return

    st.session_state[toast_key] = current_v
    if current_v == 1:
        st.toast("感谢您的肯定！我们会继续努力！")
    else:
        st.toast("收到您的反馈，我们会持续改进！")
