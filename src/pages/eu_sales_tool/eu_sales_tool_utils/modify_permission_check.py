import streamlit as st

ADMIN_EMAILS = [
    "lijiapeng8@xdf.cn",
    "sibo@xdf.cn",
    "liwenbo13@xdf.cn",
    "dingjing3@xdf.cn",
    "zhouyiyang3@xdf.cn",
    "yuenwaifong@xdf.cn",
    "cailing4@xdf.cn",
]


def check_admin_permission():
    user_email = st.session_state.get("e2_user_email", "")
    return user_email in ADMIN_EMAILS
