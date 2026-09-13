import streamlit as st


def scroll_to_anchor(anchor_id: str, delay_ms: int = 150) -> None:
    st.markdown(
        f'<div id="{anchor_id}"></div>'
        "<script>"
        "setTimeout(function(){"
        f'var el=document.getElementById("{anchor_id}");'
        'if(el)el.scrollIntoView({behavior:"smooth",block:"start"});'
        f"}},{delay_ms});"
        "</script>",
        unsafe_allow_html=True,
    )
