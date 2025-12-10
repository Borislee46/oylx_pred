import os

import streamlit as st
import streamlit.components.v1 as components

LOGO_PATH = "assets/company_logo.png"

HEADER_STYLES = """
<style>
[data-testid="stAppViewContainer"] {
    background: 
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(0, 106, 96, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(0, 106, 96, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse 40% 40% at 50% 10%, rgba(0, 138, 108, 0.04) 0%, transparent 40%),
        linear-gradient(180deg, rgba(248, 250, 252, 1) 0%, rgba(255, 255, 255, 1) 100%);
    background-attachment: fixed;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: 
        radial-gradient(rgba(0, 106, 96, 0.03) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stAppViewContainer"] > div:first-child {
    position: relative;
    z-index: 1;
}

@keyframes header-line-flow-left {
    0% { background-position: 100% 0; }
    100% { background-position: -100% 0; }
}
@keyframes header-line-flow-right {
    0% { background-position: -100% 0; }
    100% { background-position: 100% 0; }
}
.header-line-left {
    display: inline-block;
    height: 1px;
    width: 40px;
    background: linear-gradient(90deg, transparent, rgb(0, 106, 96), transparent);
    background-size: 200% 100%;
    animation: header-line-flow-left 2s linear infinite;
}
.header-line-right {
    display: inline-block;
    height: 1px;
    width: 40px;
    background: linear-gradient(270deg, transparent, rgb(0, 106, 96), transparent);
    background-size: 200% 100%;
    animation: header-line-flow-right 2s linear infinite;
}
</style>
"""


def render_header(user_nickname: str) -> None:
    st.markdown('<div id="main-page-header-anchor"></div>', unsafe_allow_html=True)

    st.markdown(HEADER_STYLES, unsafe_allow_html=True)

    if os.path.exists(LOGO_PATH):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(LOGO_PATH, width="stretch")

    st.markdown(
        """
        <div style='text-align: center; padding: 10px 0 20px 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;'>
            <h1 style='background: linear-gradient(135deg, #004d47 0%, #00695F 50%, #00897b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600; font-size: 2.2rem; margin: 0 0 12px 0; letter-spacing: -0.5px;'>欧亚数据科学平台</h1>
            <div style='display: inline-flex; align-items: center; justify-content: center; gap: 10px;'>
                <span class="header-line-left"></span>
                <p style='color: #64748b; font-size: 14px; margin: 0; letter-spacing: 1.5px; font-weight: 500;'>AI驱动的智能决策与个性化数据服务</p>
                <span class="header-line-right"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    components.html(
        """
        <script>
            setTimeout(() => {
                try {
                    const anchor = window.parent.document.getElementById('main-page-header-anchor');
                    if (anchor) {
                        anchor.scrollIntoView({behavior: 'smooth', block: 'start'});
                    }
                } catch (e) {
                    console.warn('Auto-scroll failed:', e);
                }
            }, 50);
        </script>
        """,
        height=0,
        width=0,
    )
