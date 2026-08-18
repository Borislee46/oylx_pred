import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import streamlit as st

from src.utils.auth.page_auth import get_current_nickname, get_current_username
from src.utils.logger import setup_logger
from src.utils.page_init import init_page

main_logger = setup_logger("page3", "home")

user_info = init_page(
    page_title="Signals · 脱敏演示环境",
    current_page_path="app_pages/home.py",
    layout="wide",
    hide_sidebar=True,
    require_whitelist=False,
)

st.markdown(
    """
    <style>
    .demo-hero {
      padding: 2.2rem 0 1.2rem 0;
    }
    .demo-hero h1 {
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: .3px;
      background: linear-gradient(90deg, #22d3ee, #a5f3fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: .4rem;
    }
    .demo-tag {
      display: inline-block;
      padding: .2rem .7rem;
      border-radius: 999px;
      background: rgba(6,182,212,.12);
      color: #67e8f9;
      font-size: .8rem;
      border: 1px solid rgba(6,182,212,.35);
    }
    .demo-card {
      background: #111a2c;
      border: 1px solid #1e293b;
      border-radius: 14px;
      padding: 1.3rem 1.4rem;
      height: 100%;
    }
    .demo-card .label {
      color: #64748b;
      font-size: .78rem;
      letter-spacing: .5px;
    }
    .demo-card .value {
      color: #e2e8f0;
      font-size: 1.05rem;
      font-weight: 600;
      margin-top: .35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="demo-hero">
      <div class="demo-tag">脱敏演示版 · Desensitized Demo</div>
      <h1>Signals 留学择校预测系统</h1>
      <p style="color:#94a3b8;margin-top:.4rem;">
        本环境仅用于功能演示。E2 企业登录与人力数据模块已移除，
        全部数据经过脱敏处理。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="demo-card">
          <div class="label">当前登录用户</div>
          <div class="value">{nickname}（{username}）</div>
        </div>
        """.format(nickname=get_current_nickname(), username=get_current_username()),
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="demo-card">
          <div class="label">已启用模块</div>
          <div class="value">选校预测系统</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="demo-card">
          <div class="label">已移除内容</div>
          <div class="value">E2 登录 · 人力数据</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.markdown("### 进入系统")
st.page_link("app_pages/hk.py", label="打开选校预测系统", icon=":material/school:")

st.divider()

st.markdown(
    """
    <div style="color:#64748b;font-size:.82rem;display:flex;gap:1.2rem;flex-wrap:wrap;">
      <span>登录会话为 24 小时有效签名 Cookie</span>
      <span>密码使用 argon2id 加密存储</span>
      <span><a href="/logout" style="color:#67e8f9;">退出登录</a></span>
    </div>
    """,
    unsafe_allow_html=True,
)

