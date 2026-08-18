# Signals 留学择校预测（脱敏版）

本仓库是脱敏后的可部署版本：

- 保留选校预测系统（`app_pages/hk.py`）；
- 首页为演示页（`app_pages/home.py`）；
- 登录为严格本地认证：argon2id 密码哈希 + 可选 TOTP + 签名会话 Cookie +
  WebSocket 鉴权网关，无任何 DEBUG 后门；
- 真实密钥、邮箱白名单、原始案例数据均不进入 git（见 `.gitignore`）。

## 本地运行

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows no uvloop
# 或 .venv/bin/pip install -r requirements.txt  # Linux

# 1) 配置会话签名密钥
export DEMO_COOKIE_SECRET=$(openssl rand -hex 32)
export STREAMLIT_SERVER_COOKIE_SECRET=$DEMO_COOKIE_SECRET

# 2) 创建登录账号（交互输入密码）
.venv/bin/python deploy/tools/create_admin.py --username admin

# 3) 启动（需先在本地准备 src/ml/data、src/ml/pre-trained_models、cache 数据）
.venv/bin/streamlit run main.py
```

## 部署到 Ubuntu

见 [deploy/README.md](deploy/README.md)，运行：

```bash
sudo bash deploy/install_ubuntu.sh --domain demo.你的域名.com
```
