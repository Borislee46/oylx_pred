#!/usr/bin/env bash
# 一键部署脱敏开源版到 Ubuntu 22.04 / 24.04
#
# 用法（在服务器上，先上传整个仓库目录，然后）:
#   cd /opt/oylx-demo
#   sudo bash deploy/install_ubuntu.sh --domain demo.your-domain.com
#
# 可选参数:
#   --access-code <code>   访问码（不提供则随机生成并打印一次）
#   --no-access-code       沿用公开演示码 SIGNALS2026（仅限本地/内网试用）

set -euo pipefail

DOMAIN=""
ACCESS_CODE=""
USE_DEFAULT_CODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --access-code) ACCESS_CODE="$2"; shift 2 ;;
    --no-access-code) USE_DEFAULT_CODE=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$DOMAIN" ]]; then
  echo "必须提供 --domain（例如 --domain demo.example.com）"
  exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "请使用 sudo 运行本脚本"
  exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="oylx-demo"
ENV_FILE="$APP_DIR/.env"

if [[ "$USE_DEFAULT_CODE" == "0" && -z "$ACCESS_CODE" ]]; then
  ACCESS_CODE="$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-16)"
fi

echo "==> [1/7] 安装系统依赖（nginx / python / certbot / ufw）"
apt-get update -y
apt-get install -y python3 python3-venv python3-pip nginx certbot python3-certbot-nginx ufw openssl

echo "==> [2/7] 创建运行用户 $SERVICE_USER"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

echo "==> [3/7] 创建 Python 虚拟环境并安装依赖"
if [[ ! -d "$APP_DIR/.venv" ]]; then
  python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "==> [4/7] 生成 .env（会话签名密钥 + 访问码）"
if [[ ! -f "$ENV_FILE" ]]; then
  SECRET="$(openssl rand -hex 32)"
  if [[ "$USE_DEFAULT_CODE" == "1" ]]; then
    CODE_LINE="# DEMO_ACCESS_CODE 未设置 -> 使用公开演示码 SIGNALS2026"
  else
    CODE_LINE="DEMO_ACCESS_CODE=$ACCESS_CODE"
  fi
  cat > "$ENV_FILE" <<EOF
DEMO_COOKIE_SECRET=$SECRET
STREAMLIT_SERVER_COOKIE_SECRET=$SECRET
DEMO_TRUST_PROXY=1
APP_ENV=prod
$CODE_LINE
STREAMLIT_SERVER_ADDRESS=127.0.0.1
STREAMLIT_SERVER_PORT=8501

# DeepSeek 官方 API（个人 key）
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.deepseek.com/v1/chat/completions
OPENAI_MODEL=deepseek-v4-flash
# GHOST_API_KEY 供浏览器端「幽灵补全」使用，务必单独申请
# GHOST_API_KEY=
GHOST_API_BASE_URL=https://api.deepseek.com/beta
GHOST_API_MODEL=deepseek-v4-flash
EOF
  chmod 600 "$ENV_FILE"
  echo "    .env 已生成（权限 600）。请填入 OPENAI_API_KEY 后重启服务。"
else
  echo "    .env 已存在，跳过"
fi

echo "==> [5/7] 生成 .streamlit/config.toml（暗色主题 + cookieSecret）"
STREAMLIT_DIR="$APP_DIR/.streamlit"
STREAMLIT_CONF="$STREAMLIT_DIR/config.toml"
if [[ ! -f "$STREAMLIT_CONF" ]]; then
  mkdir -p "$STREAMLIT_DIR"
  if [[ -f "$STREAMLIT_DIR/config.toml.example" ]]; then
    cp "$STREAMLIT_DIR/config.toml.example" "$STREAMLIT_CONF"
  else
    printf '[theme]\nbase = "dark"\n\n[server]\nheadless = true\naddress = "127.0.0.1"\nport = 8501\n' \
      > "$STREAMLIT_CONF"
  fi
  # cookieSecret 必填：替换模板里的空值（不依赖 .env 是否被 Streamlit 读取）
  COOKIE_SECRET="$(openssl rand -hex 32)"
  if grep -q '^cookieSecret' "$STREAMLIT_CONF"; then
    sed -i "s|^cookieSecret.*|cookieSecret = \"$COOKIE_SECRET\"|" "$STREAMLIT_CONF"
  else
    sed -i "/^\[server\]/a cookieSecret = \"$COOKIE_SECRET\"" "$STREAMLIT_CONF"
  fi
  echo "    .streamlit/config.toml 已生成（主题 + 随机 cookieSecret）"
else
  echo "    .streamlit/config.toml 已存在，跳过"
fi

echo "==> [6/7] 配置 systemd 服务"
sed -e "s|__APP_DIR__|$APP_DIR|g" -e "s|__SERVICE_USER__|$SERVICE_USER|g" \
  "$APP_DIR/deploy/oylx-demo.service" > /etc/systemd/system/oylx-demo.service
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"
systemctl daemon-reload
systemctl enable oylx-demo
systemctl restart oylx-demo

echo "==> [7/7] 配置 nginx 反向代理 + HTTPS + 防火墙"
sed "s|__DOMAIN__|$DOMAIN|g" "$APP_DIR/deploy/nginx-ssl.conf.example" \
  > /etc/nginx/sites-available/oylx-demo
ln -sf /etc/nginx/sites-available/oylx-demo /etc/nginx/sites-enabled/oylx-demo
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx || systemctl restart nginx

echo "    -> 申请 HTTPS 证书并配置防火墙"
certbot --nginx -d "$DOMAIN" --redirect --non-interactive --agree-tos \
  -m "admin@$DOMAIN" || {
    echo "certbot 自动申请未成功，请稍后手动执行："
    echo "  certbot --nginx -d $DOMAIN --redirect"
  }
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo ""
echo "=============================================================="
echo " 部署完成：https://$DOMAIN"
if [[ "$USE_DEFAULT_CODE" == "1" ]]; then
  echo " 访问码：SIGNALS2026（公开演示码，请勿用于真实数据）"
else
  echo " 访问码：$ACCESS_CODE"
fi
echo " 服务状态：systemctl status oylx-demo"
echo " 环境变量：sudo nano $ENV_FILE  （填入 OPENAI_API_KEY 后 systemctl restart oylx-demo）"
echo "=============================================================="
