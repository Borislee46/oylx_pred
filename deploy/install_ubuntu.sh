#!/usr/bin/env bash
# 一键部署脱敏版到 Ubuntu 22.04 / 24.04
#
# 用法（在服务器上，先上传整个仓库目录，然后）:
#   cd /opt/oylx-demo
#   sudo bash deploy/install_ubuntu.sh --domain demo.your-domain.com
#
# 可选参数:
#   --admin-user  <name>      登录用户名（默认 admin）
#   --admin-password <pass>   初始密码（不提供则交互输入）
#   --totp                     同时生成 TOTP 二次验证密钥

set -euo pipefail

DOMAIN=""
ADMIN_USER="admin"
ADMIN_PASSWORD=""
GEN_TOTP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --admin-user) ADMIN_USER="$2"; shift 2 ;;
    --admin-password) ADMIN_PASSWORD="$2"; shift 2 ;;
    --totp) GEN_TOTP=1; shift ;;
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
USERS_FILE="$APP_DIR/config/demo_users.json"

echo "==> [1/8] 安装系统依赖（nginx / python / certbot / ufw）"
apt-get update -y
apt-get install -y python3 python3-venv python3-pip nginx certbot python3-certbot-nginx ufw openssl

echo "==> [2/8] 创建运行用户 $SERVICE_USER"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

echo "==> [3/8] 创建 Python 虚拟环境并安装依赖"
if [[ ! -d "$APP_DIR/.venv" ]]; then
  python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "==> [4/8] 生成会话签名密钥 .env"
if [[ ! -f "$ENV_FILE" ]]; then
  SECRET="$(openssl rand -hex 32)"
  cat > "$ENV_FILE" <<EOF
DEMO_COOKIE_SECRET=$SECRET
STREAMLIT_SERVER_COOKIE_SECRET=$SECRET
DEMO_TRUST_PROXY=1
APP_ENV=prod
DEMO_USERS_PATH=$USERS_FILE
STREAMLIT_SERVER_ADDRESS=127.0.0.1
STREAMLIT_SERVER_PORT=8501

# 个人 AI key（统一一个 key，OPENAI 与 GHOST 幽灵补全共用）
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.deepseek.com/v1/chat/completions
OPENAI_MODEL=deepseek-v4-flash
# GHOST_API_KEY 可省略，缺省复用 OPENAI_API_KEY
# GHOST_API_KEY=
GHOST_API_BASE_URL=https://api.deepseek.com/beta
GHOST_API_MODEL=deepseek-v4-flash
EOF
  chmod 600 "$ENV_FILE"
  echo "    .env 已生成（权限 600）"
else
  echo "    .env 已存在，跳过"
fi

echo "==> [5/8] 创建登录账号（argon2id 加密存储）"
if [[ ! -f "$USERS_FILE" ]]; then
  TOTP_ARG=()
  if [[ "$GEN_TOTP" == "1" ]]; then TOTP_ARG+=(--gen-totp); fi
  if [[ -n "$ADMIN_PASSWORD" ]]; then
    "$APP_DIR/.venv/bin/python" "$APP_DIR/deploy/tools/create_admin.py" \
      --username "$ADMIN_USER" --password "$ADMIN_PASSWORD" --nickname "管理员" \
      --out "$USERS_FILE" "${TOTP_ARG[@]}"
  else
    "$APP_DIR/.venv/bin/python" "$APP_DIR/deploy/tools/create_admin.py" \
      --username "$ADMIN_USER" --nickname "管理员" \
      --out "$USERS_FILE" "${TOTP_ARG[@]}"
  fi
else
  echo "    账号文件已存在，跳过（如需修改请运行 create_admin.py）"
fi

echo "==> [6/8] 配置 systemd 服务"
sed -e "s|__APP_DIR__|$APP_DIR|g" -e "s|__SERVICE_USER__|$SERVICE_USER|g" \
  "$APP_DIR/deploy/oylx-demo.service" > /etc/systemd/system/oylx-demo.service
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"
systemctl daemon-reload
systemctl enable oylx-demo
systemctl restart oylx-demo

echo "==> [7/8] 配置 nginx 反向代理"
sed "s|__DOMAIN__|$DOMAIN|g" "$APP_DIR/deploy/nginx-ssl.conf.example" \
  > /etc/nginx/sites-available/oylx-demo
ln -sf /etc/nginx/sites-available/oylx-demo /etc/nginx/sites-enabled/oylx-demo
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx || systemctl restart nginx

echo "==> [8/8] 申请 HTTPS 证书并配置防火墙"
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
echo " 登录账号：$ADMIN_USER"
echo " 服务状态：systemctl status oylx-demo"
echo " 修改密码：sudo $APP_DIR/.venv/bin/python $APP_DIR/deploy/tools/create_admin.py --username $ADMIN_USER"
echo "=============================================================="
