# Ubuntu 正式服务器部署指南（脱敏开源版）

## 一、服务器要求

- Ubuntu 22.04 LTS 或 24.04 LTS
- 公网域名（需先解析到服务器 IP）
- 开放端口：22 / 80 / 443
- Python ≥ 3.11（脚本会自动安装）

## 二、上传代码

把整个仓库目录上传到服务器，例如放到 `/opt/oylx-demo`：

```bash
scp -r oylx_pred ubuntu@服务器IP:/tmp/
sudo mkdir -p /opt/oylx-demo
sudo cp -r /tmp/oylx_pred/* /opt/oylx-demo/
```

注意：仓库中不包含模型权重与案例数据（`src/ml/data`、`src/ml/pre-trained_models`、
`cache/background_target_similarity.feather`），部署前需单独上传这些文件到对应目录。

## 三、一键部署

```bash
cd /opt/oylx-demo
sudo bash deploy/install_ubuntu.sh --domain demo.你的域名.com
```

脚本会依次完成：系统依赖 → 运行用户 → venv 依赖 → 生成 `.env`（会话密钥 + 随机访问码）
→ 生成 `.streamlit/config.toml`（暗色主题 + 随机 cookieSecret）→ systemd 服务 →
nginx 反代 → HTTPS 证书 → 防火墙。

访问码想自己指定时加 `--access-code`；仅本地试用可加 `--no-access-code` 沿用公开演示码
`SIGNALS2026`。

部署完成后编辑 `.env` 填入 DeepSeek key，再 `sudo systemctl restart oylx-demo`。

### 为什么必须生成 `.streamlit/config.toml`

仓库只提交 `config.toml.example`（真配置含 `cookieSecret`，不能入库）。如果服务器上缺这个文件，
Streamlit 会退回**默认亮色主题**——页面样式是按暗色底设计的，会整体显示错乱。安装脚本会自动补齐。

## 四、安全设计

| 项目 | 实现 |
| --- | --- |
| 门禁 | 单一访问码（`DEMO_ACCESS_CODE`），默认演示码 `SIGNALS2026` |
| 会话 | itsdangerous 签名 Cookie（HttpOnly / SameSite=Lax / Secure），24 小时有效 |
| 签名密钥 | 生产建议显式注入 `DEMO_COOKIE_SECRET`；未配置时由访问码派生 |
| WebSocket | 握手校验同一 Cookie，防止绕过 HTTP 鉴权直连 |
| 暴力破解 | 同 IP 每小时 20 次失败锁定 |
| CSRF | 登录表单校验 Cookie 绑定 token，Cookie 为 SameSite |
| 后门 | 已删除 DEBUG_MODE 自动登录、企业 SSO 免密回调、邮箱白名单等旁路 |
| 端口 | 应用仅监听 127.0.0.1:8501，公网只暴露 nginx 443/80 |

> 访问码是唯一凭据，公开演示码不构成任何数据保护。部署到公网请务必使用
> `--access-code` 或手工设置私密访问码。

## 五、日常运维

```bash
systemctl status oylx-demo
journalctl -u oylx-demo -f
sudo systemctl restart oylx-demo

# 更换访问码
sudo sed -i 's/^DEMO_ACCESS_CODE=.*/DEMO_ACCESS_CODE=新访问码/' /opt/oylx-demo/.env
sudo systemctl restart oylx-demo
```

## 六、脱敏范围

- 移除企业 SSO 登录（OAuth 回调路由、白名单校验、DEBUG_MODE 自动登录），替换为访问码门禁；
- 移除人力数据模块（HR 页面、人员/薪酬/编制/离职数据、`hr_permissions.json`）；
- 移除真实邮箱白名单、合同邮箱映射、真实员工姓名、内部报表链接；
- 移除企业内部 LLM 网关与全部真实 API 密钥，统一改用 DeepSeek 官方 API；
- 移除含学生姓名、合同编号、顾问姓名的原始案例库与训练产物；
- 选校预测系统完整保留（院校、专业、GPA、语言、科研等均为脱敏特征数据）。
