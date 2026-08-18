# Ubuntu 正式服务器部署指南（脱敏演示版）

## 一、服务器要求

- Ubuntu 22.04 LTS 或 24.04 LTS
- 公网域名（需先解析到服务器 IP）
- 开放端口：22 / 80 / 443
- Python ≥ 3.11（脚本会自动安装）

## 二、上传代码

把整个仓库目录上传到服务器，例如放到 `/opt/oylx-demo`：

```bash
scp -r oylx_demo ubuntu@服务器IP:/tmp/
sudo mkdir -p /opt/oylx-demo
sudo cp -r /tmp/oylx_demo/* /opt/oylx-demo/
```

注意：仓库中不包含模型权重与案例数据（`src/ml/data`、`src/ml/pre-trained_models`、
`cache/background_target_similarity.feather`），部署前需单独上传这些文件到对应目录。

## 三、一键部署

```bash
cd /opt/oylx-demo
sudo bash deploy/install_ubuntu.sh --domain demo.你的域名.com
```

脚本会依次完成：系统依赖 → 运行用户 → venv 依赖 → 生成会话密钥 `.env` →
创建 argon2id 登录账号 → systemd 服务 → nginx 反代 → HTTPS 证书 → 防火墙。

## 四、安全设计

| 项目 | 实现 |
| --- | --- |
| 密码存储 | argon2id 哈希（`argon2-cffi`），绝不保存明文 |
| 二次验证 | 可选 TOTP（`pyotp`） |
| 会话 | itsdangerous 签名 Cookie（HttpOnly / SameSite / Secure），24 小时有效 |
| 签名密钥 | 必须通过环境变量注入，缺失时应用拒绝启动 |
| WebSocket | 握手校验同一 Cookie，防止绕过 HTTP 鉴权直连 |
| 暴力破解 | 同 IP+用户名 5 次失败锁定 15 分钟；同 IP 每小时限 30 次 |
| CSRF | 登录表单校验 Cookie 绑定 token，Cookie 为 SameSite |
| 后门 | 已删除 DEBUG_MODE 自动登录、E2 免密回调、邮箱白名单等旁路 |
| 端口 | 应用仅监听 127.0.0.1:8501，公网只暴露 nginx 443/80 |

## 五、日常运维

```bash
systemctl status oylx-demo
journalctl -u oylx-demo -f
sudo systemctl restart oylx-demo

# 修改/新增账号
sudo .venv/bin/python deploy/tools/create_admin.py --username 账号
sudo .venv/bin/python deploy/tools/create_admin.py --username admin --gen-totp
```

## 六、脱敏范围

- 移除 E2 企业登录（`e2_*`、回调路由、白名单校验、DEBUG_MODE 自动登录）；
- 移除人力数据模块（HR 页面、人员/薪酬/编制/离职数据、`hr_permissions.json`）；
- 移除真实邮箱白名单、合同邮箱映射、真实员工姓名；
- 移除 E2 / OpenAI / DeepSeek 等真实 API 密钥；
- 移除含学生姓名、合同编号、顾问姓名的原始案例库；
- 选校预测系统完整保留（院校、专业、GPA、语言、科研等均为脱敏特征数据）。

