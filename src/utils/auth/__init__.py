"""自包含的访问码门禁，用于开源演示版。

设计目标：
- 不依赖任何企业 SSO / 账号体系；
- 仅一个演示访问码（默认 `SIGNALS2026`，可用环境变量 `DEMO_ACCESS_CODE` 覆盖）；
- 会话为 itsdangerous 签名 Cookie，未配置密钥时由访问码派生，保证零配置可启动；
- WebSocket 握手校验同一 Cookie，避免绕过 HTTP 门禁直连；
- 没有 DEBUG 免登录开关，任何配置都无法旁路门禁。
"""
