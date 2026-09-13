"""访问码门禁：开源演示版用它替代企业 SSO / 账号密码体系。

设计目标：
- 零配置即可启动：未设置环境变量时使用内置演示访问码 `SIGNALS2026`；
- 生产部署通过 `DEMO_ACCESS_CODE` 覆盖，不把真实口令写进仓库；
- 比较使用 `hmac.compare_digest`，避免时序侧信道；
- **访问码永远不出现在页面上**——只显示访问码的门禁不是门禁。
"""

from __future__ import annotations

import hmac
import logging
import os

_log = logging.getLogger(__name__)

DEFAULT_ACCESS_CODE = "SIGNALS2026"
ACCESS_CODE_ENV = "DEMO_ACCESS_CODE"

_warned_default = False


def current_access_code() -> str:
    """返回当前生效的访问码（环境变量优先）。"""
    global _warned_default
    code = os.environ.get(ACCESS_CODE_ENV, "").strip()
    if code:
        return code
    if not _warned_default:
        _warned_default = True
        _log.warning(
            "%s 未设置，正在使用项目 README 里公开的默认访问码；"
            "对外部署请务必设置该环境变量，否则门禁形同虚设。",
            ACCESS_CODE_ENV,
        )
    return DEFAULT_ACCESS_CODE


def verify_access_code(candidate: str | None) -> bool:
    """常量时间比较访问码。"""
    if not candidate:
        return False
    return hmac.compare_digest(candidate.strip(), current_access_code())


def access_code_is_default() -> bool:
    """当前是否在用公开的默认访问码（供部署自检 / 监控使用）。"""
    return current_access_code() == DEFAULT_ACCESS_CODE
