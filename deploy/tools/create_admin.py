#!/usr/bin/env python3
"""Create or update a login account for the desensitized app.

Passwords are stored only as argon2id hashes. Optionally enables TOTP 2FA.

Examples:
  python deploy/tools/create_admin.py --username admin
  python deploy/tools/create_admin.py --username demo --password 'S3cret!' --nickname 演示
  python deploy/tools/create_admin.py --username admin --gen-totp
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.auth.password import hash_password  # noqa: E402


def _read_users(path: Path) -> dict:
    if not path.exists():
        return {"users": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"users": {}}
    if not isinstance(raw, dict) or not isinstance(raw.get("users"), dict):
        return {"users": {}}
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="登录用户名")
    parser.add_argument("--password", help="明文密码（不提供则交互输入）")
    parser.add_argument("--nickname", default="演示用户", help="显示昵称")
    parser.add_argument("--totp-secret", help="已生成的 TOTP base32 密钥（可选）")
    parser.add_argument("--gen-totp", action="store_true", help="生成随机 TOTP 密钥并输出 otpauth URI")
    parser.add_argument(
        "--out",
        default=os.environ.get("DEMO_USERS_PATH", "config/demo_users.json"),
        help="输出文件（默认 config/demo_users.json）",
    )
    args = parser.parse_args()

    password = args.password
    if not password:
        password = getpass.getpass("请输入新密码（至少 8 位）: ")
        confirm = getpass.getpass("再次输入确认: ")
        if password != confirm:
            print("两次输入不一致，已取消。", file=sys.stderr)
            return 1
    if len(password) < 8:
        print("密码长度至少 8 位，已取消。", file=sys.stderr)
        return 1

    totp_secret = args.totp_secret or ""
    if args.gen_totp:
        import pyotp

        totp_secret = pyotp.random_base32()
        print("=" * 60)
        print("TOTP 密钥（请用 Authenticator 应用扫描/录入）:")
        print(pyotp.totp.TOTP(totp_secret).provisioning_uri(name=args.username, issuer_name="Signals Demo"))
        print("=" * 60)

    out_path = Path(args.out).resolve()
    data = _read_users(out_path)
    data["users"][args.username] = {
        "password_hash": hash_password(password),
        "nickname": args.nickname,
        "totp_secret": totp_secret,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(out_path, 0o600)
    print(f"账号已写入 {out_path}（仅保存 argon2id 哈希，不保存明文）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

