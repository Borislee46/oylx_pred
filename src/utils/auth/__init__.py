"""Strict, self-contained authentication for the desensitized app.

Design goals:
- No E2 / SSO dependency.
- Passwords stored only as argon2id hashes (never plaintext).
- Optional TOTP second factor (per user).
- Session cookie signed with itsdangerous + a required server secret.
- WebSocket handshake gated by the same cookie (no sneak-in path).
- No debug-mode bypass: login cannot be skipped by config or env flag.
"""

