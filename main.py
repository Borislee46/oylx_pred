
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starlette.middleware import Middleware

from src.utils.auth.http import AuthEnforcementMiddleware, auth_routes
from src.utils.ws_auth import install_ws_auth_gate

install_ws_auth_gate()

import streamlit as st


def _parse_flags() -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server.port", dest="server_port", type=int, default=None)
    parser.add_argument("--server.address", dest="server_address", default=None)
    parser.add_argument("--server.headless", dest="server_headless", default=None)
    args = parser.parse_args()

    flags: dict[str, object] = {}
    if args.server_port is not None:
        flags["server.port"] = args.server_port
    if args.server_address:
        flags["server.address"] = args.server_address
    if args.server_headless is not None:
        flags["server.headless"] = args.server_headless.lower() in ("1", "true", "yes")
    return flags


app = st.App(
    str(ROOT / "server.py"),
    routes=auth_routes(),
    middleware=[Middleware(AuthEnforcementMiddleware)],
)


def main() -> None:
    app.run(config=_parse_flags())


if __name__ == "__main__":
    main()
