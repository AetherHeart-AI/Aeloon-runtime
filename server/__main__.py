"""Run the Aeloon Core local web server."""

from __future__ import annotations

import argparse
from pathlib import Path

from aiohttp import web

from aeloon_core.config import load_config
from server.app import create_app


def build_parser() -> argparse.ArgumentParser:
    """Build the server CLI parser."""

    parser = argparse.ArgumentParser(description="Run the Aeloon Core web server.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    return parser


def main() -> None:
    """Run the server."""

    args = build_parser().parse_args()
    config = load_config(args.config)
    app = create_app(config)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
