"""Probe the local Aeloon Core WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from aiohttp import ClientSession, WSMsgType


async def probe(url: str, prompt: str | None) -> None:
    async with ClientSession() as session:
        async with session.ws_connect(url) as ws:
            if prompt:
                await ws.send_str(
                    json.dumps(
                        {
                            "id": "probe-chat",
                            "method": "chat.send",
                            "params": {"message": prompt},
                        }
                    )
                )
                seen: set[str] = set()
                async for message in ws:
                    if message.type != WSMsgType.TEXT:
                        continue
                    payload: dict[str, Any] = json.loads(message.data)
                    if payload.get("type") == "event":
                        event = str(payload.get("event"))
                        seen.add(event)
                        print(event)
                    if payload.get("type") == "response":
                        if "error" in payload:
                            raise SystemExit(payload["error"])
                        required = {"chat.turn.start", "chat.turn.end"}
                        missing = required - seen
                        if missing:
                            raise SystemExit(f"Missing events: {sorted(missing)}")
                        print("ok")
                        return
            else:
                await ws.send_str(json.dumps({"id": "probe-health", "method": "debug.health"}))
                while True:
                    message = await ws.receive(timeout=10)
                    if message.type != WSMsgType.TEXT:
                        continue
                    payload = json.loads(message.data)
                    if payload.get("type") != "response":
                        continue
                    if payload.get("result", {}).get("ok") is not True:
                        raise SystemExit(payload)
                    print(json.dumps(payload["result"], indent=2))
                    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/ws")
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()
    asyncio.run(probe(args.url, args.prompt))


if __name__ == "__main__":
    main()
