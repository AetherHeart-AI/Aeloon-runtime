#!/usr/bin/env python3
"""Terminate leftover R3 harness processes that still advertise ownership.

Only processes whose environ contains ``AELOON_R3_OWNED=<prefix...>`` and whose
cmdline is a known harness entry are signalled. Historical PID files are never
trusted. This script must not set ``AELOON_R3_OWNED`` on itself.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

MARKER = b"AELOON_R3_OWNED="
OWNED_COMMANDS = (
    b"r3_test_server.py",
    b"r3_runtime_bench.py",
)


def is_owned(environ: bytes, cmdline: bytes, prefix: str) -> bool:
    needle = prefix.encode("utf-8")
    owned = False
    for item in environ.split(b"\0"):
        if item.startswith(MARKER) and item[len(MARKER) :].startswith(needle):
            owned = True
            break
    return owned and any(token in cmdline for token in OWNED_COMMANDS)


def owned_pids(prefix: str) -> list[int]:
    self_pid = os.getpid()
    found: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        try:
            environ = (entry / "environ").read_bytes()
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if is_owned(environ, cmdline, prefix):
            found.append(pid)
    return found


def terminate(prefix: str, *, sigterm: int = signal.SIGTERM) -> list[int]:
    pids = owned_pids(prefix)
    for pid in pids:
        try:
            os.kill(pid, sigterm)
        except OSError:
            continue
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and owned_pids(prefix):
        time.sleep(0.05)
    remaining = owned_pids(prefix)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue
    return pids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prefix = args.prefix
    if not prefix.startswith("/"):
        print("prefix must be an absolute path", file=sys.stderr)
        return 2
    if args.dry_run:
        for pid in owned_pids(prefix):
            print(pid)
        return 0
    terminate(prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
