"""Small Runtime-owned PTY manager used by the local terminal methods."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import os
import pty
import signal
import struct
import termios
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .blocking import run_blocking

TerminalEvent = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(slots=True)
class _Terminal:
    terminal_id: str
    thread_id: str
    master_fd: int
    pid: int
    callback: asyncio.Handle | None = None
    finishing: bool = False


class PTYManager:
    def __init__(self, on_event: TerminalEvent) -> None:
        self._on_event = on_event
        self._terminals: dict[str, _Terminal] = {}
        self._by_thread: dict[str, str] = {}
        self._loop = asyncio.get_running_loop()

    async def open(
        self, thread_id: str, workspace: Path, columns: int = 120, rows: int = 36
    ) -> dict[str, Any]:
        if columns <= 0 or rows <= 0:
            raise ValueError("terminal dimensions must be positive")
        existing = self._by_thread.get(thread_id)
        if existing is not None:
            return {"opened": True, "terminal_id": existing, "columns": columns, "rows": rows}
        master, slave = pty.openpty()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child process
            try:
                os.setsid()
                os.close(master)
                os.dup2(slave, 0)
                os.dup2(slave, 1)
                os.dup2(slave, 2)
                if slave > 2:
                    os.close(slave)
                os.chdir(workspace)
                shell = os.environ.get("SHELL") or "/bin/sh"
                os.execvp(shell, [shell, "-l"])
            except BaseException:
                os._exit(127)
        os.close(slave)
        os.set_blocking(master, False)
        terminal_id = str(uuid.uuid4())
        terminal = _Terminal(terminal_id, thread_id, master, pid)
        self._terminals[terminal_id] = terminal
        self._by_thread[thread_id] = terminal_id
        self._resize_fd(master, columns, rows)
        self._loop.add_reader(master, self._read_ready, terminal_id)
        await self._emit(
            "terminal.opened", thread_id, terminal_id, {"columns": columns, "rows": rows}
        )
        return {"opened": True, "terminal_id": terminal_id, "columns": columns, "rows": rows}

    async def input(self, thread_id: str, data: str) -> bool:
        terminal = self._terminal_for_thread(thread_id)
        try:
            os.write(terminal.master_fd, data.encode("utf-8"))
        except OSError as exc:
            if exc.errno in {errno.EIO, errno.EBADF}:
                return False
            raise
        return True

    async def resize(self, thread_id: str, columns: int, rows: int) -> bool:
        terminal = self._terminal_for_thread(thread_id)
        self._resize_fd(terminal.master_fd, columns, rows)
        return True

    async def close(self, thread_id: str) -> bool:
        terminal_id = self._by_thread.get(thread_id)
        if terminal_id is None:
            return False
        return await self.close_terminal(terminal_id)

    async def close_terminal(self, terminal_id: str) -> bool:
        terminal = self._terminals.get(terminal_id)
        if terminal is None:
            return False
        if terminal.finishing:
            return False
        terminal.finishing = True
        self._terminals.pop(terminal_id, None)
        self._by_thread.pop(terminal.thread_id, None)
        self._loop.remove_reader(terminal.master_fd)
        with contextlib.suppress(OSError):
            os.kill(terminal.pid, signal.SIGHUP)
        with contextlib.suppress(OSError):
            os.close(terminal.master_fd)
        exit_payload = await self._wait_for_child(terminal.pid, status="closed")
        await self._emit("terminal.exit", terminal.thread_id, terminal_id, exit_payload)
        return True

    async def close_all(self) -> None:
        for terminal_id in tuple(self._terminals):
            await self.close_terminal(terminal_id)

    def _read_ready(self, terminal_id: str) -> None:
        terminal = self._terminals.get(terminal_id)
        if terminal is None or terminal.finishing:
            return
        try:
            data = os.read(terminal.master_fd, 64 * 1024)
        except OSError as exc:
            if exc.errno not in {errno.EIO, errno.EBADF}:
                return
            data = b""
        if data:
            self._loop.create_task(
                self._emit(
                    "terminal.output",
                    terminal.thread_id,
                    terminal_id,
                    {"data": data.decode("utf-8", errors="replace")},
                )
            )
            return
        terminal.finishing = True
        self._loop.create_task(self._finish_after_eof(terminal))

    async def _finish_after_eof(self, terminal: _Terminal) -> None:
        self._loop.remove_reader(terminal.master_fd)
        with contextlib.suppress(OSError):
            os.close(terminal.master_fd)
        self._terminals.pop(terminal.terminal_id, None)
        self._by_thread.pop(terminal.thread_id, None)
        exit_payload = await self._wait_for_child(terminal.pid, status="exited")
        await self._emit(
            "terminal.exit", terminal.thread_id, terminal.terminal_id, exit_payload
        )

    async def _wait_for_child(self, pid: int, *, status: str) -> dict[str, Any]:
        deadline = self._loop.time() + 1.0
        child_status: int | None = None
        while self._loop.time() < deadline:
            try:
                waited_pid, child_status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                break
            if waited_pid == pid:
                break
            await asyncio.sleep(0.01)
        if child_status is None:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError, OSError):
                _waited_pid, child_status = await run_blocking(os.waitpid, pid, 0)
        payload: dict[str, Any] = {"status": status}
        if child_status is not None:
            if os.WIFEXITED(child_status):
                payload["exit_code"] = os.WEXITSTATUS(child_status)
            elif os.WIFSIGNALED(child_status):
                payload["signal"] = os.WTERMSIG(child_status)
        return payload

    async def _emit(
        self, name: str, thread_id: str, terminal_id: str, payload: dict[str, Any]
    ) -> None:
        value = {
            "name": name,
            "thread_id": thread_id,
            "terminal_id": terminal_id,
            "payload": payload,
        }
        result = self._on_event(value)
        if asyncio.iscoroutine(result):
            await result

    def _terminal_for_thread(self, thread_id: str) -> _Terminal:
        terminal_id = self._by_thread.get(thread_id)
        if terminal_id is None or terminal_id not in self._terminals:
            raise RuntimeError("terminal_not_found")
        return self._terminals[terminal_id]

    @staticmethod
    def _resize_fd(fd: int, columns: int, rows: int) -> None:
        if columns <= 0 or rows <= 0:
            raise ValueError("terminal dimensions must be positive")
        packed = struct.pack("HHHH", rows, columns, 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)


__all__ = ["PTYManager"]
