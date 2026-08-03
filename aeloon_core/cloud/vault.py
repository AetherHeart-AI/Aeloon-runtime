"""Refresh-token storage owned exclusively by Core."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Protocol


class TokenVault(Protocol):
    kind: str

    def save(self, token: str) -> None: ...

    def load(self) -> str | None: ...

    def delete(self) -> None: ...


class InMemoryTokenVault:
    kind = "memory"

    def __init__(self) -> None:
        self.token: str | None = None

    def save(self, token: str) -> None:
        self.token = token

    def load(self) -> str | None:
        return self.token

    def delete(self) -> None:
        self.token = None


class FileTokenVault:
    """0600 fallback for platforms without a native credential vault."""

    kind = "file"

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, token: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.write_text(json.dumps({"refresh_token": token}), encoding="utf-8")
        if os.name != "nt":
            os.chmod(self.path, 0o600)

    def load(self) -> str | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        token = payload.get("refresh_token") if isinstance(payload, dict) else None
        return token if isinstance(token, str) and token else None

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class MacOSKeychainVault:
    kind = "macos-keychain"

    def __init__(self, account: str, *, service: str = "dev.aeloon.core.cloud") -> None:
        self.account = account
        self.service = service
        self.security = "/usr/bin/security"

    @classmethod
    def available(cls) -> bool:
        return platform.system() == "Darwin" and bool(shutil.which("security"))

    def save(self, token: str) -> None:
        subprocess.run(
            [
                self.security,
                "add-generic-password",
                "-a",
                self.account,
                "-s",
                self.service,
                "-w",
                token,
                "-U",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def load(self) -> str | None:
        result = subprocess.run(
            [self.security, "find-generic-password", "-a", self.account, "-s", self.service, "-w"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def delete(self) -> None:
        subprocess.run(
            [self.security, "delete-generic-password", "-a", self.account, "-s", self.service],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def default_token_vault(data_dir: Path, account: str) -> TokenVault:
    if MacOSKeychainVault.available():
        return MacOSKeychainVault(account)
    return FileTokenVault(data_dir / "cloud-refresh-token.json")
