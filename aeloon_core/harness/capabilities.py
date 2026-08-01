"""Host capabilities composed around the Pi agent runtime."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import signal
import stat
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from aeloon_core.config import Config
from aeloon_core.harness.tool import FunctionTool, GlobTool, GrepTool, ListTool, Tool

HARNESS_PROTECTED_PATTERNS = (
    ".aeloon-core/*",
    ".git/*",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "**/secrets*",
)
DENIED_ENV_PATTERNS = (
    "ANTHROPIC_*",
    "OPENAI_*",
    "DEEPSEEK_*",
    "GOOGLE_*",
    "GEMINI_*",
    "GROQ_*",
    "MISTRAL_*",
    "OPENROUTER_*",
    "ARK_*",
    "AELOON_CORE_API_KEY",
    "*API_KEY*",
    "*CREDENTIAL*",
    "*PASSWORD*",
    "*SECRET*",
    "*TOKEN*",
    "DATABASE_URL",
    "SSH_AUTH_SOCK",
)
DENIED_SHELL_COMMANDS = (
    "rm",
    "rmdir",
    "mkfs",
    "dd",
    "format",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init",
)


class CapabilityUnavailable(RuntimeError):
    """Raised when an optional Expert capability is not configured."""


class RuntimeCapability:
    """Capability descriptor understood by the Python/Pi bridge."""

    def runtime_config(self) -> dict[str, Any] | None:
        return None

    def host_tools(self) -> tuple[Tool, ...]:
        return ()

    async def close(self) -> None:
        """Release turn-local resources owned by the capability."""


@dataclass(frozen=True, slots=True)
class SlidingWindow(RuntimeCapability):
    """Zero-LLM context compaction policy applied before Pi provider calls."""

    max_tokens: int
    keep_tokens: int
    preserve_first_user_message: bool = True

    def runtime_config(self) -> dict[str, Any]:
        return {
            "kind": "sliding_window",
            "max_tokens": self.max_tokens,
            "keep_tokens": self.keep_tokens,
            "preserve_first_user_message": self.preserve_first_user_message,
        }


@dataclass(frozen=True, slots=True)
class FileSystem(RuntimeCapability):
    """Pi Core read/write/edit tools rooted at the configured workspace."""

    root_dir: Path
    protected_patterns: tuple[str, ...] = HARNESS_PROTECTED_PATTERNS

    def runtime_config(self) -> dict[str, Any]:
        return {
            "kind": "filesystem",
            "cwd": str(self.root_dir),
            "protected_patterns": list(self.protected_patterns),
            "tool_names": {
                "read": "read_file",
                "write": "write_file",
                "edit": "edit_file",
            },
        }

    def host_tools(self) -> tuple[Tool, ...]:
        options = {
            "workspace": self.root_dir,
            "confine_to_workspace": True,
        }
        list_tool = ListTool(**options)
        glob_tool = GlobTool(**options)
        grep_tool = GrepTool(**options)

        async def list_directory(path: str = ".") -> str:
            return await list_tool.execute(path=path)

        async def find_files(pattern: str, path: str = ".") -> str:
            return await glob_tool.execute(pattern=pattern, root=path)

        async def search_files(
            pattern: str,
            path: str = ".",
            include_glob: str | None = None,
        ) -> str:
            return await grep_tool.execute(
                pattern=pattern,
                path=path,
                include=include_glob,
            )

        async def create_directory(path: str) -> str:
            resolved = self._resolve(path, write=True)
            resolved.mkdir(parents=True, exist_ok=True)
            return f"Created directory: {path}"

        async def file_info(path: str) -> str:
            resolved = self._resolve(path, write=False)
            if not resolved.exists():
                return f"Error: Path not found: {path}"
            info = resolved.stat()
            kind = "directory" if resolved.is_dir() else "file"
            lines = [
                f"path: {path}",
                f"type: {kind}",
                f"size: {info.st_size} bytes",
                f"mode: {stat.filemode(info.st_mode)}",
            ]
            if resolved.is_file():
                content = resolved.read_text(encoding="utf-8", errors="replace")
                lines.append(f"lines: {len(content.splitlines())}")
            return "\n".join(lines)

        return (
            FunctionTool(
                name="list_directory",
                description="List one workspace directory.",
                args_model=_ListDirectoryArgs,
                handler=list_directory,
                concurrency_mode="read_only",
            ),
            FunctionTool(
                name="find_files",
                description="Find workspace files by glob pattern.",
                args_model=_FindFilesArgs,
                handler=find_files,
                concurrency_mode="read_only",
            ),
            FunctionTool(
                name="search_files",
                description="Search workspace file contents with a regular expression.",
                args_model=_SearchFilesArgs,
                handler=search_files,
                concurrency_mode="read_only",
            ),
            FunctionTool(
                name="create_directory",
                description="Create a workspace directory and any missing parents.",
                args_model=_PathArgs,
                handler=create_directory,
                concurrency_mode="mutating",
            ),
            FunctionTool(
                name="file_info",
                description="Return metadata for one workspace file or directory.",
                args_model=_PathArgs,
                handler=file_info,
                concurrency_mode="read_only",
            ),
        )

    def _resolve(self, path: str, *, write: bool) -> Path:
        root = self.root_dir.expanduser().resolve(strict=False)
        candidate = (root / path).resolve(strict=False)
        if candidate != root and not candidate.is_relative_to(root):
            raise PermissionError(f"path escapes the workspace: {path}")
        relative = candidate.relative_to(root).as_posix()
        if write and any(
            _matches_protected_path(relative, pattern)
            for pattern in self.protected_patterns
        ):
            raise PermissionError(f"path is protected from agent tools: {path}")
        return candidate


@dataclass(slots=True)
class Shell(RuntimeCapability):
    """Pi Core bash tool with host-owned environment filtering."""

    cwd: Path
    default_timeout: float
    denied_env_patterns: tuple[str, ...] = DENIED_ENV_PATTERNS
    denied_commands: tuple[str, ...] = DENIED_SHELL_COMMANDS
    allow_interactive: bool = False
    max_output_chars: int = 50_000
    _background: dict[str, _BackgroundCommand] = field(default_factory=dict, init=False)

    def runtime_config(self) -> dict[str, Any]:
        return {
            "kind": "shell",
            "cwd": str(self.cwd),
            "default_timeout": self.default_timeout,
            "denied_env_patterns": list(self.denied_env_patterns),
            "denied_commands": list(self.denied_commands),
            "allow_interactive": self.allow_interactive,
            "tool_name": "run_command",
        }

    def host_tools(self) -> tuple[Tool, ...]:
        async def start_command(command: str) -> str:
            reason = _blocked_command_reason(
                command,
                denied_commands=self.denied_commands,
                allow_interactive=self.allow_interactive,
            )
            if reason:
                return f"Error: {reason}"
            command_id = os.urandom(6).hex()
            stdout = tempfile.NamedTemporaryFile(
                prefix=f"aeloon_{command_id}_out_",
                delete=False,
            )
            stderr = tempfile.NamedTemporaryFile(
                prefix=f"aeloon_{command_id}_err_",
                delete=False,
            )
            try:
                process = await asyncio.create_subprocess_shell(
                    command,
                    cwd=self.cwd,
                    env=_safe_environment(self.denied_env_patterns),
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except BaseException:
                Path(stdout.name).unlink(missing_ok=True)
                Path(stderr.name).unlink(missing_ok=True)
                raise
            finally:
                stdout.close()
                stderr.close()
            self._background[command_id] = _BackgroundCommand(
                process=process,
                command=command,
                stdout=Path(stdout.name),
                stderr=Path(stderr.name),
            )
            return f"Started background command: {command!r}\nID: {command_id}"

        async def check_command(command_id: str) -> str:
            background = self._background.get(command_id)
            if background is None:
                return f"[Error: unknown command ID {command_id!r}]"
            return self._render_background(background, stopped=False)

        async def stop_command(command_id: str) -> str:
            background = self._background.pop(command_id, None)
            if background is None:
                return f"[Error: unknown command ID {command_id!r}]"
            await _stop_process(background.process)
            rendered = self._render_background(background, stopped=True)
            background.stdout.unlink(missing_ok=True)
            background.stderr.unlink(missing_ok=True)
            return rendered

        return (
            FunctionTool(
                name="start_command",
                description="Start a long-running command in the background.",
                args_model=_CommandArgs,
                handler=start_command,
                concurrency_mode="exclusive",
            ),
            FunctionTool(
                name="check_command",
                description="Check a background command and read its recent output.",
                args_model=_CommandIdArgs,
                handler=check_command,
                concurrency_mode="read_only",
            ),
            FunctionTool(
                name="stop_command",
                description="Stop a background command and return its final output.",
                args_model=_CommandIdArgs,
                handler=stop_command,
                concurrency_mode="exclusive",
            ),
        )

    async def close(self) -> None:
        for background in tuple(self._background.values()):
            await _stop_process(background.process)
            background.stdout.unlink(missing_ok=True)
            background.stderr.unlink(missing_ok=True)
        self._background.clear()

    def _render_background(
        self,
        background: _BackgroundCommand,
        *,
        stopped: bool,
    ) -> str:
        stdout = background.stdout.read_text(encoding="utf-8", errors="replace")
        stderr = background.stderr.read_text(encoding="utf-8", errors="replace")
        if stopped:
            lines = [f"[stopped: {background.command!r}]"]
        else:
            status = "finished" if background.process.returncode is not None else "running"
            lines = [f"[status: {status}]"]
        if background.process.returncode is not None:
            lines.append(f"[exit code: {background.process.returncode}]")
        output = "\n".join(
            part
            for part in (
                f"[stdout]\n{stdout}" if stdout else "",
                f"[stderr]\n{stderr}" if stderr else "",
            )
            if part
        )
        if len(output) > self.max_output_chars:
            output = (
                f"[... output truncated, showing last {self.max_output_chars} chars]\n"
                + output[-self.max_output_chars :]
            )
        lines.append(
            (output or "(no output yet)") if not stopped else (output or "(no output)")
        )
        return "\n".join(lines)


@dataclass(slots=True)
class _BackgroundCommand:
    process: asyncio.subprocess.Process
    command: str
    stdout: Path
    stderr: Path


@dataclass(frozen=True, slots=True)
class RepoContext(RuntimeCapability):
    """Bounded Python observations used for repository navigation."""

    workspace_dir: Path
    denied_paths: tuple[Path, ...] = ()

    def host_tools(self) -> tuple[Tool, ...]:
        async def inventory_agent_context() -> str:
            roots: list[dict[str, Any]] = []
            notes = {
                ".codex": (
                    "Codex uses TOML config; assets are derived from the "
                    ".claude/.agents setup."
                ),
                ".grok": "Grok setup is derived from the .claude/.agents setup.",
            }
            for name in (".claude", ".agents", ".codex", ".grok"):
                directory = self.workspace_dir / name
                entry: dict[str, Any] = {
                    "root": name,
                    "exists": directory.is_dir(),
                    "skills": [],
                    "agents": [],
                    "settings": None,
                    "notes": notes.get(name),
                }
                if directory.is_dir():
                    entry["skills"] = sorted(
                        path.relative_to(self.workspace_dir).as_posix()
                        for path in directory.glob("skills/**/SKILL.md")
                        if path.is_file()
                    )
                    entry["agents"] = sorted(
                        path.relative_to(self.workspace_dir).as_posix()
                        for path in directory.glob("agents/*.md")
                        if path.is_file()
                    )
                    settings = directory / "settings.json"
                    if settings.is_file():
                        entry["settings"] = settings.relative_to(
                            self.workspace_dir
                        ).as_posix()
                roots.append(entry)
            return json.dumps({"roots": roots}, ensure_ascii=False)

        return (
            FunctionTool(
                name="inventory_agent_context",
                description=(
                    "Report where this repository's coding-assistant assets live."
                ),
                args_model=_NoArgs,
                handler=inventory_agent_context,
                concurrency_mode="read_only",
            ),
        )


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ListDirectoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = "."


class _FindFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(min_length=1)
    path: str = "."


class _SearchFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(min_length=1)
    path: str = "."
    include_glob: str | None = None


class _PathArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)


class _CommandArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)


class _CommandIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_id: str = Field(min_length=1)


def _matches_protected_path(path: str, pattern: str) -> bool:
    if pattern.endswith("/*") and path.startswith(pattern[:-1]):
        return True
    candidate = path if "/" in pattern else Path(path).name
    return fnmatch.fnmatch(candidate, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:])
    )


def _safe_environment(patterns: tuple[str, ...]) -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
    }


def _blocked_command_reason(
    command: str,
    *,
    denied_commands: tuple[str, ...],
    allow_interactive: bool,
) -> str | None:
    executable = command.strip().split(maxsplit=1)[0] if command.strip() else ""
    if executable in denied_commands:
        return f"Command {executable!r} is denied."
    if not allow_interactive and executable in {
        "vi",
        "vim",
        "nano",
        "emacs",
        "less",
        "more",
        "top",
        "htop",
        "man",
        "sudo",
        "passwd",
        "ssh",
        "telnet",
        "ftp",
    }:
        return f"Interactive command {executable!r} is not allowed."
    return None


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        await process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
        return
    except TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    await process.wait()


class _PlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=2_000)
    status: Literal["pending", "in_progress", "completed", "cancelled"] = "pending"


class _UpdatePlanArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[_PlanItem] = Field(min_length=1, max_length=64)


@dataclass(slots=True)
class Planning(RuntimeCapability):
    """Turn-local plan state exposed through the existing write_plan tool."""

    _plan: list[dict[str, str]] = field(default_factory=list)

    def host_tools(self) -> tuple[Tool, ...]:
        async def write_plan(items: list[dict[str, str]]) -> str:
            self._plan = [dict(item) for item in items]
            icons = {
                "pending": "[ ]",
                "in_progress": "[~]",
                "completed": "[x]",
                "cancelled": "[-]",
            }
            rendered = [
                f"{index}. {icons[item['status']]} {item['content']}"
                for index, item in enumerate(self._plan, start=1)
            ]
            completed = sum(item["status"] == "completed" for item in self._plan)
            rendered.append(f"({completed}/{len(self._plan)} completed)")
            in_progress = sum(item["status"] == "in_progress" for item in self._plan)
            note = (
                "\n\nNote: keep only one step in_progress at a time."
                if in_progress > 1
                else ""
            )
            return (
                f"Plan updated: {len(self._plan)} step(s).\n\n"
                + "\n".join(rendered)
                + note
            )

        return (
            FunctionTool(
                name="write_plan",
                description=(
                    "Create or replace the complete ordered plan for the current turn."
                ),
                args_model=_UpdatePlanArgs,
                handler=write_plan,
                concurrency_mode="exclusive",
            ),
        )


class _WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4_000)
    num_results: int = Field(default=5, ge=1, le=10)


class _GetPageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=8_000)


@dataclass(frozen=True, slots=True)
class ExaSearch(RuntimeCapability):
    """Small Exa search adapter kept behind Aeloon's capability policy."""

    api_key: str

    def host_tools(self) -> tuple[Tool, ...]:
        async def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"https://api.exa.ai/{path}",
                    headers={
                        "x-api-key": self.api_key,
                        "content-type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
            return dict(response.json())

        async def web_search(query: str, num_results: int = 5) -> str:
            return json.dumps(
                await post(
                    "search",
                    {
                        "query": query,
                        "numResults": num_results,
                        "contents": {"text": {"maxCharacters": 8_000}},
                    },
                ),
                ensure_ascii=False,
            )

        async def get_page(url: str) -> str:
            return json.dumps(
                await post(
                    "contents",
                    {
                        "urls": [url],
                        "text": {"maxCharacters": 10_000},
                    },
                ),
                ensure_ascii=False,
            )

        return (
            FunctionTool(
                name="web_search",
                description=(
                    "Search the web with Exa and return direct URLs plus bounded page text."
                ),
                args_model=_WebSearchArgs,
                handler=web_search,
                concurrency_mode="read_only",
            ),
            FunctionTool(
                name="get_page",
                description="Retrieve bounded text for one URL through Exa.",
                args_model=_GetPageArgs,
                handler=get_page,
                concurrency_mode="read_only",
            ),
        )


WebCapabilityFactory = Callable[[], RuntimeCapability]
MASTER_CAPABILITY_NAMES = ("filesystem", "shell", "repo_context", "planning")


def history_capability(config: Config) -> SlidingWindow | None:
    """Translate Aeloon's context policy into a Pi-compatible sliding window."""

    compaction = config.agents.defaults.context_compaction
    if not compaction.enabled:
        return None
    trigger_tokens = max(
        1,
        int(config.agents.defaults.context_window_tokens * compaction.trigger_ratio),
    )
    keep_tokens = compaction.preserve_recent_tokens or max(8_000, trigger_tokens // 2)
    keep_tokens = min(keep_tokens, max(0, trigger_tokens - 1))
    return SlidingWindow(
        max_tokens=trigger_tokens,
        keep_tokens=keep_tokens,
        preserve_first_user_message=True,
    )


def harness_capabilities(
    *,
    config: Config,
    names: Iterable[str],
    web_capability_factory: WebCapabilityFactory | None = None,
) -> list[RuntimeCapability]:
    """Build exactly the trusted capabilities declared for one agent."""

    requested = frozenset(names)
    capabilities: list[RuntimeCapability] = []
    if "filesystem" in requested:
        capabilities.append(FileSystem(root_dir=config.workspace))
    if "shell" in requested:
        capabilities.append(
            Shell(
                cwd=config.workspace,
                default_timeout=float(config.tools.exec.timeout),
            )
        )
    if "repo_context" in requested:
        capabilities.append(
            RepoContext(
                workspace_dir=config.workspace,
                denied_paths=(config.data_dir,),
            )
        )
    if "planning" in requested:
        capabilities.append(Planning())
    if "web_search" in requested:
        capabilities.append(
            web_capability_factory()
            if web_capability_factory is not None
            else _default_exa_capability()
        )
    compaction = history_capability(config)
    if compaction is not None:
        capabilities.append(compaction)
    return capabilities


def master_capabilities(config: Config) -> list[RuntimeCapability]:
    """Build the mode-specific Master capability surface."""

    return harness_capabilities(
        config=config,
        names=master_capability_names(config),
    )


def master_capability_names(config: Config) -> tuple[str, ...]:
    """Expose the full normal surface or the configured expert-mode subset."""

    if config.mode == "normal":
        return MASTER_CAPABILITY_NAMES
    return tuple(config.tools.master_capabilities)


def _default_exa_capability() -> RuntimeCapability:
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise CapabilityUnavailable(
            "research expert requires EXA_API_KEY for the default Exa web backend"
        )
    return ExaSearch(api_key)


__all__ = [
    "CapabilityUnavailable",
    "DENIED_ENV_PATTERNS",
    "DENIED_SHELL_COMMANDS",
    "ExaSearch",
    "FileSystem",
    "HARNESS_PROTECTED_PATTERNS",
    "MASTER_CAPABILITY_NAMES",
    "Planning",
    "RepoContext",
    "RuntimeCapability",
    "Shell",
    "SlidingWindow",
    "WebCapabilityFactory",
    "harness_capabilities",
    "history_capability",
    "master_capabilities",
    "master_capability_names",
]
