"""Runtime-owned append-only JSONL session tree."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aeloon_core.blocking import run_blocking
from aeloon_core.core.context_stats import (
    MESSAGE_TYPES,
    cache_statistics,
    context_statistics,
    context_statistics_from_aggregates,
    estimate_tokens,
    message_type_for_statistics,
)
from aeloon_core.core.types import (
    AgentMessage,
    AssistantMessage,
    RunError,
    Usage,
    UserMessage,
    message_from_dict,
    message_to_dict,
)


class SessionError(RunError):
    """Stable runtime failure raised for invalid or missing session state."""

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following "
    "summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"
BRANCH_SUMMARY_PREFIX = (
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
)
BRANCH_SUMMARY_SUFFIX = "</summary>"


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp_ms(value: Any) -> int | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    id: str
    created_at: str
    cwd: str
    path: Path
    parent_session_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SessionContext:
    messages: tuple[AgentMessage, ...]
    thinking_level: str | None = None
    model: tuple[str, str] | None = None
    active_tool_names: tuple[str, ...] | None = None
    compaction_boundary_ms: int | None = None
    compaction_boundary_index: int | None = None


@dataclass(slots=True)
class _LifetimeStats:
    message_count: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    input_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    request_count: int = 0
    hit_request_count: int = 0

    def add(self, entry: dict[str, Any]) -> None:
        usage: dict[str, Any] | None = None
        if entry.get("type") == "message":
            self.message_count += 1
            message = entry.get("message") or {}
            if message.get("role") == "assistant":
                usage = message.get("usage")
        elif entry.get("type") in {"compaction", "branch_summary"}:
            usage = entry.get("usage")
        if not usage:
            return
        self.total_tokens += int(usage.get("totalTokens") or 0)
        self.cost += float((usage.get("cost") or {}).get("total") or 0)
        uncached = max(0, int(usage.get("input") or 0))
        cache_read = max(0, int(usage.get("cacheRead") or 0))
        cache_write = max(0, int(usage.get("cacheWrite") or 0))
        self.input_tokens += uncached
        self.cache_read += cache_read
        self.cache_write += cache_write
        if uncached or cache_read or cache_write:
            self.request_count += 1
            if cache_read:
                self.hit_request_count += 1

    def cache(self) -> dict[str, Any]:
        cacheable = self.input_tokens + self.cache_read
        return {
            "inputTokens": self.input_tokens,
            "readTokens": self.cache_read,
            "writeTokens": self.cache_write,
            "cacheableTokens": cacheable,
            "hitTokenPercent": _percent(self.cache_read, cacheable),
            "requestCount": self.request_count,
            "hitRequestCount": self.hit_request_count,
            "hitRequestPercent": _percent(self.hit_request_count, self.request_count),
        }


@dataclass(slots=True)
class _ContextStatsCache:
    version: int
    leaf_id: str | None
    message_counts: dict[str, int]
    estimated_tokens: dict[str, int]
    estimated_total: int
    usage_anchor: int | None
    trailing_tokens: int

    @property
    def used_tokens(self) -> int:
        if self.usage_anchor is None:
            return self.estimated_total
        return self.usage_anchor + self.trailing_tokens

    def append(self, message: AgentMessage, *, version: int, leaf_id: str) -> None:
        estimated = estimate_tokens(message)
        message_type = message_type_for_statistics(message)
        self.message_counts[message_type] += 1
        self.estimated_tokens[message_type] += estimated
        self.estimated_total += estimated
        if (
            isinstance(message, AssistantMessage)
            and message.stop_reason not in {"error", "aborted"}
            and message.usage.total_tokens > 0
        ):
            self.usage_anchor = message.usage.total_tokens
            self.trailing_tokens = 0
        elif self.usage_anchor is not None:
            self.trailing_tokens += estimated
        self.version = version
        self.leaf_id = leaf_id


class Session:
    """Stateful session facade backed by one append-only JSONL file."""

    def __init__(
        self,
        metadata: SessionMetadata,
        entries: list[dict[str, Any]],
        *,
        current_leaf_id: str | None,
    ) -> None:
        self.metadata = metadata
        self._entries = entries
        self._by_id = {str(entry["id"]): entry for entry in entries}
        self._current_leaf_id = current_leaf_id
        self._lock = asyncio.Lock()
        self._stats_cache_version = 0
        self._lifetime_stats = _lifetime_stats(entries)
        self._context_stats_cache: _ContextStatsCache | None = None

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def path(self) -> Path:
        return self.metadata.path

    async def get_leaf_id(self) -> str | None:
        return self._current_leaf_id

    async def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        entry = self._by_id.get(entry_id)
        return dict(entry) if entry is not None else None

    async def get_entries(
        self, *, after_entry_seq: int = 0, limit: int | None = None
    ) -> list[dict[str, Any]]:
        end = None if limit is None else after_entry_seq + limit
        return [dict(entry) for entry in self._entries[after_entry_seq:end]]

    async def find_entries(self, entry_type: str) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._entries if entry.get("type") == entry_type]

    async def append_message(self, message: AgentMessage) -> str:
        return await self._append_entry("message", message=message_to_dict(message))

    async def append_model_change(self, provider: str, model_id: str) -> str:
        return await self._append_entry("model_change", provider=provider, modelId=model_id)

    async def append_thinking_level_change(self, level: str) -> str:
        return await self._append_entry("thinking_level_change", thinkingLevel=level)

    async def append_active_tools_change(self, names: list[str] | tuple[str, ...]) -> str:
        return await self._append_entry("active_tools_change", activeToolNames=list(names))

    async def append_compaction(
        self,
        *,
        summary: str,
        tokens_before: int,
        first_kept_entry_id: str | None = None,
        retained_tail: tuple[AgentMessage, ...] = (),
        usage: Usage | None = None,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "summary": summary,
            "tokensBefore": tokens_before,
            "fromHook": from_hook,
        }
        if first_kept_entry_id:
            payload["firstKeptEntryId"] = first_kept_entry_id
        if retained_tail:
            payload["retainedTail"] = [message_to_dict(message) for message in retained_tail]
        if usage is not None:
            payload["usage"] = usage.to_dict()
        if details is not None:
            payload["details"] = details
        return await self._append_entry("compaction", **payload)

    async def append_branch_summary(
        self,
        *,
        from_id: str,
        summary: str,
        usage: Usage | None = None,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "fromId": from_id,
            "summary": summary,
            "fromHook": from_hook,
        }
        if usage is not None:
            payload["usage"] = usage.to_dict()
        if details is not None:
            payload["details"] = details
        return await self._append_entry("branch_summary", **payload)

    async def append_custom_message(
        self,
        *,
        custom_type: str,
        content: str,
        display: bool = True,
        details: Any = None,
    ) -> str:
        return await self._append_entry(
            "custom_message",
            customType=custom_type,
            content=content,
            display=display,
            details=details,
        )

    async def append_run_start(
        self,
        *,
        run_id: str,
        input: dict[str, Any],
        model_id: str,
        thinking_level: str,
    ) -> str:
        """Record a public prompt-run boundary without affecting model context."""

        return await self._append_entry(
            "run_start",
            runId=run_id,
            input=input,
            modelId=model_id,
            thinkingLevel=thinking_level,
        )

    async def append_run_end(
        self,
        *,
        run_id: str,
        status: str,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> str:
        """Record the terminal state for a public prompt run."""

        payload: dict[str, Any] = {"runId": run_id, "status": status}
        if duration_ms is not None:
            payload["durationMs"] = duration_ms
        if error:
            payload["error"] = error
        return await self._append_entry("run_end", **payload)

    async def append_artifact_delivery(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        artifacts: list[dict[str, Any]],
    ) -> str:
        """Persist display-only deliverables without adding model context."""

        return await self._append_entry(
            "artifact_delivery",
            runId=run_id,
            toolCallId=tool_call_id,
            artifacts=artifacts,
        )

    async def append_session_config(self, config: dict[str, Any]) -> str:
        """Persist runtime session overrides without adding model context."""

        return await self._append_entry("session_config", config=config)

    async def append_next_turn_input(self, input: dict[str, Any]) -> str:
        """Persist input that should precede the next prompt."""

        return await self._append_entry("next_turn_input", input=input)

    async def append_next_turn_consumed(self, entry_ids: list[str]) -> str:
        return await self._append_entry("next_turn_consumed", entryIds=entry_ids)

    async def set_label(self, target_id: str, label: str | None) -> str:
        if target_id not in self._by_id:
            raise SessionError("not_found", f"Entry {target_id} not found")
        return await self._append_entry("label", targetId=target_id, label=label)

    async def get_label(self, target_id: str) -> str | None:
        label: str | None = None
        for entry in self._entries:
            if entry.get("type") == "label" and entry.get("targetId") == target_id:
                value = entry.get("label")
                label = str(value).strip() if value else None
        return label

    async def set_name(self, name: str | None) -> str:
        return await self._append_entry("session_info", name=name)

    async def get_name(self) -> str | None:
        entries = [entry for entry in self._entries if entry.get("type") == "session_info"]
        if not entries:
            return None
        name = entries[-1].get("name")
        return str(name).strip() if name else None

    async def set_leaf_id(self, leaf_id: str | None) -> None:
        if leaf_id is not None and leaf_id not in self._by_id:
            raise SessionError("not_found", f"Entry {leaf_id} not found")
        entry = {
            "type": "leaf",
            "id": self._entry_id(),
            "parentId": self._current_leaf_id,
            "timestamp": _iso_now(),
            "targetId": leaf_id,
        }
        await self._persist(entry)
        self._entries.append(entry)
        self._by_id[entry["id"]] = entry
        self._current_leaf_id = leaf_id
        self._stats_cache_version += 1
        self._lifetime_stats.add(entry)
        self._context_stats_cache = None

    async def get_branch(self, from_id: str | None = None) -> list[dict[str, Any]]:
        leaf_id = self._current_leaf_id if from_id is None else from_id
        if leaf_id is None:
            return []
        current = self._by_id.get(leaf_id)
        if current is None:
            raise SessionError("not_found", f"Entry {leaf_id} not found")
        path: list[dict[str, Any]] = []
        stop_at: str | None = None
        while current is not None:
            path.insert(0, current)
            if stop_at is not None and current["id"] == stop_at:
                break
            if current.get("type") == "compaction":
                if current.get("retainedTail"):
                    break
                stop_at = current.get("firstKeptEntryId")
            parent_id = current.get("parentId")
            if not parent_id:
                break
            current = self._by_id.get(str(parent_id))
            if current is None:
                raise SessionError("invalid_session", f"Entry {parent_id} not found")
        return [dict(entry) for entry in path]

    async def build_context(self) -> SessionContext:
        branch = await self.get_branch()
        state_branch = await self._get_full_branch()
        thinking: str | None = None
        model: tuple[str, str] | None = None
        active_tools: tuple[str, ...] | None = None
        compaction_boundary_ms: int | None = None
        compaction_boundary_index: int | None = None
        for entry in state_branch:
            if entry.get("type") == "thinking_level_change":
                thinking = str(entry.get("thinkingLevel") or "off")
            elif entry.get("type") == "model_change":
                model = (str(entry.get("provider")), str(entry.get("modelId")))
            elif entry.get("type") == "active_tools_change":
                active_tools = tuple(str(name) for name in entry.get("activeToolNames") or [])
            elif (
                entry.get("type") == "message"
                and entry.get("message", {}).get("role") == "assistant"
            ):
                raw = entry["message"]
                model = (str(raw.get("provider")), str(raw.get("model")))
        messages: list[AgentMessage] = []
        ordered_branch = branch
        latest_compaction_index: int | None = None
        for index in range(len(branch) - 1, -1, -1):
            entry = branch[index]
            if entry.get("type") == "compaction" and not entry.get("retainedTail"):
                # The compaction is appended after the messages it retains in the tree,
                # but its synthetic summary must precede that retained tail for the model.
                ordered_branch = [entry, *branch[:index], *branch[index + 1 :]]
                latest_compaction_index = index
                break
            if entry.get("type") == "compaction" and latest_compaction_index is None:
                latest_compaction_index = index
        post_compaction_ids = (
            {str(entry.get("id")) for entry in branch[latest_compaction_index + 1 :]}
            if latest_compaction_index is not None
            else set()
        )
        fresh_context_started = False
        for entry in ordered_branch:
            if (
                latest_compaction_index is not None
                and not fresh_context_started
                and str(entry.get("id")) in post_compaction_ids
            ):
                compaction_boundary_index = len(messages) - 1
                fresh_context_started = True
            entry_type = entry.get("type")
            if entry_type == "message":
                messages.append(message_from_dict(entry["message"]))
            elif entry_type == "custom_message":
                content = entry.get("content")
                messages.append(UserMessage(str(content or "")))
            elif entry_type == "compaction":
                compaction_boundary_ms = _timestamp_ms(entry.get("timestamp"))
                summary = str(entry.get("summary") or "")
                messages.append(
                    UserMessage(COMPACTION_SUMMARY_PREFIX + summary + COMPACTION_SUMMARY_SUFFIX)
                )
                for raw in entry.get("retainedTail") or []:
                    messages.append(message_from_dict(raw))
            elif entry_type == "branch_summary" and entry.get("summary"):
                messages.append(
                    UserMessage(
                        BRANCH_SUMMARY_PREFIX + str(entry["summary"]) + BRANCH_SUMMARY_SUFFIX
                    )
                )
        if latest_compaction_index is not None and compaction_boundary_index is None:
            compaction_boundary_index = len(messages) - 1
        return SessionContext(
            tuple(messages),
            thinking,
            model,
            active_tools,
            compaction_boundary_ms,
            compaction_boundary_index,
        )

    async def _get_full_branch(self) -> list[dict[str, Any]]:
        current = self._by_id.get(self._current_leaf_id) if self._current_leaf_id else None
        path: list[dict[str, Any]] = []
        while current is not None:
            path.insert(0, current)
            parent_id = current.get("parentId")
            if not parent_id:
                break
            current = self._by_id.get(str(parent_id))
            if current is None:
                raise SessionError("invalid_session", f"Entry {parent_id} not found")
        return [dict(entry) for entry in path]

    async def stats(self, *, context_window: int | None = None) -> dict[str, Any]:
        """Return lifetime totals plus statistics for the effective context branch."""
        cache = self._context_stats_cache
        if cache is None or cache.version != self._stats_cache_version:
            context = await self.build_context()
            cache = _context_stats_cache(
                context,
                version=self._stats_cache_version,
                leaf_id=self._current_leaf_id,
            )
            self._context_stats_cache = cache
        return {
            "messageCount": self._lifetime_stats.message_count,
            "totalTokens": self._lifetime_stats.total_tokens,
            "costTotal": self._lifetime_stats.cost,
            **context_statistics_from_aggregates(
                cache.used_tokens,
                cache.message_counts,
                cache.estimated_tokens,
                context_window=context_window,
            ),
            "cache": self._lifetime_stats.cache(),
        }

    async def _stats_full(self, *, context_window: int | None = None) -> dict[str, Any]:
        """Correct full-scan oracle used by fallbacks and regression tests."""

        lifetime = _lifetime_stats(self._entries)
        usages = _entry_usages(self._entries)
        context = await self.build_context()
        return {
            "messageCount": lifetime.message_count,
            "totalTokens": lifetime.total_tokens,
            "costTotal": lifetime.cost,
            **context_statistics(
                context.messages,
                context_window=context_window,
                usage_after_ms=context.compaction_boundary_ms,
                usage_after_index=context.compaction_boundary_index,
            ),
            "cache": cache_statistics(usages),
        }

    async def _append_entry(self, entry_type: str, **payload: Any) -> str:
        previous_leaf = self._current_leaf_id
        previous_version = self._stats_cache_version
        entry = {
            "type": entry_type,
            "id": self._entry_id(),
            "parentId": self._current_leaf_id,
            "timestamp": _iso_now(),
            **payload,
        }
        await self._persist(entry)
        self._entries.append(entry)
        self._by_id[entry["id"]] = entry
        self._current_leaf_id = entry["id"]
        self._stats_cache_version += 1
        self._lifetime_stats.add(entry)
        cache = self._context_stats_cache
        message = _context_message(entry)
        if (
            message is not None
            and cache is not None
            and cache.version == previous_version
            and cache.leaf_id == previous_leaf
        ):
            cache.append(message, version=self._stats_cache_version, leaf_id=entry["id"])
        else:
            self._context_stats_cache = None
        return str(entry["id"])

    def _entry_id(self) -> str:
        for _ in range(100):
            entry_id = uuid.uuid4().hex[-8:]
            if entry_id not in self._by_id:
                return entry_id
        return uuid.uuid4().hex

    async def _persist(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._lock:
            await run_blocking(_append_fsync, self.path, line)


class JsonlSessionRepository:
    """Create, open, list, fork, and delete runtime sessions."""

    def __init__(self, data_dir: Path | str) -> None:
        self.directory = Path(data_dir).expanduser().resolve(strict=False) / "harness-sessions"
        self.directory.mkdir(parents=True, exist_ok=True)

    async def create(
        self,
        *,
        cwd: Path | str,
        session_id: str | None = None,
        parent_session_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        resolved_id = session_id or uuid.uuid4().hex
        path = self._path(resolved_id)
        if path.exists():
            raise SessionError("storage", f"Session already exists: {resolved_id}")
        header = {
            "type": "session",
            "version": 3,
            "id": resolved_id,
            "timestamp": _iso_now(),
            "cwd": str(Path(cwd).expanduser().resolve(strict=False)),
        }
        if parent_session_path:
            header["parentSession"] = parent_session_path
        if metadata:
            header["metadata"] = metadata
        await run_blocking(
            _write_fsync,
            path,
            json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n",
        )
        return Session(_metadata(header, path), [], current_leaf_id=None)

    async def open(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.is_file():
            raise SessionError("not_found", f"Session {session_id} not found")
        header, entries, leaf = await run_blocking(_read_session, path)
        if header["id"] != session_id:
            raise SessionError("invalid_session", f"Session header id differs from {session_id}")
        return Session(_metadata(header, path), entries, current_leaf_id=leaf)

    async def list(self, *, cwd: Path | str | None = None) -> list[SessionMetadata]:
        expected_cwd = (
            str(Path(cwd).expanduser().resolve(strict=False)) if cwd is not None else None
        )
        result: list[SessionMetadata] = []
        for path in self.directory.glob("*.jsonl"):
            try:
                header, _, _ = await run_blocking(_read_session, path, True)
            except SessionError:
                continue
            metadata = _metadata(header, path)
            if expected_cwd is None or metadata.cwd == expected_cwd:
                result.append(metadata)
        return sorted(result, key=lambda item: item.created_at, reverse=True)

    async def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if not path.exists():
            raise SessionError("not_found", f"Session {session_id} not found")
        await run_blocking(path.unlink)

    async def fork(
        self,
        session_id: str,
        *,
        entry_id: str | None = None,
        position: str = "at",
        new_session_id: str | None = None,
    ) -> Session:
        source = await self.open(session_id)
        target_id = entry_id or await source.get_leaf_id()
        if target_id is not None and position == "before":
            entry = await source.get_entry(target_id)
            target_id = str(entry.get("parentId")) if entry and entry.get("parentId") else None
        branch = await source.get_branch(target_id)
        forked = await self.create(
            cwd=source.metadata.cwd,
            session_id=new_session_id,
            parent_session_path=str(source.path),
        )
        for entry in branch:
            if entry.get("type") == "message":
                await forked.append_message(message_from_dict(entry["message"]))
        return forked

    def _path(self, session_id: str) -> Path:
        if not session_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in session_id
        ):
            raise SessionError(
                "invalid_session", "session id must contain only letters, digits, '-' or '_'"
            )
        return self.directory / f"{session_id}.jsonl"


def _metadata(header: dict[str, Any], path: Path) -> SessionMetadata:
    return SessionMetadata(
        id=str(header["id"]),
        created_at=str(header["timestamp"]),
        cwd=str(header["cwd"]),
        path=path,
        parent_session_path=(str(header["parentSession"]) if header.get("parentSession") else None),
        metadata=dict(header.get("metadata") or {}),
    )


def _read_session(
    path: Path, header_only: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise SessionError("invalid_session", f"Invalid JSONL session {path}: missing header")
    try:
        header = json.loads(raw_lines[0])
    except json.JSONDecodeError as exc:
        raise SessionError(
            "invalid_session", f"Invalid JSONL session {path}: bad header", cause=exc
        ) from exc
    if (
        not isinstance(header, dict)
        or header.get("type") != "session"
        or header.get("version") != 3
    ):
        raise SessionError("invalid_session", f"Invalid JSONL session {path}: unsupported header")
    for key in ("id", "timestamp", "cwd"):
        if not isinstance(header.get(key), str) or not header[key]:
            raise SessionError("invalid_session", f"Invalid JSONL session {path}: missing {key}")
    if header_only:
        return header, [], None
    entries: list[dict[str, Any]] = []
    leaf: str | None = None
    for index, line in enumerate(raw_lines[1:], 2):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(raw_lines):
                break
            raise SessionError(
                "invalid_entry", f"Invalid JSONL session {path}: line {index}", cause=exc
            ) from exc
        if not isinstance(entry, dict) or not all(
            key in entry for key in ("type", "id", "parentId", "timestamp")
        ):
            raise SessionError("invalid_entry", f"Invalid JSONL session {path}: line {index}")
        entries.append(entry)
        leaf = entry.get("targetId") if entry.get("type") == "leaf" else str(entry["id"])
    return header, entries, leaf


def _append_fsync(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _lifetime_stats(entries: list[dict[str, Any]]) -> _LifetimeStats:
    result = _LifetimeStats()
    for entry in entries:
        result.add(entry)
    return result


def _entry_usages(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in entries:
        usage: dict[str, Any] | None = None
        if entry.get("type") == "message":
            message = entry.get("message") or {}
            if message.get("role") == "assistant":
                usage = message.get("usage")
        elif entry.get("type") in {"compaction", "branch_summary"}:
            usage = entry.get("usage")
        if usage:
            result.append(usage)
    return result


def _context_message(entry: dict[str, Any]) -> AgentMessage | None:
    if entry.get("type") == "message":
        return message_from_dict(entry["message"])
    if entry.get("type") == "custom_message":
        return UserMessage(str(entry.get("content") or ""))
    return None


def _context_stats_cache(
    context: SessionContext,
    *,
    version: int,
    leaf_id: str | None,
) -> _ContextStatsCache:
    counts = {message_type: 0 for message_type in MESSAGE_TYPES}
    estimates = {message_type: 0 for message_type in MESSAGE_TYPES}
    estimated_total = 0
    usage_anchor: int | None = None
    trailing_tokens = 0
    for index, message in enumerate(context.messages):
        estimated = estimate_tokens(message)
        message_type = message_type_for_statistics(message)
        counts[message_type] += 1
        estimates[message_type] += estimated
        estimated_total += estimated
        after_boundary = (
            index > context.compaction_boundary_index
            if context.compaction_boundary_index is not None
            else context.compaction_boundary_ms is None
            or message.timestamp > context.compaction_boundary_ms
        )
        if (
            isinstance(message, AssistantMessage)
            and message.stop_reason not in {"error", "aborted"}
            and message.usage.total_tokens > 0
            and after_boundary
        ):
            usage_anchor = message.usage.total_tokens
            trailing_tokens = 0
        elif usage_anchor is not None:
            trailing_tokens += estimated
    return _ContextStatsCache(
        version=version,
        leaf_id=leaf_id,
        message_counts=counts,
        estimated_tokens=estimates,
        estimated_total=estimated_total,
        usage_anchor=usage_anchor,
        trailing_tokens=trailing_tokens,
    )


def _percent(value: int, total: int) -> float:
    return round(value * 100 / total, 2) if total > 0 else 0.0


def _write_fsync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "BRANCH_SUMMARY_PREFIX",
    "BRANCH_SUMMARY_SUFFIX",
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "JsonlSessionRepository",
    "Session",
    "SessionContext",
    "SessionError",
    "SessionMetadata",
]
