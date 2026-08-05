"""Runtime conversation state and session persistence."""

from __future__ import annotations

from aeloon_core.core.events import RunEventDispatcher
from aeloon_core.core.types import AgentMessage, message_to_dict
from aeloon_core.runtime.session import Session, SessionContext


class ConversationTranscript:
    """Keep in-memory context synchronized with an optional session."""

    def __init__(
        self,
        session: Session | None,
        events: RunEventDispatcher,
    ) -> None:
        self._session = session
        self._events = events
        self._messages: list[AgentMessage] = []

    @property
    def session(self) -> Session | None:
        return self._session

    @session.setter
    def session(self, session: Session | None) -> None:
        self._session = session

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return tuple(self._messages)

    async def restore(self) -> SessionContext | None:
        if self._session is None:
            return None
        context = await self._session.build_context()
        self._messages = list(context.messages)
        return context

    async def append(
        self,
        message: AgentMessage,
        *,
        emit_events: bool = True,
        message_started: bool = False,
    ) -> None:
        payload = {"message": message_to_dict(message)}
        if emit_events and not message_started:
            await self._events.emit("message_start", payload)
        self._messages.append(message)
        if self._session is not None:
            await self._session.append_message(message)
        if emit_events:
            await self._events.emit("message_end", payload)
        if self._session is not None:
            await self._events.emit("save_point", {"hadPendingMutations": True})


__all__ = ["ConversationTranscript"]
