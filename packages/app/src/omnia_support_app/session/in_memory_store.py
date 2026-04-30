"""In-memory session store implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .store import SessionStore

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


@dataclass
class InMemorySessionStore(SessionStore):
    """Stores conversation history per session in memory."""

    _sessions: dict[str, list[ChatCompletionMessageParam]] = field(
        default_factory=dict,
    )

    def get_history(self, session_id: str) -> list[ChatCompletionMessageParam]:
        """Return the conversation history for a session."""
        return list(self._sessions.get(session_id, []))

    def add_messages(
        self,
        session_id: str,
        messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Append messages to a session's history."""
        self._sessions.setdefault(session_id, []).extend(messages)

    def clear(self, session_id: str) -> None:
        """Clear a session's history."""
        self._sessions.pop(session_id, None)
