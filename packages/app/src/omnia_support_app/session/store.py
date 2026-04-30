"""Abstract session store for chat history."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class SessionStore(ABC):
    """Abstract session store for conversation history."""

    @abstractmethod
    def get_history(self, session_id: str) -> list[ChatCompletionMessageParam]:
        """Return the conversation history for a session."""

    @abstractmethod
    def add_messages(
        self,
        session_id: str,
        messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Append messages to a session's history."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Clear a session's history."""
