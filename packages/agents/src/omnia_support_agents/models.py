"""Pydantic models and enums for the agent architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from omnia_support_store import ScoredChunk


class AgentType(StrEnum):
    """Type of agent that should handle a sub-query."""

    INFORMATION = "information"
    ACTION = "action"


class Urgency(StrEnum):
    """Urgency level of a customer query."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SubQuery(BaseModel):
    """A single sub-query from query decomposition."""

    query: str
    agent: AgentType


class TriageDecision(BaseModel):
    """Result of the triage agent's classification and decomposition of a query."""

    urgency: Urgency
    sub_queries: list[SubQuery]


class ToolCallResult(BaseModel):
    """Result from invoking an MCP tool."""

    tool_name: str
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class SubQueryResult(Protocol):
    """Protocol for sub-query results from any agent."""

    query: str
    answer: str

    def format_for_aggregator(self) -> str:
        """Format the result for the response aggregator."""
        ...


@dataclass(frozen=True)
class InformationResult:
    """Result from the information agent."""

    query: str
    answer: str
    context: list[ScoredChunk] = field(default_factory=list)

    def format_for_aggregator(self) -> str:
        """Format the result for the response aggregator."""
        chunks = "\n\n".join(c.page_content for c in self.context)
        return f"Sub-query: {self.query}\nAnswer: {self.answer}\n\nContext:\n{chunks}"


@dataclass(frozen=True)
class ActionResult:
    """Result from the action agent."""

    query: str
    answer: str
    tool_calls: list[ToolCallResult] = field(default_factory=list)

    def format_for_aggregator(self) -> str:
        """Format the result for the response aggregator."""
        tools = "\n".join(f"- {tc.tool_name}: {tc.data}" for tc in self.tool_calls)
        return f"Sub-query: {self.query}\nAnswer: {self.answer}\n\nTool results:\n{tools}"


@dataclass(frozen=True)
class AgentResult:
    """Final result from the agent orchestration pipeline."""

    answer: str
    urgency: Urgency
    sub_query_results: list[InformationResult | ActionResult] = field(default_factory=list)
