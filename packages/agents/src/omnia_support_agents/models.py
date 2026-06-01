"""Pydantic models and enums for the agent architecture."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field, JsonValue

from omnia_support_store import ScoredChunk  # noqa: TC001 — needed at runtime by dataclasses


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


class TriageResult(BaseModel):
    """Result of the triage agent's classification and decomposition of a query."""

    urgency: Urgency
    sub_queries: list[SubQuery] = Field(min_length=1)


class ToolCallResultOk(BaseModel):
    """Successful result from invoking an MCP tool."""

    data: JsonValue


class ToolCallResultErr(BaseModel):
    """Failed result from invoking an MCP tool."""

    error: str


ToolCallResult = ToolCallResultOk | ToolCallResultErr


class ToolCall(BaseModel):
    """An MCP tool invocation and its result."""

    tool_name: str
    arguments: dict[str, JsonValue]
    result: ToolCallResult

    def format(self) -> str:
        """Format the tool call and its result as a human-readable string."""
        args = ", ".join(f"{k}={json.dumps(v)}" for k, v in self.arguments.items())
        header = f"{self.tool_name}({args})"
        if isinstance(self.result, ToolCallResultOk):
            return f"{header}\nResult: {json.dumps(self.result.data)}"
        return f"{header}\nError: {self.result.error}"


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
    tool_calls: list[ToolCall] = field(default_factory=list)

    def format_for_aggregator(self) -> str:
        """Format the result for the response aggregator."""
        tools = "\n\n".join(f"{i}. {tc.format()}" for i, tc in enumerate(self.tool_calls, 1))
        return f"Sub-query: {self.query}\nAnswer: {self.answer}\n\nTool calls:\n{tools}"


@dataclass(frozen=True)
class AgentResult:
    """Final result from the agent orchestration pipeline."""

    answer: str
    urgency: Urgency
    sub_query_results: list[InformationResult | ActionResult] = field(default_factory=list)
