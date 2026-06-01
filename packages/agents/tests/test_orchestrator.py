"""Tests for the orchestrator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnia_support_agents.action import UNAVAILABLE_MESSAGE
from omnia_support_agents.models import (
    ActionResult,
    AgentType,
    InformationResult,
    SubQuery,
    TriageResult,
    Urgency,
)
from omnia_support_agents.orchestrator import AgentOrchestrator


def _make_triage_decision(
    *,
    urgency: Urgency = Urgency.LOW,
    sub_queries: list[SubQuery] | None = None,
) -> TriageResult:
    return TriageResult(
        urgency=urgency,
        sub_queries=sub_queries or [SubQuery(query="test query", agent=AgentType.INFORMATION)],
    )


def _make_information_result(
    query: str = "test query",
    answer: str = "info answer",
) -> InformationResult:
    return InformationResult(query=query, answer=answer)


def _make_run_result(final_output: str = "action answer") -> MagicMock:
    result = MagicMock()
    result.final_output = final_output
    result.new_items = []
    return result


def _make_orchestrator(
    *,
    triage: MagicMock | None = None,
    information: MagicMock | None = None,
    aggregator: MagicMock | None = None,
    action_agent: MagicMock | None = None,
) -> AgentOrchestrator:
    return AgentOrchestrator(
        triage_agent=triage or MagicMock(),
        information_agent=information or MagicMock(),
        aggregator_agent=aggregator or MagicMock(),
        action_agent=action_agent or MagicMock(),
    )


class TestOrchestrator:
    def test_single_information_query(self):
        triage = MagicMock()
        triage.triage.return_value = _make_triage_decision(
            sub_queries=[SubQuery(query="return policy", agent=AgentType.INFORMATION)],
        )
        information = MagicMock()
        information.handle.return_value = _make_information_result(
            query="return policy",
            answer="You can return items within 30 days.",
        )
        aggregator = MagicMock()

        result = asyncio.run(
            _make_orchestrator(
                triage=triage,
                information=information,
                aggregator=aggregator,
            ).handle("What is your return policy?", []),
        )

        assert result.answer == "You can return items within 30 days."
        assert result.urgency == Urgency.LOW
        assert len(result.sub_query_results) == 1
        assert isinstance(result.sub_query_results[0], InformationResult)
        aggregator.aggregate.assert_not_called()

    @patch("omnia_support_agents.orchestrator.Runner")
    def test_single_action_query(self, mock_runner):
        triage = MagicMock()
        triage.triage.return_value = _make_triage_decision(
            urgency=Urgency.MEDIUM,
            sub_queries=[SubQuery(query="order status for ORD-001", agent=AgentType.ACTION)],
        )
        mock_runner.run = AsyncMock(return_value=_make_run_result("Your order is in transit."))
        aggregator = MagicMock()

        result = asyncio.run(
            _make_orchestrator(triage=triage, aggregator=aggregator).handle(
                "Where is my order ORD-001?",
                [],
            ),
        )

        assert result.answer == "Your order is in transit."
        assert result.urgency == Urgency.MEDIUM
        assert len(result.sub_query_results) == 1
        assert isinstance(result.sub_query_results[0], ActionResult)
        aggregator.aggregate.assert_not_called()

    @patch("omnia_support_agents.orchestrator.Runner")
    def test_multiple_sub_queries_calls_aggregator(self, mock_runner):
        triage = MagicMock()
        triage.triage.return_value = _make_triage_decision(
            urgency=Urgency.HIGH,
            sub_queries=[
                SubQuery(query="damaged item", agent=AgentType.ACTION),
                SubQuery(query="return policy", agent=AgentType.INFORMATION),
            ],
        )
        mock_runner.run = AsyncMock(return_value=_make_run_result("Refund initiated."))
        information = MagicMock()
        information.handle.return_value = _make_information_result(
            query="return policy",
            answer="30-day returns.",
        )
        aggregator = MagicMock()
        aggregator.aggregate.return_value = "Refund initiated and you have 30 days to return."

        result = asyncio.run(
            _make_orchestrator(
                triage=triage,
                information=information,
                aggregator=aggregator,
            ).handle("Item arrived damaged, what's the return policy?", []),
        )

        assert result.answer == "Refund initiated and you have 30 days to return."
        assert result.urgency == Urgency.HIGH
        assert len(result.sub_query_results) == 2
        aggregator.aggregate.assert_called_once()

    @patch("omnia_support_agents.orchestrator.Runner")
    def test_action_failure_returns_unavailable_message(self, mock_runner):
        triage = MagicMock()
        triage.triage.return_value = _make_triage_decision(
            sub_queries=[SubQuery(query="check order", agent=AgentType.ACTION)],
        )
        mock_runner.run = AsyncMock(side_effect=ConnectionError("MCP server unreachable"))

        result = asyncio.run(
            _make_orchestrator(triage=triage).handle("Check my order", []),
        )

        assert result.answer == UNAVAILABLE_MESSAGE
        assert isinstance(result.sub_query_results[0], ActionResult)
        assert result.sub_query_results[0].tool_calls == []

    def test_urgency_preserved_from_triage(self):
        triage = MagicMock()
        triage.triage.return_value = _make_triage_decision(urgency=Urgency.CRITICAL)
        information = MagicMock()
        information.handle.return_value = _make_information_result()

        result = asyncio.run(
            _make_orchestrator(triage=triage, information=information).handle("urgent!", []),
        )

        assert result.urgency == Urgency.CRITICAL

    def test_information_agent_receives_history(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        triage = MagicMock()
        triage.triage.return_value = _make_triage_decision(
            sub_queries=[SubQuery(query="follow-up", agent=AgentType.INFORMATION)],
        )
        information = MagicMock()
        information.handle.return_value = _make_information_result()

        asyncio.run(
            _make_orchestrator(triage=triage, information=information).handle("follow-up", history),  # ty: ignore[invalid-argument-type]
        )

        information.handle.assert_called_once_with("follow-up", history)

    def test_triage_agent_receives_history(self):
        history = [{"role": "user", "content": "Hi"}]
        triage = MagicMock()
        triage.triage.return_value = _make_triage_decision()
        information = MagicMock()
        information.handle.return_value = _make_information_result()

        asyncio.run(
            _make_orchestrator(triage=triage, information=information).handle("question", history),  # ty: ignore[invalid-argument-type]
        )

        triage.triage.assert_called_once_with("question", history)

    @patch("omnia_support_agents.orchestrator.Runner")
    def test_unexpected_error_propagates(self, mock_runner):
        triage = MagicMock()
        triage.triage.return_value = _make_triage_decision(
            sub_queries=[SubQuery(query="check order", agent=AgentType.ACTION)],
        )
        mock_runner.run = AsyncMock(side_effect=TypeError("unexpected bug"))

        with pytest.raises(TypeError, match="unexpected bug"):
            asyncio.run(
                _make_orchestrator(triage=triage).handle("Check my order", []),
            )
