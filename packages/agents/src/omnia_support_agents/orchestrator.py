"""Orchestrator: pure control-flow fan-out/fan-in coordination of agents."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import mlflow
from agents import AgentsException, Runner
from openai import OpenAIError

from .action import UNAVAILABLE_MESSAGE, extract_tool_calls
from .models import ActionResult, AgentResult, AgentType, InformationResult, SubQuery

if TYPE_CHECKING:
    from agents import Agent
    from openai.types.chat import ChatCompletionMessageParam

    from .aggregator import AggregatorAgent
    from .information import InformationAgent
    from .triage import TriageAgent


class AgentOrchestrator:
    """Coordinates triage, information, action, and aggregator agents."""

    def __init__(
        self,
        *,
        triage_agent: TriageAgent,
        information_agent: InformationAgent,
        aggregator_agent: AggregatorAgent,
        action_agent: Agent,
    ) -> None:
        """Create an orchestrator."""
        self._triage_agent = triage_agent
        self._information_agent = information_agent
        self._aggregator_agent = aggregator_agent
        self._action_agent = action_agent

    @mlflow.trace(name="orchestrator")
    def handle(
        self,
        question: str,
        history: list[ChatCompletionMessageParam],
    ) -> AgentResult:
        """Run the full orchestration pipeline for a user question."""
        triage_result = self._triage_agent.triage(question, history)

        with ThreadPoolExecutor() as pool:
            sub_query_results = list(
                pool.map(lambda sq: self._dispatch(sq, history), triage_result.sub_queries),
            )

        answer = (
            self._aggregator_agent.aggregate(sub_query_results)
            if len(sub_query_results) > 1
            else sub_query_results[0].answer
        )

        return AgentResult(
            answer=answer,
            urgency=triage_result.urgency,
            sub_query_results=sub_query_results,
        )

    def _dispatch(
        self,
        sub_query: SubQuery,
        history: list[ChatCompletionMessageParam],
    ) -> InformationResult | ActionResult:
        if sub_query.agent == AgentType.INFORMATION:
            return self._information_agent.handle(sub_query.query, history)
        return self._handle_action(sub_query.query)

    def _handle_action(self, query: str) -> ActionResult:
        try:
            run_result = Runner.run_sync(self._action_agent, query)
            return ActionResult(
                query=query,
                answer=run_result.final_output,
                tool_calls=extract_tool_calls(run_result),
            )
        except AgentsException, OpenAIError, ConnectionError, TimeoutError:
            return ActionResult(query=query, answer=UNAVAILABLE_MESSAGE)
