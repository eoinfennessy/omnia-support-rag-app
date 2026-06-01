"""Orchestrator: pure control-flow fan-out/fan-in coordination of agents."""

from __future__ import annotations

import asyncio
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
    async def handle(
        self,
        question: str,
        history: list[ChatCompletionMessageParam],
    ) -> AgentResult:
        """Run the full orchestration pipeline for a user question."""
        triage_result = await asyncio.to_thread(
            self._triage_agent.triage,
            question,
            history,
        )

        tasks = (self._dispatch(sq, history) for sq in triage_result.sub_queries)
        sub_query_results = await asyncio.gather(*tasks)

        answer = (
            await asyncio.to_thread(self._aggregator_agent.aggregate, sub_query_results)
            if len(sub_query_results) > 1
            else sub_query_results[0].answer
        )

        return AgentResult(
            answer=answer,
            urgency=triage_result.urgency,
            sub_query_results=sub_query_results,
        )

    async def _dispatch(
        self,
        sub_query: SubQuery,
        history: list[ChatCompletionMessageParam],
    ) -> InformationResult | ActionResult:
        match sub_query.agent:
            case AgentType.INFORMATION:
                return await asyncio.to_thread(
                    self._information_agent.handle,
                    sub_query.query,
                    history,
                )
            case AgentType.ACTION:
                return await self._handle_action(sub_query.query)

    async def _handle_action(self, query: str) -> ActionResult:
        try:
            run_result = await Runner.run(self._action_agent, query)
            return ActionResult(
                query=query,
                answer=run_result.final_output,
                tool_calls=extract_tool_calls(run_result),
            )
        except AgentsException, OpenAIError, ConnectionError, TimeoutError:
            return ActionResult(query=query, answer=UNAVAILABLE_MESSAGE)
