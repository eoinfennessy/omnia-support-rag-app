"""Action agent evaluation: config, predict function, and MLflow run."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlflow
from agents import Agent, Runner
from mlflow.genai.scorers import Correctness, RelevanceToQuery

from ..common import content_hash, resolve_prompt_uri
from .gold_set import ActionGoldSetEntry, sync_gold_set
from .scorers import tool_usage_recall

if TYPE_CHECKING:
    from collections.abc import Callable

    from agents.mcp import MCPServer
    from mlflow.entities.model_registry.prompt_version import PromptVersion


@dataclass(frozen=True)
class RunEvaluationConfig:
    """Configuration for an action agent evaluation run."""

    gold_set: list[ActionGoldSetEntry]
    prompt_name: str
    prompt_version: int | None
    prompt_alias: str
    mcp_server: MCPServer
    llm_model: str
    experiment_name: str


def make_predict_fn(
    action_agent: Agent,
    loop: asyncio.AbstractEventLoop,
) -> Callable[[str], str]:
    """Create a predict function for mlflow.genai.evaluate()."""

    def predict(question: str) -> str:
        future = asyncio.run_coroutine_threadsafe(Runner.run(action_agent, question), loop)
        result = future.result()
        return result.final_output

    return predict


def run_evaluation(config: RunEvaluationConfig) -> None:
    """Run the action agent evaluation pipeline."""
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    mlflow.set_experiment(config.experiment_name)

    prompt_uri = resolve_prompt_uri(config.prompt_name, config.prompt_version, config.prompt_alias)
    prompt = mlflow.genai.load_prompt(prompt_uri)
    print(f"Loaded prompt '{config.prompt_name}' version {prompt.version}")

    # Run event loop in a background thread so that MCP connect, Runner.run,
    # and MCP cleanup all share the same loop.  The predict function uses
    # run_coroutine_threadsafe() which is safe to call from MLflow's worker
    # threads.
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    future = asyncio.run_coroutine_threadsafe(config.mcp_server.connect(), loop)
    future.result()
    try:
        _run_eval(config, prompt, loop)
    finally:
        future = asyncio.run_coroutine_threadsafe(config.mcp_server.cleanup(), loop)
        future.result()
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join()
        loop.close()


def _run_eval(
    config: RunEvaluationConfig,
    prompt: PromptVersion,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Build the agent and run evaluation (MCP already connected on *loop*)."""
    if not isinstance(prompt.template, list):
        raise SystemExit("Error: action prompt must be a chat-message template")
    instructions = prompt.template[0]["content"]
    action_agent = Agent(
        name="action",
        instructions=instructions,
        mcp_servers=[config.mcp_server],
        model=config.llm_model,
    )

    print(f"Loaded {len(config.gold_set)} gold set entries")

    gold_set_hash = content_hash(
        json.dumps([e.model_dump() for e in config.gold_set], sort_keys=True),
    )
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        raise SystemExit(f"Error: experiment '{config.experiment_name}' not found")
    dataset = sync_gold_set(config.gold_set, gold_set_hash, experiment.experiment_id)

    predict_fn = make_predict_fn(action_agent, loop)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "prompt_name": config.prompt_name,
                "prompt_version": prompt.version,
                "model": config.llm_model,
                "gold_set_size": len(config.gold_set),
            }
        )

        eval_result = mlflow.genai.evaluate(
            predict_fn=predict_fn,
            data=dataset,
            scorers=[
                tool_usage_recall,
                Correctness(model=f"openai:/{config.llm_model}"),
                RelevanceToQuery(model=f"openai:/{config.llm_model}"),
            ],
        )

    print(f"\nEvaluation complete. Results at {mlflow.get_tracking_uri()}")
    print(f"  Metrics: {eval_result.metrics}")
