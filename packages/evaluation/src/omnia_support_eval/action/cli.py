"""CLI argument parsing and component construction for action agent evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.mcp import MCPServerSse, MCPServerStdio
from dotenv import load_dotenv

from ..common import load_yaml_config, resolve_config
from .evaluate import RunEvaluationConfig, run_evaluation
from .gold_set import load_gold_set

if TYPE_CHECKING:
    import argparse

_PACKAGE_DIR = Path(__file__).parent

_DEFAULTS: dict[str, Any] = {
    "gold_set": str(_PACKAGE_DIR / "gold_set.json"),
    "prompt_name": "action",
    "prompt_version": None,
    "prompt_alias": "production",
    "model": "gpt-5.4-mini",
    "experiment_name": "omnia-support-action-eval",
    "mcp_server_url": None,
}


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register action-agent-specific CLI arguments on *parser*."""
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (CLI args override config values)",
    )
    parser.add_argument("--gold-set", help="Path to gold set JSON file")
    parser.add_argument("--prompt-name", help="MLflow prompt registry name")

    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument("--prompt-version", type=int, help="Specific prompt version number")
    version_group.add_argument("--prompt-alias", help="Prompt alias to load")

    parser.add_argument("--model", help="LLM model for action agent")
    parser.add_argument("--experiment-name", help="MLflow experiment name")
    parser.add_argument(
        "--mcp-server-url",
        help="MCP server SSE URL (e.g. http://host:8081/sse). "
        "If omitted, spawns a local MCP server subprocess via stdio.",
    )


def main(args: argparse.Namespace) -> None:
    """Run action agent evaluation from parsed CLI arguments."""
    load_dotenv()

    yaml_config = load_yaml_config(Path(args.config) if args.config else None)
    resolve_config(args, _DEFAULTS, yaml_config)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Error: OPENAI_API_KEY environment variable is not set")

    gold_set = load_gold_set(Path(args.gold_set))

    if args.mcp_server_url:
        mcp_server = MCPServerSse(
            params={"url": args.mcp_server_url},
            name="omnia-support-tools",
        )
    else:
        mcp_server = MCPServerStdio(
            params={"command": "uv", "args": ["run", "omnia-support-mcp"]},
            name="omnia-support-tools",
        )

    config = RunEvaluationConfig(
        gold_set=gold_set,
        prompt_name=args.prompt_name,
        prompt_version=args.prompt_version,
        prompt_alias=args.prompt_alias,
        mcp_server=mcp_server,
        llm_model=args.model,
        experiment_name=args.experiment_name,
    )
    run_evaluation(config)
