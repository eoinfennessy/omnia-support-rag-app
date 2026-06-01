"""CLI router for agent evaluation subcommands."""

from __future__ import annotations

import argparse

from .information import cli as information_cli


def main() -> None:
    """CLI entry point: dispatches to per-agent evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate agents against gold sets",
    )
    subparsers = parser.add_subparsers(dest="agent", required=True)

    info_parser = subparsers.add_parser(
        "information",
        help="Evaluate the information / RAG agent",
    )
    information_cli.add_args(info_parser)

    args = parser.parse_args()

    if args.agent == "information":
        information_cli.main(args)
