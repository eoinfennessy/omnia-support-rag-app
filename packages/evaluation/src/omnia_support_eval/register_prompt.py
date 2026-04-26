"""Register system prompts into MLflow Prompt Registry.

Usage:
    register-prompt [--prompt-path PATH] [--prompt-name NAME] [--alias ALIAS]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException


def _load_prompt(prompt_name: str, *, alias: str | None = None) -> Any | None:
    """Load a prompt from MLflow, returning None if it doesn't exist."""
    try:
        if alias:
            return mlflow.genai.load_prompt(f"prompts:/{prompt_name}@{alias}")
        return mlflow.genai.load_prompt(prompt_name)
    except MlflowException:
        return None


def _templates_equal(a: Any, b: Any) -> bool:
    """Compare two templates by their canonical JSON representation."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def register(prompt_path: Path, prompt_name: str, *, alias: str | None) -> None:
    """Read a YAML prompt file and register it in MLflow Prompt Registry.

    Idempotent: skips registration if the latest version already has the
    same template, and only updates the alias if it doesn't already point
    to the target version.
    """
    if not prompt_path.exists():
        raise SystemExit(f"Error: prompt file not found: {prompt_path}")

    raw = prompt_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise SystemExit(f"Error: prompt file is empty: {prompt_path}")

    template = yaml.safe_load(raw)
    if not isinstance(template, list) or not all(
        isinstance(m, dict) and "role" in m and "content" in m for m in template
    ):
        raise SystemExit(f"Error: {prompt_path} must be a list of messages with 'role' and 'content' keys")

    latest = _load_prompt(prompt_name)

    if latest is not None and _templates_equal(template, latest.template):
        target_version = latest.version
        print(f"Prompt '{prompt_name}' template unchanged (version {target_version})")
    else:
        registered = mlflow.genai.register_prompt(
            name=prompt_name,
            template=template,
        )
        target_version = registered.version
        print(f"Registered prompt '{prompt_name}' version {target_version}")

    if alias:
        aliased = _load_prompt(prompt_name, alias=alias)
        if aliased is not None and aliased.version == target_version:
            print(f"  Alias @{alias} already points to version {target_version}")
        else:
            mlflow.genai.set_prompt_alias(
                name=prompt_name,
                alias=alias,
                version=target_version,
            )
            print(f"  Set alias @{alias} -> version {target_version}")


def main() -> None:
    """CLI entry point for registering prompts in MLflow."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Register a system prompt in MLflow Prompt Registry",
    )
    parser.add_argument(
        "--prompt-path",
        default="prompts/omnia_support_system.yaml",
        help="Path to the prompt YAML file (default: prompts/omnia_support_system.yaml)",
    )
    parser.add_argument(
        "--prompt-name",
        default="omnia_support_system",
        help="Name for the prompt in MLflow (default: omnia_support_system)",
    )
    parser.add_argument(
        "--alias",
        default="production",
        help="Alias to assign to this version (default: production)",
    )
    parser.add_argument(
        "--no-alias",
        action="store_true",
        help="Skip setting an alias",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )

    register(
        Path(args.prompt_path),
        prompt_name=args.prompt_name,
        alias=None if args.no_alias else args.alias,
    )
